# Gemma 4 31B QAT — what the model itself looks like

**Model:** `google/gemma-4-31B-it-qat-w4a16-ct`
**Measured:** 2026-08-01, spot v6e-1 (`us-central1-a`), pure-JAX engine
**Harness:** `ports/gemma4/jax_31b_model_intel.py` (`--passes static,dynamic`)

Everything in `REPORT.md` measures the *engine*. This measures the *model*: what its
weights say about how it was built, and what its activations do when it runs. As far
as this repository knows, none of it has been recorded for the 31B before.

---

## 1. The residual stream is clamped at both ends

Each decoder layer ends with `h *= layer_scalar`, one learned scalar per layer
applied to the whole residual stream. Across the 60 layers:

```
mean 0.7823   min 0.0317 (layer 59)   max 0.9922 (layer 2)
depth 0->59: ▁▁█▇▇▇▇▇▇▇▇▆▅▇▇▆▇▆▇▇▇▇▇▅▆▆▅▅▄▄▅▆▆▆▆▅▆▆▆▅▅▅▆▆▆▇▇▇▇▆▇▇▆▇▇▇▇▆▅▁
```

| depth | value | effect |
| :--- | ---: | :--- |
| L0 | 0.0986 | **~10x suppression** |
| L1 | 0.0732 | **~14x suppression** |
| L2–L4 | 0.9922, 0.9727, 0.9844 | essentially pass-through |
| L28–L32 | 0.4551 → 0.7734 | mid-stack dip |
| L57–L58 | 0.7930, 0.6953 | taper |
| **L59** | **0.0317** | **~31x suppression** |

This is a deliberate three-phase schedule: **clamp hard on entry, run nearly free
through the body, clamp hard on exit.** The first two and the last layer are doing
something categorically different from the other 57.

**Full-attention layers are damped harder than sliding ones**: mean `layer_scalar`
0.6891 vs 0.8010, a 14% difference that is consistent across depth.

## 2. Why the damping exists: the norms are enormous

RMSNorm weight magnitude, averaged over all 60 layers:

| norm | mean abs | max abs |
| :--- | ---: | ---: |
| `input_layernorm` | **23.475** | **1248.0** |
| `pre_feedforward_layernorm` | **40.797** | 860.0 |
| `post_attention_layernorm` | 0.473 | 19.625 |
| `post_feedforward_layernorm` | 0.737 | 63.0 |

The **pre**-norms carry gains of 23–41 on average with individual channels up to
**1248**; the **post**-norms are sub-unit. So each block amplifies hugely going in
and is scaled back coming out, with `layer_scalar` as the final counterweight. A
single channel at 1248 is the classic massive-activation / attention-sink signature,
and it is a standing hazard for any quantization scheme that assumes a well-behaved
per-tensor range.

This also confirms the engine's own comment that `layer_scalar` "is the counterweight
to this checkpoint's large RMSNorm weights" — the magnitudes above are what that
comment is about, now quantified.

## 3. The stream SHRINKS with depth — 8x

Residual RMS entering each layer, one real chat-templated prompt (20 tokens):

```
L0    L6    L12   L18   L24   L30   L36   L42   L48   L54
10.06 4.25  0.81  1.14  2.71  2.74  2.10  1.25  1.06  0.89
```

**10.06 at layer 0 → 1.239 at layer 59: 0.12x.** Most transformers grow the residual
stream with depth; this one contracts it by 8x, with a hump around layers 24–30. The
entry clamp (L0/L1) does most of the initial collapse: 10.06 → 4.25 by layer 6.

Early layers are where the stream is restructured rather than refined — attention
output *exceeds* the residual it is added to at layers 2, 3 and 4
(`attn/resid` = 1.92, 1.85, 2.32), which does not happen again until very deep.

## 4. Full-attention layers punch above their weight

The 10 full-attention layers sit at **i ≡ 5 (mod 6)** — indices 5, 11, 17, 23, 29,
35, 41, 47, 53, **59**. Note the *last* layer of the model is a full-attention layer.

| | sliding (50) | full (10) | ratio |
| :--- | ---: | ---: | ---: |
| `attn_out` RMS | 1.412 | **1.931** | 1.37x |
| `attn_out / resid_in` | 0.7253 | **0.9676** | 1.33x |
| `layer_scalar` | 0.8010 | **0.6891** | 0.86x |

Full-attention layers produce **33–37% larger** attention outputs relative to the
stream, and are damped **14% harder** to compensate. They are 1/6 of the layers and
are structurally the load-bearing ones — which is notable given they are also the
layers with only **4** KV heads (vs 16) and `attention_k_eq_v` (V aliased to K, no
`v_proj` at all).

## 5. Quantization stress: where W4A16 hurts most

Group-scale dynamic range (p99.9 / median of |scale|). Higher = more outlier
channels within the tensor = harder for a 4-bit grid to represent.

| projection | all | sliding | full | early (0–19) | late (40–59) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **v_proj** | **4.22** | 4.22 | *n/a* | 3.96 | **4.61** |
| k_proj | 3.56 | 3.58 | 3.49 | 3.06 | 3.67 |
| o_proj | 3.44 | 3.46 | 3.37 | 3.66 | 3.34 |
| q_proj | 3.25 | 3.27 | 3.16 | 2.78 | 3.37 |
| down_proj | 3.13 | 3.14 | 3.06 | **3.56** | 2.42 |
| gate_proj | 2.75 | 2.75 | 2.78 | 2.31 | 2.79 |
| up_proj | 2.41 | 2.40 | 2.46 | 2.22 | 2.19 |

Three things fall out:

1. **`v_proj` is the hardest tensor to quantize in the model** (4.22), and it gets
   worse with depth (3.96 → 4.61). The `n/a` in the "full" column is not missing
   data — it is `attention_k_eq_v`: full-attention layers **ship no `v_proj` at
   all**. The architecture is confirmed by the absence.
2. **Attention projections get harder with depth; the MLP does not.** q/k/v all rise
   from early to late, while `down_proj` *falls* (3.56 → 2.42) and `up_proj` is flat.
   If you were choosing where to spend bits, late-layer attention is the place.
3. **MLP projections are the easiest** (`up_proj` 2.41, `gate_proj` 2.75) despite
   being by far the largest tensors — good news, since they dominate the footprint.

## 6. The model is extremely confident on real prompts

Final-position logits for `"What is the capital of France?"`:

| metric | value |
| :--- | ---: |
| logit RMS | 6.026 |
| max logit | 27.9 |
| **top-1 to top-2 margin** | **15.28** |
| top-1 probability | 1.0000 |
| top-10 mass | 1.0000 |
| entropy | **0.000 nats** |
| fraction at softcap (30.0) | 0.000000 |

The distribution is effectively a point mass. Two consequences:

- **This quantitatively closes the false alarm in `REPORT.md` Addendum 7.** Fusion
  reordering between `window_kv` settings perturbs logits by ~0.69. Against a top-1
  margin of **15.28**, that is a 22x safety factor and argmax cannot flip. On
  random-token prompts the margin collapses toward zero and the same perturbation
  flips it — which is exactly what was measured and misread as a correctness bug.
  **Greedy decoding on this model is stable on real input and meaningless to
  evaluate on out-of-distribution input.**
- The max logit of 27.9 against a 30.0 softcap means `tanh` is operating well into
  its saturating region, even though no logit is fully clipped. The softcap is
  shaping this distribution, not merely bounding it.

---

## 7. Massive activations: 15,665x outliers on the first two tokens

`ports/gemma4/jax_31b_massive_act.py`. The huge norm weights in §2 predicted a
massive-activation structure; this is it, and it is extreme.

### Runtime: where the magnitude lands

Largest `|activation|` in the residual stream entering each layer, one 20-token
chat-templated prompt:

| layer | max\|h\| | ratio to median | at position | token | channel |
| ---: | ---: | ---: | ---: | :--- | ---: |
| 0 | 860.0 | 330 | 1 | `'<\|turn>'` | 3970 |
| 2 | 281.0 | 2,217 | 0 | `'<bos>'` | 1682 |
| 3 | 499.6 | 4,459 | 0 | `'<bos>'` | 1682 |
| 10 | 788.2 | 2,678 | 0 | `'<bos>'` | 1682 |
| 30 | 403.6 | 517 | 5 | `' is'` | 3770 |
| 50 | 169.8 | 210 | 1 | `'<\|turn>'` | 3770 |
| **59** | **18.33** | **32** | 4 | `'What'` | 3069 |

**max/median ratio: mean 1,214x, peak 15,665x at layer 7.**

**Position concentration — the attention-sink test, passed:**

| position | token | layers holding the max |
| ---: | :--- | ---: |
| 1 | `'<\|turn>'` | **32 / 60** |
| 0 | `'<bos>'` | **15 / 60** |
| 5 | `' is'` | 5 / 60 |
| 6 | `' the'` | 5 / 60 |

**78% of layers put their peak activation on one of the first two tokens.**

**Channel concentration:** ch3770 (23/60) and ch1682 (11/60) account for 57% of
layers. ch1682 owns the early stack (layers 2–10), ch3770 the middle and late.

Note that mean `|h|` at position 0 is only **1.3x** the other positions. The
positions are not uniformly large — a *few channels* at those positions are
enormous. Massive activations are sparse in (position, channel) space, which is
exactly why a per-tensor scale cannot survive them.

### The exit clamp is what tames them

Layer 59 peaks at **18.33** with a ratio of 32, against ~100–800 and ratios in the
hundreds-to-thousands everywhere else. That is §1's `layer_scalar = 0.0317` doing
its job: **the exit clamp exists to crush the massive activations before the
lm_head sees them.** The two findings are one mechanism.

### The norm outliers and the activation outliers are DIFFERENT channels

Static top channels by norm weight — pre-norms: ch1081, ch1400, ch1924, ch4206
(recurring across both `input_layernorm` and `pre_feedforward_layernorm`);
post-norms: a tight contiguous band at **ch33–47**, with ch39 the argmax in 12/60
and 16/60 layers.

Runtime top channels: ch3770, ch1682, ch2130, ch3067.

**They do not overlap.** The high-gain norm channels and the massive-activation
channels are separate populations, so you cannot predict one from the other — the
norm weights are not simply amplifying the sink channels.

Structural aside: 13.9% of `input_layernorm` and 23.0% of `pre_feedforward_layernorm`
channels sit above 10x the median, versus **0.13–0.17%** for the post-norms. The
pre-norms carry a broad heavy tail; the post-norms carry a handful of outliers in a
narrow low index band. Two different designs in the same layer.

### Practical consequences

1. **Per-tensor activation quantization (e.g. A8) will not survive this model.** A
   15,665x in-tensor dynamic range leaves nothing for the other channels.
2. **The engine's KV quantization is already the right shape, and now there is a
   reason.** `quantize_kv` uses **per-(batch, head, position)** scales over
   `head_dim` (jax_e_model.py:473). Per-position granularity is precisely what
   isolates the sink tokens; a per-tensor scale would be destroyed by them. This
   validates the existing int8/fp8 KV support rather than merely permitting it.
3. **Do not trim or drop the leading tokens.** `<bos>` and `<\|turn>` carry the
   sinks for 78% of layers. Anything that truncates a prompt's head, or evicts
   early cache slots, is removing the model's attention sink.
4. **Windowed sliding KV is safe here only because the sinks live in the last
   1024 positions of a short prompt.** At long context, a 1024-slot ring on the
   sliding layers *will* evict positions 0–1. Whether those layers depend on the
   sink is untested and is the obvious follow-up — it bears directly on the
   long-context work in `REPORT.md`.

---

## Reproduce

```bash
JAX_PLATFORMS=cpu python3.13 ports/gemma4/jax_31b_model_intel.py --passes static
python3.13 ports/gemma4/jax_31b_model_intel.py --passes dynamic
```

The static pass needs no accelerator. It also carries a small dependency-free
safetensors reader, because the `safetensors` build on a bare JAX VM cannot decode
`bfloat16` through *any* framework — numpy and flax both raise
`data type 'bfloat16' not understood`, and torch is not installed.

## Caveats

- Activation figures come from **one** 20-token chat-templated prompt. Residual
  magnitudes and the confidence metrics will move with prompt length and content;
  the sliding-vs-full *ratios* are the more robust part.
- Scale dynamic range is a proxy for quantization difficulty, not a measurement of
  quantization error. Confirming it needs a dequantize-and-compare against the
  `q4_0-unquantized` variant, which was not run.
- Everything here describes the **W4A16 QAT** checkpoint. The bf16 variants may
  differ, particularly in the norm outliers.

## Artifacts

- `results/gemma4_31b_model_intel.json` — static pass (norms, layer_scalar, quant stress)
- `results/gemma4_31b_model_intel_dyn.json` — dynamic pass, full 60-layer table
- `results/gemma4_31b_massive_act.json` — norm channel structure
- `results/gemma4_31b_massive_act_dyn.json` — per-layer max activation, position, channel
- `logs/intel_static.log`, `logs/intel_dyn.log`, `logs/mact_static.log`, `logs/mact_dyn.log`
- `jax_31b_model_intel.py`, `jax_31b_massive_act.py` — the harnesses
