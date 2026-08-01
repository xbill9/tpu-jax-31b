"""Prefill logits must not depend on the SIZE of the KV buffer they are stored into.

`prefill_with_kv_cache` documents that prefill "attends over the freshly computed
K/V (S keys), not the padded cache". If that holds, allocating a longer cache
changes only where results are *stored*, never what is *computed* — so the returned
logits must be identical for any buffer length >= S.

These tests pass, and they are here because for a while it looked like they would
not. On a TPU with the real 31B at S=1024, sliding buffers of 1032 vs 1024 gave
logits 12% apart with a different top-1 token — which reads like cache contents
leaking into attention.

It was not. Two follow-ups settled it: these tests pass at tiny scale on CPU, and
the same TPU A/B run on a REAL prompt (rather than random tokens) agrees on argmax
with a divergence of 3% against a top-1 margin of 1.565. Different buffer sizes
compile to different HLO, so fusion and accumulation order differ, and ~3% of bf16
drift accumulates over 60 layers. On random-token prompts the logits are near-flat
and any drift flips the argmax; on real input it does not.

The invariant below is still the right one to encode — it is what would break if
the cache genuinely leaked — so it stays as a regression guard. Shapes mirror the
case that raised the alarm: `sliding_window == S`, so `window_kv=True` allocates
exactly S slots (no padding) and `window_kv=False` allocates S + max_new_tokens.

See benchmarks/runs/2026-07-31-gemma4-31b-v6e1/REPORT.md, Addendum 7.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ports.gemma4.jax_e_model import (
    Gemma4EConfig,
    Gemma4EModelJAX,
    init_kv_cache,
    make_prefill_causal_mask,
    prefill_with_kv_cache,
)

WINDOW = 8


def _config(window: int = WINDOW) -> Gemma4EConfig:
    """31B-shaped miniature: mixed sliding/full, no PLE, no KV sharing, k_eq_v."""
    return Gemma4EConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_global_key_value_heads=1,
        global_head_dim=16,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=window,
        layer_types=["sliding_attention", "sliding_attention",
                     "full_attention", "sliding_attention"],
        logit_softcapping=30.0,
        attention_k_eq_v=True,
    )


def _params(cfg: Gemma4EConfig, seed: int = 0):
    key = jax.random.PRNGKey(seed)

    def rnd(*shape):
        nonlocal key
        key, sub = jax.random.split(key)
        return jax.random.normal(sub, shape, dtype=jnp.float32) * 0.05

    p = {"embed_tokens": rnd(cfg.vocab_size, cfg.hidden_size),
         "final_norm": rnd(cfg.hidden_size) + 1.0}
    for i in range(cfg.num_hidden_layers):
        sliding = cfg.layer_types[i] == "sliding_attention"
        n_kv = cfg.num_key_value_heads if sliding else cfg.num_global_key_value_heads
        h = cfg.head_dim if sliding else cfg.global_head_dim
        k = rnd(cfg.hidden_size, n_kv * h)
        p[f"layer_{i}"] = {
            "input_layernorm": rnd(cfg.hidden_size) + 1.0,
            "post_attention_layernorm": rnd(cfg.hidden_size) + 1.0,
            "pre_feedforward_layernorm": rnd(cfg.hidden_size) + 1.0,
            "post_feedforward_layernorm": rnd(cfg.hidden_size) + 1.0,
            "layer_scalar": jnp.asarray(0.6, dtype=jnp.float32),
            "attn": {"q_proj": rnd(cfg.hidden_size, cfg.num_attention_heads * h),
                     "k_proj": k, "v_proj": k,
                     "o_proj": rnd(cfg.num_attention_heads * h, cfg.hidden_size),
                     "q_norm": rnd(h) + 1.0, "k_norm": rnd(h) + 1.0},
            "mlp": {"gate_proj": rnd(cfg.hidden_size, cfg.intermediate_size),
                    "up_proj": rnd(cfg.hidden_size, cfg.intermediate_size),
                    "down_proj": rnd(cfg.intermediate_size, cfg.hidden_size)},
        }
    return p


def _prompt(cfg, B=1, S=WINDOW):
    rng = np.random.default_rng(0)
    ids = jnp.asarray(rng.integers(1, cfg.vocab_size, size=(B, S)), dtype=jnp.int32)
    return ids, jnp.ones((B, S), dtype=bool)


def _no_cache_logits(model, cfg, ids, valid):
    """Ground truth: run the model with no KV cache at all, take the last position."""
    B, S = ids.shape
    pos = jnp.arange(S, dtype=jnp.int32)[None, :].repeat(B, axis=0)
    out = model(ids, _PARAMS, pos,
                attention_mask=make_prefill_causal_mask(valid),
                quant_mode="fp16",
                sliding_attention_mask=make_prefill_causal_mask(
                    valid, window=cfg.sliding_window))
    return np.asarray(out, np.float32)[:, -1, :]


_CFG = _config()
_MODEL = Gemma4EModelJAX(_CFG)
_PARAMS = _params(_CFG)


def _prefill(window_kv, S=WINDOW, new=2):
    ids, valid = _prompt(_CFG, S=S)
    out, _, _ = prefill_with_kv_cache(
        _MODEL, ids, valid, _PARAMS, new,
        quant_mode="fp16", cache_dtype=jnp.float32, window_kv=window_kv)
    return np.asarray(out, np.float32)


def test_buffer_lengths_actually_differ():
    """Guard: the two configurations must allocate different sliding buffers.

    Without this the invariance test below could pass vacuously.
    """
    a = init_kv_cache(_CFG, 1, WINDOW + 2, jnp.float32, window_kv=False)[0][0].shape[2]
    b = init_kv_cache(_CFG, 1, WINDOW + 2, jnp.float32, window_kv=True)[0][0].shape[2]
    assert a == WINDOW + 2 and b == WINDOW, f"expected {WINDOW+2} vs {WINDOW}, got {a} vs {b}"


def test_prefill_logits_are_invariant_to_kv_buffer_size():
    """The bug, at tiny scale: padded vs exact cache must give the same logits."""
    padded = _prefill(window_kv=False)     # buffer S + new  (2 unused slots)
    exact = _prefill(window_kv=True)       # buffer S        (no unused slots)
    denom = float(np.max(np.abs(padded))) or 1.0
    rel = float(np.max(np.abs(exact - padded))) / denom
    assert rel < 1e-5, (
        f"prefill logits depend on KV buffer size: rel diff {rel:.4e}. "
        "Prefill is documented to attend over freshly computed K/V, so a longer "
        "cache must change storage only."
    )


def test_prefill_matches_a_cache_free_forward():
    """Both cache configurations must match running the model with no cache."""
    ids, valid = _prompt(_CFG, S=WINDOW)
    ref = _no_cache_logits(_MODEL, _CFG, ids, valid)
    denom = float(np.max(np.abs(ref))) or 1.0
    for label, wk in (("window_kv=False", False), ("window_kv=True", True)):
        rel = float(np.max(np.abs(_prefill(window_kv=wk) - ref))) / denom
        assert rel < 1e-5, f"{label} diverges from a cache-free forward: {rel:.4e}"


@pytest.mark.parametrize("S", [WINDOW // 2, WINDOW, WINDOW * 2])
def test_invariance_across_prompt_lengths(S):
    """Below, at, and above the sliding window."""
    padded = _prefill(window_kv=False, S=S)
    exact = _prefill(window_kv=True, S=S)
    denom = float(np.max(np.abs(padded))) or 1.0
    rel = float(np.max(np.abs(exact - padded))) / denom
    assert rel < 1e-5, f"S={S}: rel diff {rel:.4e}"
