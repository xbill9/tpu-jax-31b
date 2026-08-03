# Gemma 4 31B QAT smoke and stability validation — libtpu 0.0.45

**Date:** 2026-08-02  
**VM:** `gemma4-31b-qat-v6e1`, `asia-northeast1-b`  
**Hardware:** Cloud TPU v6e-1 (`ct6e-standard-1t`)  
**Model:** `google/gemma-4-31B-it-qat-w4a16-ct`  
**Stack:** JAX 0.11.0, jaxlib 0.11.0, libtpu 0.0.45, Python 3.13

## Result

The 31B W4A16 checkpoint loads, fits, generates correct smoke-test answers, and is
stable on a single v6e-1 with libtpu 0.0.45. The latest runtime produced no new
correctness, memory, or performance regression in the tests below.

This does **not** close two pre-existing engine findings:

1. The sliced-logits prefill optimization is argmax-equivalent at the three tested
   shapes but not bitwise identical to the legacy full-logits path.
2. Decode HLO still contains three full float32 embedding-table operations totaling
   16.911 GB of instruction output shapes. This is a performance opportunity, not a
   smoke-test failure.

## Evidence

### Load and parity

- Resident weights/HBM after load: 19.30 GB of 33.55 GB.
- Templated prompt: `The capital of France is Paris.`
- Bare prompt: recovered the channel scaffold and returned the same answer.
- Both stopped on an expected stop token.
- Warm decode: 12.97 tok/s.

### Multi-context soak

| Context | Prefill | Decode step | Result |
|---:|---:|---:|:---|
| 512 | 241.26 ms | 78.807 ms | OK |
| 1,024 | 292.79 ms | 78.899 ms | OK |
| 2,048 | 480.02 ms | 78.453 ms | OK |
| 4,096 | 1,058.17 ms | 78.530 ms | OK |

Decode spread was 0.446 ms (0.57%) across the tested contexts.

### Long-lived-engine burn-in

`ports/gemma4/jax_31b_burnin.py` loaded one engine and issued 40 requests: four
real prompts over ten cycles, greedy decoding up to 64 tokens.

- Token nondeterminism: **0 failures**.
- Non-finite timing/stat values: **0 failures**.
- Warm prefill range: approximately 216.0–217.5 ms.
- Warm decode range: 12.99–13.24 tok/s.
- HBM after every completed cycle: **19.3444 GB**.
- Post-warmup HBM growth: **0.000 MB**.

The 47.501 MB difference between post-load HBM and cycle 0 is one-time compiled
executable residency; it does not continue growing.

### Numerical review

| Shape | Bitwise equal | Max abs diff | Same argmax |
|:---|:---:|---:|:---:|
| B=1, S=512 | no | 3.702e-2 | yes |
| B=1, S=1,024 | no | 3.696e-2 | yes |
| B=2, S=1,024 | no | 5.722e-6 | yes |

Decode compilation contains three `f32[262144,5376]` instructions (broadcast,
convert, multiply), 5.637 GB each. The executable reports 0.146 GB temporary
memory, so aggregate HLO instruction shapes must not be misread as simultaneously
resident buffers.

## Artifacts

- `results/libtpu-0.0.45-parity.json`
- `results/libtpu-0.0.45-soak.json`
- `results/libtpu-0.0.45-review.json`
- `results/libtpu-0.0.45-burnin.json`
- `results/libtpu-0.0.45-burnin-10x.json`

## Reproduction

```bash
python3.13 ports/gemma4/jax_31b_port.py \
  --stages load,parity --parity-tokens 64

python3.13 ports/gemma4/jax_31b_port.py \
  --stages load,sweep --batch-sizes 1 \
  --contexts 512,1024,2048,4096 --decode-steps 64 --repeats 3

python3.13 ports/gemma4/jax_31b_review_checks.py

python3.13 ports/gemma4/jax_31b_burnin.py \
  --cycles 10 --max-new-tokens 64
```
