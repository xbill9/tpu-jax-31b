"""Repeated real-prompt stability check for Gemma 4 31B W4A16 on TPU.

Loads the checkpoint once, cycles through several prompt buckets, and requires
greedy token output to remain identical across cycles.  This specifically tests
repeated KV-cache allocation/use on one long-lived engine rather than repeatedly
starting fresh Python processes.
"""

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
from transformers import AutoTokenizer

from ports.gemma4.jax_31b_port import DEFAULT_MODEL, Streaming31BEngine


PROMPTS = (
    "What is the capital of France?",
    "Return only the integer result of 37 * 19.",
    "Explain in two sentences why the sky appears blue during the day.",
    "Write a Python function that returns the nth Fibonacci number, then state its time complexity.",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--json-out", default="gemma4_31b_burnin.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=4096, window_kv=True)
    eng.load()
    eng.bos_token_id = tok.bos_token_id
    eos_ids = {i for i in (tok.eos_token_id,) if i is not None}
    for name in ("<turn|>", "<end_of_turn>", "<eos>"):
        tid = tok.convert_tokens_to_ids(name)
        if tid is not None and tid != tok.unk_token_id:
            eos_ids.add(tid)

    baseline = {}
    rows = []
    failures = []
    cycle_hbm = []
    start_mem = eng.memory_stats()["hbm_bytes_in_use"]
    started = time.perf_counter()
    for cycle in range(args.cycles):
        for prompt_index, prompt in enumerate(PROMPTS):
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            )
            ids = tok(rendered, add_special_tokens=False,
                      return_tensors="np")["input_ids"][0].tolist()
            t0 = time.perf_counter()
            tokens, stats = eng.generate(
                ids, max_new_tokens=args.max_new_tokens, temperature=0.0,
                eos_token_ids=sorted(eos_ids),
            )
            wall = time.perf_counter() - t0
            tokens = [int(x) for x in tokens]
            key = str(prompt_index)
            if key not in baseline:
                baseline[key] = tokens
            elif tokens != baseline[key]:
                failures.append({"cycle": cycle, "prompt_index": prompt_index,
                                 "reason": "nondeterministic_tokens"})
            rate = float(stats.decode_tok_per_s)
            prefill = float(stats.prefill_ms)
            if not math.isfinite(rate) or not math.isfinite(prefill):
                failures.append({"cycle": cycle, "prompt_index": prompt_index,
                                 "reason": "nonfinite_stats"})
            rows.append({
                "cycle": cycle,
                "prompt_index": prompt_index,
                "prompt_tokens": len(ids),
                "completion_tokens": len(tokens),
                "output": tok.decode(tokens),
                "prefill_ms": round(prefill, 3),
                "decode_tok_s": round(rate, 3),
                "wall_s": round(wall, 3),
            })
            print(f"cycle={cycle} prompt={prompt_index} tokens={len(tokens)} "
                  f"prefill={prefill:.1f}ms decode={rate:.2f}tok/s")
        jax.effects_barrier()
        cycle_bytes = eng.memory_stats()["hbm_bytes_in_use"]
        cycle_hbm.append(cycle_bytes)
        print(f"cycle={cycle} HBM={cycle_bytes / 1e9:.4f} GB")

    jax.effects_barrier()
    end_mem = eng.memory_stats()["hbm_bytes_in_use"]
    out = {
        "model": args.model,
        "jax": jax.__version__,
        "libtpu": importlib.metadata.version("libtpu"),
        "devices": [str(d) for d in jax.devices()],
        "cycles": args.cycles,
        "requests": len(rows),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "start_hbm_gb": round(start_mem / 1e9, 4),
        "end_hbm_gb": round(end_mem / 1e9, 4),
        "hbm_delta_mb": round((end_mem - start_mem) / 1e6, 3),
        "post_warmup_hbm_delta_mb": round(
            (cycle_hbm[-1] - cycle_hbm[0]) / 1e6, 3
        ) if len(cycle_hbm) > 1 else 0.0,
        "cycle_hbm_gb": [round(x / 1e9, 4) for x in cycle_hbm],
        "failures": failures,
        "rows": rows,
    }
    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"HBM delta: {out['hbm_delta_mb']:.3f} MB")
    print(f"post-warmup HBM delta: {out['post_warmup_hbm_delta_mb']:.3f} MB")
    print(f"failures: {len(failures)}")
    print(f"Wrote {args.json_out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
