#!/usr/bin/env python3
"""vLLM batch-1 greedy decode timing on exactly the prompt ETX uses (the prompt ids in a golden file).

Same method as fleet-mi300x/bench/vllm_decode_timing.py: per-token decode cost =
(t(N new tokens) - t(1 new token)) / (N - 1), after a warm-up, so prefill and the first sample cancel.
Run inside the rocm/vllm container; VLLM_ROCM_USE_AITER=1 selects AITER kernels (the "best kernels" run).

  python3 examples/llm/vllm_bench.py --model ~/llm/qwen3-8b/hf --golden ~/llm/qwen3-8b/golden_c1024.txt --label qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def read_golden(path: Path):
    lines = path.read_text().splitlines()
    prompt = [int(x) for x in lines[0].split()[1:]]
    gen = [int(x) for x in lines[1].split()[1:]]
    return prompt, gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--custom-ops", default=None, help="'all': vLLM's own HIP ops instead of Inductor-generated ones (the kernels ETX imports)")
    a = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    prompt_ids, golden = read_golden(a.golden)
    kw = {"compilation_config": {"custom_ops": [a.custom_ops]}} if a.custom_ops else {}
    llm = LLM(model=a.model, dtype="bfloat16", trust_remote_code=True,
              max_model_len=len(prompt_ids) + a.tokens + 16, gpu_memory_utilization=0.85, enforce_eager=False, **kw)
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)

    def gen(n_new: int):
        sp = SamplingParams(temperature=0.0, max_tokens=n_new, ignore_eos=True)
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)[0]
        return (time.perf_counter() - t0) * 1e3, list(out.outputs[0].token_ids)

    gen(a.tokens); gen(1)                                   # warm-up (graph capture, autotuning)
    full, one, toks = [], [], []
    for _ in range(a.repeats):
        dt, toks = gen(a.tokens); full.append(dt)
        d1, _ = gen(1); one.append(d1)
    per_tok = [(f - o) / (a.tokens - 1) for f, o in zip(full, one)]
    med = statistics.median(per_tok)
    n = min(len(golden), len(toks))
    match = next((i for i in range(n) if golden[i] != toks[i]), n)
    aiter = os.environ.get("VLLM_ROCM_USE_AITER", "0")
    print(f"VLLM label={a.label} aiter={aiter} custom_ops={a.custom_ops} version={vllm.__version__} context={len(prompt_ids)} "
          f"median_ms={med:.3f} per_token={[round(x, 3) for x in per_tok]} first_divergence={match}/{n}")
    if a.json:
        a.json.write_text(json.dumps({"label": a.label, "aiter": aiter, "version": vllm.__version__, "context": len(prompt_ids),
                                      "median_ms": med, "per_token_ms": per_tok, "full_ms": full, "one_ms": one,
                                      "tokens_out": toks, "golden_prefix_match": match}, indent=1))


if __name__ == "__main__":
    main()
