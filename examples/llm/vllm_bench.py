#!/usr/bin/env python3
"""vLLM batch-1 greedy decode timing on exactly the prompt ETX uses (the prompt ids in a golden file).

Same method as fleet-mi300x/bench/vllm_decode_timing.py: per-token decode cost =
(t(N new tokens) - t(1 new token)) / (N - 1), after a warm-up, so prefill and the first sample cancel.
Run inside the rocm/vllm container; VLLM_ROCM_USE_AITER=1 selects AITER kernels (the "best kernels" run).

  python3 examples/llm/vllm_bench.py --model ~/llm/qwen3-8b/hf --golden ~/llm/qwen3-8b/golden_c1024.txt --label qwen3-8b

--batch N decodes N sequences together: sequence 0 is the golden prompt (checked against the golden tokens), sequence i
the same prompt rotated by 97*i tokens, so every sequence has the same length but different content (a MoE routes them
to different experts, as real concurrent requests would). The reported step time is per decode step of the whole batch.

--goldens G0 G1 ... --batches 1,2,4,8 (docs/BATCH-2026-10-14.md): sequence r's prompt and golden ids come from its own
file Gr, and ONE engine measures every listed batch size in turn (sequences 0..B-1). Every sequence is checked on the
warm-up and on every sample; nothing is filtered or stops the run. Per sequence and phase one `VLLMSEQ` line, per batch
size one `VLLMB` line with the per-sample decode cost of the whole batch (etx/session/batch.py reads both).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


CONFIG_PREFIX = "VLLM-CONFIG "


def encode_config(config):
    return CONFIG_PREFIX + json.dumps(config, default=str, sort_keys=True)


def decode_config(line):
    if not line.startswith(CONFIG_PREFIX):
        raise ValueError("missing VLLM config prefix")
    return json.loads(line[len(CONFIG_PREFIX):])


def read_golden(path: Path):
    lines = path.read_text().splitlines()
    prompt = [int(x) for x in lines[0].split()[1:]]
    gen = [int(x) for x in lines[1].split()[1:]]
    return prompt, gen


def kw_of(a) -> dict:
    cc = ({"custom_ops": [a.custom_ops]} if a.custom_ops else {}) | ({"pass_config": json.loads(a.pass_config)} if a.pass_config else {}) \
         | ({"cudagraph_mode": a.cudagraph_mode} if a.cudagraph_mode else {}) | ({"mode": a.compile_mode} if a.compile_mode is not None else {})
    kw = dict(a.engine_kw)
    if cc:
        kw["compilation_config"] = dict(kw.get("compilation_config", {})) | cc
    return kw


def first_divergence(golden: list[int], toks: list[int], n: int) -> int:
    """How many leading tokens of the first n agree (n = all of them)."""
    return next((i for i in range(n) if i >= len(toks) or i >= len(golden) or golden[i] != toks[i]), n)


def batch_lines(label: str, b: int, phase: str, goldens: list[list[int]], outs: list[list[int]], n: int) -> list[str]:
    return [f"VLLMSEQ label={label} batch={b} phase={phase} seq={r} first_divergence={first_divergence(goldens[r], toks, n)}/{n} "
            f"tokens={len(toks)}" for r, toks in enumerate(outs)]


def batch_mode(a, kw) -> None:
    """One engine, every batch size in --batches; per-sequence goldens; decode cost (t(N) - t(1)) / (N - 1) per sample."""
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    files = [read_golden(g) for g in a.goldens]
    batches = [int(x) for x in a.batches.split(",")]
    if max(batches) > len(files) or min(batches) < 1:
        raise SystemExit(f"batch sizes {batches} need {max(batches)} golden files, got {len(files)}")
    plen = len(files[0][0])
    if any(len(p) != plen for p, _ in files):
        raise SystemExit("the sequences' prompts differ in length")
    if len({tuple(p) for p, _ in files}) != len(files):
        raise SystemExit("two golden files hold the same prompt")
    llm = LLM(model=a.model, dtype="bfloat16", trust_remote_code=True, max_model_len=plen + a.tokens + 16,
              gpu_memory_utilization=0.85, enforce_eager=a.eager, **kw)        # vLLM's defaults: max_num_seqs >= 8 already
    if a.echo_config:
        config = llm.llm_engine.vllm_config
        print(encode_config({"cudagraph_mode": config.compilation_config.cudagraph_mode,
                             "attention_backend": config.attention_config.backend,
                             "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("VLLM_", "HIP_"))}}), flush=True)
    print(f"VLLM-BATCH-MODE label={a.label} version={vllm.__version__} context={plen} batches={a.batches} "
          f"aiter={os.environ.get('VLLM_ROCM_USE_AITER', '0')}", flush=True)
    result = {}
    for b in batches:
        prompts = [TokensPrompt(prompt_token_ids=p) for p, _ in files[:b]]
        goldens = [g for _, g in files[:b]]

        def gen(n_new: int):
            sp = SamplingParams(temperature=0.0, max_tokens=n_new, ignore_eos=True, detokenize=not a.no_detok)
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            return (time.perf_counter() - t0) * 1e3, [list(o.outputs[0].token_ids) for o in outs]

        print(f"VLLM-PROGRESS label={a.label} batch={b} warm-up", flush=True)
        _, outs = gen(a.tokens)
        print("\n".join(batch_lines(a.label, b, "warmup", goldens, outs, a.tokens)), flush=True)
        _, short = gen(1)
        print("\n".join(batch_lines(a.label, b, "warmup_short", goldens, short, 1)), flush=True)
        per = []
        for k in range(a.repeats):
            full, outs = gen(a.tokens)
            one, short = gen(1)
            per.append((full - one) / (a.tokens - 1))
            print("\n".join(batch_lines(a.label, b, f"sample{k}", goldens, outs, a.tokens)), flush=True)
            print("\n".join(batch_lines(a.label, b, f"sample{k}_short", goldens, short, 1)), flush=True)
        print(f"VLLMB label={a.label} batch={b} per_token_ms={','.join(f'{x:.6f}' for x in per)}", flush=True)
        result[b] = per
    if a.json:
        a.json.write_text(json.dumps({"label": a.label, "version": vllm.__version__, "context": plen,
                                      "aiter": os.environ.get("VLLM_ROCM_USE_AITER", "0"),
                                      "per_token_ms": {str(b): v for b, v in result.items()}}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-kw", type=json.loads, default={}, help="extra LLM keyword arguments as JSON")
    ap.add_argument("--no-detok", action="store_true")
    ap.add_argument("--echo-config", action="store_true")
    ap.add_argument("--gate-warmup", action="store_true", help="check the full warmup before timing; print burst progress outside timers")
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", type=Path, default=None)
    ap.add_argument("--goldens", type=Path, nargs="+", default=None, help="one golden file per sequence (batch mode)")
    ap.add_argument("--batches", default="1", help="batch mode: the batch sizes one engine measures, e.g. 1,2,4,8")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=1, help="sequences decoded together (distinct prompts of equal length)")
    ap.add_argument("--custom-ops", default=None, help="'all': vLLM's own HIP ops instead of Inductor-generated ones (the kernels ETX imports)")
    ap.add_argument("--pass-config", default=None, help='compilation_config pass_config as JSON, e.g. {"enable_qk_norm_rope_fusion": true}')
    # vLLM's own fusion ladder: --eager drops torch.compile AND the graphs (one launch per op); --cudagraph-mode NONE
    # keeps the compiled/fused kernels but drops the graph replay; --compile-mode 0 keeps the graphs but drops the
    # Inductor fusion. They are the analogues of ETX's fused-vs-unfused pair, so all of it lands on one scale.
    ap.add_argument("--eager", action="store_true", help="enforce_eager: no torch.compile, no cuda graphs")
    ap.add_argument("--cudagraph-mode", default=None, help="NONE | PIECEWISE | FULL | FULL_DECODE_ONLY | FULL_AND_PIECEWISE")
    ap.add_argument("--compile-mode", type=int, default=None, help="CompilationMode: 0 NONE, 1 STOCK_TORCH_COMPILE, 2 DYNAMO_TRACE_ONCE, 3 VLLM_COMPILE")
    a = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if (a.golden is None) == (a.goldens is None):
        ap.error("give --golden (one prompt) or --goldens (one file per sequence)")
    if a.goldens:
        return batch_mode(a, kw_of(a))
    prompt_ids, golden = read_golden(a.golden)
    kw = kw_of(a)
    llm = LLM(model=a.model, dtype="bfloat16", trust_remote_code=True,
              max_model_len=len(prompt_ids) + a.tokens + 16, gpu_memory_utilization=0.85, enforce_eager=a.eager, **kw)
    if a.echo_config:
        config = llm.llm_engine.vllm_config
        print(encode_config({
            "async_scheduling": config.scheduler_config.async_scheduling,
            "cudagraph_mode": config.compilation_config.cudagraph_mode,
            "pass_config": config.compilation_config.pass_config,
            "kernel_config": config.kernel_config,
            "attention_backend": config.attention_config.backend,
            "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("VLLM_", "HIP_"))},
        }), flush=True)
    print(f"VLLM-SAMPLING detok={int(not a.no_detok)}", flush=True)
    plen = len(prompt_ids)
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids[(97 * i) % plen:] + prompt_ids[:(97 * i) % plen]) for i in range(a.batch)]

    def gen(n_new: int):
        sp = SamplingParams(temperature=0.0, max_tokens=n_new, ignore_eos=True, detokenize=not a.no_detok)
        if a.gate_warmup: print(f"VLLM-GENERATE-BEGIN tokens={n_new}", flush=True)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        elapsed = (time.perf_counter() - t0) * 1e3
        if a.gate_warmup: print(f"VLLM-GENERATE-END tokens={n_new}", flush=True)
        return elapsed, list(outs[0].outputs[0].token_ids)

    _, warm_tokens = gen(a.tokens)                         # warm-up (graph capture, autotuning)
    if a.gate_warmup:
        if warm_tokens != golden[:a.tokens] or len(warm_tokens) != a.tokens:
            raise RuntimeError("VLLM warmup token gate failed; timing not started")
        print(f"VLLM-WARMUP-GATE OK {a.tokens}/{a.tokens}", flush=True)
    gen(1)
    full, one, toks = [], [], []
    for sample in range(a.repeats):
        dt, toks = gen(a.tokens); full.append(dt)
        d1, _ = gen(1); one.append(d1)
        matched = next((i for i, (g, t) in enumerate(zip(golden, toks)) if g != t), min(len(golden), len(toks)))
        print(f"VLLM-SAMPLE sample={sample} first_divergence={matched}/{len(toks)}", flush=True)
    per_tok = [(f - o) / (a.tokens - 1) for f, o in zip(full, one)]
    med = statistics.median(per_tok)
    n = min(len(golden), len(toks))
    match = next((i for i in range(n) if golden[i] != toks[i]), n)
    aiter = os.environ.get("VLLM_ROCM_USE_AITER", "0")
    print(f"VLLM label={a.label} aiter={aiter} custom_ops={a.custom_ops} eager={int(a.eager)} cudagraph={a.cudagraph_mode} "
          f"compile_mode={a.compile_mode} batch={a.batch} version={vllm.__version__} "
          f"context={len(prompt_ids)} median_ms={med:.3f} tokens_per_s={a.batch * 1e3 / med:.0f} "
          f"per_token={[round(x, 3) for x in per_tok]} first_divergence={match}/{n}")
    if a.json:
        a.json.write_text(json.dumps({"label": a.label, "aiter": aiter, "batch": a.batch, "version": vllm.__version__, "context": len(prompt_ids),
                                      "eager": a.eager, "cudagraph_mode": a.cudagraph_mode, "compile_mode": a.compile_mode, "custom_ops": a.custom_ops,
                                      "median_ms": med, "per_token_ms": per_tok, "full_ms": full, "one_ms": one,
                                      "tokens_out": toks, "golden_prefix_match": match}, indent=1))


if __name__ == "__main__":
    main()
