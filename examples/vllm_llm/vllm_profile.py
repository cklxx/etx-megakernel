#!/usr/bin/env python3
"""How much of vLLM's decode time is kernels, and how much is the space between them.

  run:      rocprofv3 --kernel-trace --output-format csv -d /tmp/prof -- \
              python3 examples/vllm_llm/vllm_profile.py run --model M --golden G [--custom-ops all]
            warm-up, then generate(1) and generate(N+1) with 0.2 s of idle between the calls, so the kernel
            trace splits into one burst per call
  analyze:  python3 examples/vllm_llm/vllm_profile.py analyze /tmp/prof [--tokens N]
            per decode token = (burst(N+1) - burst(1)) / N, for the GPU span (first kernel start to last kernel
            end), the busy time (union of kernel intervals) and each kernel name's time and count. The gap
            (span - busy) is what launching kernels one after another costs on the GPU: the upper bound on what
            running the same kernels inside one persistent kernel can save.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from collections import defaultdict
from pathlib import Path


def run(a):
    import os
    # vLLM V1 runs the engine core in a child process by default; the kernel tracer follows the process it
    # launched, so keep the engine in this one (the 2026-09-26 traces came back empty)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    lines = Path(a.golden).read_text().splitlines()
    prompt = [int(x) for x in lines[0].split()[1:]]
    kw = {"compilation_config": {"custom_ops": [a.custom_ops]}} if a.custom_ops else {}
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=len(prompt) + a.tokens + 16, gpu_memory_utilization=0.85, **kw)
    p = TokensPrompt(prompt_token_ids=prompt)

    def gen(n):
        return llm.generate([p], SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True), use_tqdm=False)[0].outputs[0].token_ids

    gen(a.tokens); gen(1)
    time.sleep(0.2); gen(1)
    time.sleep(0.2); out = gen(a.tokens + 1)
    time.sleep(0.2)
    print("PROFILE-RUN-DONE tokens", list(out)[:8])


def load(d: str):
    files = glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True)
    if not files:
        raise SystemExit(f"no kernel_trace.csv under {d}")
    ks = []
    for f in files:
        with open(f) as h:
            for r in csv.DictReader(h):
                ks.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    ks.sort()
    return ks


def bursts(ks, gap_ns=100_000_000):
    out, cur = [], [ks[0]]
    for k in ks[1:]:
        if k[0] - max(x[1] for x in cur[-4:]) > gap_ns:
            out.append(cur); cur = []
        cur.append(k)
    out.append(cur)
    return out


def stats(b):
    span = b[-1][1] - b[0][0]
    busy, end = 0, 0
    for s, e, _ in b:
        if e <= end:
            continue
        busy += e - max(s, end); end = e
    per = defaultdict(lambda: [0, 0])
    for s, e, n in b:
        per[n][0] += e - s; per[n][1] += 1
    return span, busy, per


def analyze(a):
    ks = load(a.dir)
    bs = bursts(ks)
    if len(bs) < 2:
        raise SystemExit(f"{len(bs)} burst(s): expected the generate(1) and generate(N+1) calls")
    s1, b1, p1 = stats(bs[-2]); sN, bN, pN = stats(bs[-1])
    n = a.tokens
    span, busy = (sN - s1) / n / 1e6, (bN - b1) / n / 1e6
    print(f"bursts {len(bs)}; last two: {len(bs[-2])} and {len(bs[-1])} kernels")
    print(f"per decode token: GPU span {span:.3f} ms, kernels busy {busy:.3f} ms, gaps {span - busy:.3f} ms "
          f"({(span - busy) / span * 100:.1f}%), {(len(bs[-1]) - len(bs[-2])) / n:.1f} kernels")
    rows = []
    for name, (t, c) in pN.items():
        t1, c1 = p1.get(name, (0, 0))
        rows.append(((t - t1) / n / 1e3, (c - c1) / n, name))
    rows.sort(reverse=True)
    print("per token, by kernel (us, launches):")
    for t, c, name in rows[:25]:
        print(f"  {t:9.1f} us  {c:6.1f}  {name[:110]}")
    if a.json:
        Path(a.json).write_text(json.dumps({"span_ms": span, "busy_ms": busy, "kernels_per_token": (len(bs[-1]) - len(bs[-2])) / n,
                                            "by_kernel": [{"us": t, "launches": c, "name": nm} for t, c, nm in rows]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--golden", required=True)
    r.add_argument("--tokens", type=int, default=32); r.add_argument("--custom-ops", default=None)
    z = sp.add_parser("analyze"); z.add_argument("dir"); z.add_argument("--tokens", type=int, default=32); z.add_argument("--json", default=None)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else analyze(a)


if __name__ == "__main__":
    main()
