#!/usr/bin/env python3
"""The compiler's configuration search (docs/qwen/MODEL-vllm-kernels.md section 15): enumerate the build switches of
examples/vllm_llm/model.py, compile each plan, cost every task from the tile it links (kmodel.kern_for_symbol) and
simulate the step on the XCD simulator with the measured phase model (examples/vllm_llm/pmodel.py); print
the ranking. What used to be hand-set
environment variables (chain, fused q/k norm + rope, DSL GEMVs, DSL top-k, the attention block by kv head, slots)
becomes a search over a small space, best first.

  python3 -m examples.vllm_llm.autoconf [--bench results_2026-09-28b] [--top 8]
"""
from __future__ import annotations
from . import paths

import argparse
import itertools
import json
import tempfile
from pathlib import Path

from etx.sim.xcd import Hw, simulate
from . import dsl_ceiling as dc
from .bench_params import derive
from .bench_params import read_dir
from . import pmodel
from etx.ops import attnprep, gemv, topk

SWITCHES = {                                   # env name -> the values searched
    "ETX_VL_CHAIN": ["0", "1"],
    "ETX_VL_QKNR": ["0", "grid", "1"],
    "ETX_VL_DSL_GEMV": ["0", "1"],
    "ETX_VL_DSL_TOPK": ["0", "1"],
    "ETX_VL_DSL_ATTN": ["0", "head"],
}


def valid(env: dict) -> bool:
    if env["ETX_VL_DSL_ATTN"] == "head" and env["ETX_VL_QKNR"] != "0":
        return False                           # the head block has its own norm + rope
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="results_2026-09-28b")
    ap.add_argument("--imports", default="build/rocm22/qwen3-30b-a3b_xcd_qknr/imp/imports.json")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--config", default="examples/llm/configs/qwen3-30b-a3b.json")
    ap.add_argument("--mapping", default="contig", choices=["contig", "rows"])
    a = ap.parse_args()
    bench = derive(read_dir(a.bench))
    p = dc.params(bench, "new")
    imp = json.load(open(a.imports)) | gemv.import_entries() | topk.import_entries() | attnprep.import_entries()
    tmp = Path(tempfile.mkdtemp())
    (tmp / "imports.json").write_text(json.dumps(imp))
    imp1 = dict(imp); imp1["attn_mfma16_g8"] = dict(imp["attn_mfma16_g8"], slots=1)
    (tmp / "imports_s1.json").write_text(json.dumps(imp1))
    cfg = json.load(open(a.config))
    hw = pmodel.hw()
    results = []
    for vals in itertools.product(*SWITCHES.values()):
        env = dict(zip(SWITCHES, vals))
        if not valid(env):
            continue
        env = {"ETX_VL_XCD": "1", "ETX_VL_DSL": "0", "ETX_VL_REQUESTS": "1", **env}
        impf = "imports_s1.json" if env["ETX_VL_DSL_ATTN"] == "head" else "imports.json"
        try:
            plan = dc.make_plan(env, tmp / impf, tmp)
        except SystemExit as e:
            results.append((float("inf"), env, f"rejected: {e}")); continue
        rows, st = simulate(plan, pmodel.cost_fn(plan, cfg, a.mapping), hw)
        results.append((st["makespan_us"] / 1e3, env, ""))
    results.sort(key=lambda r: r[0])
    print(f"{'sim ms':>7s}  " + "  ".join(k.replace('ETX_VL_', '') for k in SWITCHES))
    for ms, env, note in results[:a.top]:
        print(f"{ms:7.3f}  " + "  ".join(f"{env[k]:>{len(k) - 7}s}" for k in SWITCHES) + ("  " + note if note else ""))
    print(f"{len(results)} configurations; measured build 3 5.895 ms, vLLM 4.67 ms")


if __name__ == "__main__":
    main()
