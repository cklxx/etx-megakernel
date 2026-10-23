#!/usr/bin/env python3
"""Build 3's simulated step under the parameters the microbenchmarks left uncertain (docs/qwen/MODEL-vllm-kernels.md
section 17), so a measurement that misses the prediction points at a parameter rather than at "the model":

  python3 -m examples.vllm_llm.sensitivity [--bench results_2026-09-28b] [--requests 1,2]
Variants: hand-offs from the trace (0.4 / 1.8 us) or from the evx microbenchmark (2.2 / 3.9); the argument path
(A_task) and the wake-up spread at 0.5x / 1.5x; the prologue round trip t_v idle MALL (0.26) / idle HBM (0.41) /
loaded (0.7). Prints the simulated ms per token and x1.11 (the realistic factor of the measured builds); then the
same build with the touch hook (ETX_VL_TOUCH=1, section 18) under the first two variants.
"""
from __future__ import annotations
from . import paths

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from etx.sim.xcd import Hw, simulate
from . import dsl_ceiling as dc
from .bench_params import derive
from .bench_params import read_dir
from .xcd_run import cost_fn
from etx.ops import attnprep, gemv, topk

BUILD3 = {"ETX_VL_XCD": "1", "ETX_VL_CHAIN": "1", "ETX_VL_QKNR": "0", "ETX_VL_DSL": "0", "ETX_VL_DSL_GEMV": "1",
          "ETX_VL_DSL_TOPK": "1", "ETX_VL_DSL_ATTN": "head", "ETX_VL_DSL_FOLD": "1"}


def variants(p):
    yield "baseline (trace hand-offs 0.4 / 1.8)", p, dict(handoff={"DOMAIN": 0.4, "DEVICE": 1.8}, wake=0.6)
    yield "hand-offs = evx microbenchmark 2.2 / 3.9", p, dict(handoff={"DOMAIN": 2.2, "DEVICE": 3.9}, wake=0.6)
    yield "hand-offs 0 (what a perfect event costs)", p, dict(handoff={"DOMAIN": 0.0, "DEVICE": 0.0}, wake=0.0)
    yield "argument path A_task x 0.5", replace(p, A_task=p.A_task * 0.5, A_unf=p.A_unf * 0.5), dict(handoff={"DOMAIN": 0.4, "DEVICE": 1.8}, wake=0.6)
    yield "argument path A_task x 1.5", replace(p, A_task=p.A_task * 1.5, A_unf=p.A_unf * 1.5), dict(handoff={"DOMAIN": 0.4, "DEVICE": 1.8}, wake=0.6)
    yield "wake-up spread x 0.5 (0.3 us)", p, dict(handoff={"DOMAIN": 0.4, "DEVICE": 1.8}, wake=0.3)
    yield "wake-up spread x 1.5 (0.9 us)", p, dict(handoff={"DOMAIN": 0.4, "DEVICE": 1.8}, wake=0.9)
    yield "prologue round trip t_v = 0.41 (idle HBM)", replace(p, t_v=0.412), dict(handoff={"DOMAIN": 0.4, "DEVICE": 1.8}, wake=0.6)
    yield "prologue round trip t_v = 0.70 (loaded)", replace(p, t_v=0.70), dict(handoff={"DOMAIN": 0.4, "DEVICE": 1.8}, wake=0.6)
    yield "everything pessimistic (evx, A x1.5, t_v 0.7)", replace(p, A_task=p.A_task * 1.5, A_unf=p.A_unf * 1.5, t_v=0.70), dict(handoff={"DOMAIN": 2.2, "DEVICE": 3.9}, wake=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="results_2026-09-28b")
    ap.add_argument("--imports", default="build/rocm22/qwen3-30b-a3b_xcd_qknr/imp/imports.json")
    ap.add_argument("--requests", default="1,2")
    a = ap.parse_args()
    p0 = dc.params(derive(read_dir(a.bench)), "new")
    imp = json.load(open(a.imports)) | gemv.import_entries() | topk.import_entries() | attnprep.import_entries()
    imp["attn_mfma16_g8"] = dict(imp["attn_mfma16_g8"], slots=1)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "imports.json").write_text(json.dumps(imp))
    reqs = [int(r) for r in a.requests.split(",")]
    plans = {r: dc.make_plan(dict(BUILD3, ETX_VL_REQUESTS=str(r)), tmp / "imports.json", tmp) for r in reqs}
    print(f"build 3, simulated ms per token (x1.11 = realistic); requests {reqs}")
    print(f"  {'variant':46s}" + "".join(f"{f'R={r}: sim':>12s}{'x1.11':>8s}" for r in reqs))
    for name, p, hw in variants(p0):
        line = f"  {name:46s}"
        for r in reqs:
            h = Hw(bw_max=p.bw, n_half=0.0, handoff_us=hw["handoff"], wake_us=hw["wake"], rate_cv=0.03, lat_us=p.t_v)
            ms = simulate(plans[r], cost_fn("auto", p, plan=plans[r]), h)[1]["makespan_us"] / 1e3 / r
            line += f"{ms:12.3f}{ms * 1.11:8.2f}"
        print(line)
    # the touch hook (ETX_VL_TOUCH=1, section 18): the dense GEMVs' first weight trip pulled into L2 during the wait
    tplans = {r: dc.make_plan(dict(BUILD3, ETX_VL_REQUESTS=str(r), ETX_VL_TOUCH="1"), tmp / "imports.json", tmp) for r in reqs}
    for name, p, hw in list(variants(p0))[:2]:
        line = f"  {'+ touch, ' + name[:36]:46s}"
        for r in reqs:
            h = Hw(bw_max=p.bw, n_half=0.0, handoff_us=hw["handoff"], wake_us=hw["wake"], rate_cv=0.03, lat_us=p.t_v)
            ms = simulate(tplans[r], cost_fn("auto", p, plan=tplans[r]), h)[1]["makespan_us"] / 1e3 / r
            line += f"{ms:12.3f}{ms * 1.11:8.2f}"
        print(line)


if __name__ == "__main__":
    main()
