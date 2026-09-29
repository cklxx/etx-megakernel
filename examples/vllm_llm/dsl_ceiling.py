#!/usr/bin/env python3
"""What the fused small-op structure (ETX_VL_DSL, docs/MODEL-vllm-kernels.md section 12) is worth, simulated before any
of it is built: the xcd + fused-q/k-norm plan of today against the DSL plan, same kernel-model parameters from the
microbenchmarks (results_2026-09-28b), hand-offs from the trace.

  python3 -m examples.vllm_llm.dsl_ceiling [--bench results_2026-09-28b] [--layers 48]
Prints the simulated step of each plan, the calibration against the measured builds (old 6.56, new 6.31, qknr 5.86 ms)
and layer 5's span per role.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from pathlib import Path

from etx.codegen.kernel import emit_plan_json
from etx.passes import compile_graph
from etx.sim.xcd import Hw, simulate
from examples.vllm_llm import kmodel as km
from examples.vllm_llm.bench_params import derive, read_dir
from examples.vllm_llm.xcd_run import cost_fn
from etx.ops import gemv as gemv_op

ROOT = Path(__file__).resolve().parents[2]
DSL_IMPORTS = {"dsl_attn": {"slots": 1, "arg_bytes": 160}, "dsl_topk": {"slots": 4, "arg_bytes": 104}}


def make_plan(env: dict, imports_json: Path, tmp: Path) -> dict:
    for k, v in env.items():
        os.environ[k] = v
    os.environ["ETX_VL_IMPORTS"] = str(imports_json)
    os.environ["ETX_LLM_CONFIG"] = str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json")
    os.environ["ETX_VL_PLAN"] = str(tmp / "vl_plan.txt")
    os.environ["ETX_VL_CHAINS"] = str(tmp / "chain_tiles.hip")
    import examples.vllm_llm.model as vm
    vm = importlib.reload(vm)
    return json.loads(emit_plan_json(compile_graph(vm.build(), "gfx942", {}, {})))


def params(bench: dict, build: str) -> km.Params:
    p = km.Params(t_v=bench["t_v"][0], t_s=bench["t_s"][0], F=bench["F"][0], bw=bench["bw"][0], t_i=0.0015)
    p.hw["t_swbar"] = bench["t_swbar"][0] if build == "old" else bench["t_round"][0]
    p.A_task = bench["A_task_vector"][0] if build == "old" else bench["A_task_scalar"][0]
    p.A_unf = p.A_task
    return p


def sim(plan: dict, model: str, p: km.Params):
    hw = Hw(bw_max=p.bw, n_half=0.0, handoff_us={"DOMAIN": 0.4, "DEVICE": 1.8}, wake_us=0.6, rate_cv=0.03, lat_us=p.t_v)
    rows, st = simulate(plan, cost_fn(model, p), hw)
    names = {v["id"]: k for k, v in plan["types"].items()}
    span = {}
    for d, r in zip(plan["descs"], rows):
        n = names[d["type"]]
        if n.endswith("_5") or n.endswith("_5_x"):
            role = n.replace("_5_x", "").replace("_5", "")
            a, b = r[6] * 0.01, r[7] * 0.01
            s = span.setdefault(role, [1e12, 0, 0])
            s[0], s[1], s[2] = min(s[0], a), max(s[1], b), s[2] + 1
    return st["makespan_us"] / 1e3, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="results_2026-09-28b")
    ap.add_argument("--imports", default="build/rocm22/qwen3-30b-a3b_xcd_qknr/imp/imports.json")
    ap.add_argument("--layers", default=None, help="fewer layers for a quick look (ETX_LLM_LAYERS)")
    a = ap.parse_args()
    bench = derive(read_dir(a.bench))
    imp = json.load(open(a.imports)) | DSL_IMPORTS | gemv_op.import_entries()
    tmp = Path(tempfile.mkdtemp())
    (tmp / "imports.json").write_text(json.dumps(imp))
    base_env = {"ETX_VL_XCD": "1", "ETX_VL_CHAIN": "1", "ETX_VL_QKNR": "0", "ETX_VL_DSL": "0", "ETX_VL_REQUESTS": "1", "ETX_VL_DSL_GEMV": "0"}
    if a.layers:
        base_env["ETX_LLM_CONFIG_LAYERS"] = a.layers
        os.environ["ETX_LLM_LAYERS"] = a.layers
    plans = {
        "xcd (four small launches chained)": (dict(base_env), "qwen3-30b-a3b", "new"),
        "xcd + fused q/k norm + rope (grid)": (dict(base_env, ETX_VL_QKNR="grid"), "qwen3-30b-a3b", "new"),
        "DSL: fused small ops": (dict(base_env, ETX_VL_DSL="1"), "qwen3-30b-a3b-dsl", "new"),
        "DSL + DSL streaming GEMVs": (dict(base_env, ETX_VL_DSL="1"), "qwen3-30b-a3b-dslgemv", "new"),
        "phase 1: DSL GEMVs + top-k + fold": (dict(base_env, ETX_VL_DSL="2"), "qwen3-30b-a3b-dslp1", "new"),
        "build 1: qknr grid + DSL GEMVs only": (dict(base_env, ETX_VL_QKNR="grid", ETX_VL_DSL_GEMV="1"), "qwen3-30b-a3b-gemvonly", "new"),
    }
    measured = {"xcd (four small launches chained)": 6.31, "xcd + fused q/k norm + rope (grid)": 5.86}
    print("simulated ms per token (microbenchmark parameters, trace hand-offs 0.4 / 1.8 us); measured where it exists")
    spans = {}
    for name, (env, model, build) in plans.items():
        plan = make_plan(env, tmp / "imports.json", tmp)
        ms, span = sim(plan, model, params(bench, build))
        spans[name] = span
        m = measured.get(name)
        print(f"  {name:38s} {ms:6.3f}" + (f"   measured {m:.2f}  (sim x {m / ms:.2f})" if m else ""))
    print("\nlayer 5, us from the layer's first task (start -> end of every task of the role; tasks)")
    roles = sorted({r for s in spans.values() for r in s}, key=lambda r: min(s.get(r, [1e12])[0] for s in spans.values()))
    print(f"  {'role':10s}" + "".join(f"{n[:22]:>26s}" for n in spans))
    for r in roles:
        line = f"  {r:10s}"
        for s in spans.values():
            t0 = min(v[0] for v in s.values())
            line += f"{'-':>26s}" if r not in s else f"{s[r][0] - t0:8.1f} ->{s[r][1] - t0:7.1f} ({s[r][2]:3d})   "
        print(line)


if __name__ == "__main__":
    main()
