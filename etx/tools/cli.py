"""etx command line.

  python -m etx compile examples/moe_layer.py --arch gfx942 --out build/moe --sim
  python -m etx explain examples/moe_layer.py --arch gfx942
  python -m etx archs
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from ..codegen import emit_kernel, emit_lowering_header, emit_plan_json
from ..frontends import collect_sources, emit_tile_decls, emit_triton_skeleton
from ..machine import list_archs, load_machine
from ..passes import PassOptions, compile_graph
from ..sim import simulate


def _load_example(path: str):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(p.parent.parent))
    sys.modules[spec.name] = mod            # dataclasses in the example resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _plan_from_args(a):
    mod = _load_example(a.graph)
    graph = mod.build()
    overrides = {k: int(v) for k, v in (kv.split("=") for kv in a.bind)}
    bindings = mod.bindings(**overrides)
    runtime = mod.runtime(bindings)
    opts = PassOptions(event_elimination=not a.no_event_elim, prefetch=not a.no_prefetch,
                       split_resource_classes=a.split_classes, force_mode=a.force_mode,
                       inline_tiles=getattr(a, "inline_tiles", False), relay=getattr(a, "relay", "off"))
    plan = compile_graph(graph, load_machine(a.arch), bindings, runtime, opts)
    return plan, graph


def cmd_compile(a) -> int:
    plan, graph = _plan_from_args(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "etx_lowering.h").write_text(emit_lowering_header(plan.machine), encoding="utf-8")
    ext = "hip" if plan.machine.vendor == "amd" else "cu"
    for d in range(plan.n_devices):
        (out / f"megakernel_d{d}.{ext}").write_text(emit_kernel(plan, d), encoding="utf-8")
    (out / "etx_tiles.h").write_text(emit_tile_decls(plan.graph), encoding="utf-8")
    (out / "plan.json").write_text(emit_plan_json(plan), encoding="utf-8")
    from ..codegen.plan_header import emit_plan_header
    (out / "plan_data.h").write_text(emit_plan_header(plan, 0), encoding="utf-8")
    (out / "decisions.log").write_text("\n".join(plan.log) + "\n", encoding="utf-8")
    if any(g.body.kind == "triton" for g in plan.graph.grids):
        (out / "megakernel_triton.py").write_text(emit_triton_skeleton(plan), encoding="utf-8")
    print(plan.summary())
    print(f"sources to link: {collect_sources(plan.graph)}")
    print(f"wrote {out}/")
    if a.sim:
        res = simulate(plan, seed=a.seed)
        print("sim:", res.summary())
        if res.deadlock:
            return 2
    return 0


def cmd_explain(a) -> int:
    plan, _ = _plan_from_args(a)
    print(plan.summary())
    print("--- decision log")
    for line in plan.log:
        print(line)
    if a.sim:
        for mode in ("static", "dynamic", "hybrid"):
            a.force_mode = mode
            p, _ = _plan_from_args(a)
            r = simulate(p, seed=a.seed)
            print(f"sim[{mode:7s}] {r.summary()}")
    return 0


def cmd_archs(a) -> int:
    for n in list_archs():
        m = load_machine(n)
        print(f"{n:8s} {m.family:10s} domains={m.num_domains} cus/domain={m.cus_per_domain()} lds={m.resources['lds_kb']}KB hbm={m.bandwidth['hbm_tbs']}TB/s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="etx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("compile", cmd_compile), ("explain", cmd_explain)):
        s = sub.add_parser(name)
        s.add_argument("graph")
        s.add_argument("--arch", default="gfx942")
        s.add_argument("--out", default="build/out")
        s.add_argument("--bind", action="append", default=[], help="SYM=value overrides")
        s.add_argument("--sim", action="store_true")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--no-event-elim", action="store_true")
        s.add_argument("--no-prefetch", action="store_true")
        s.add_argument("--split-classes", action="store_true")
        s.add_argument("--force-mode", choices=["static", "dynamic", "hybrid"], default=None)
        s.add_argument("--inline-tiles", action="store_true", help="#include the tile sources into the kernel TU so bodies can inline")
        s.add_argument("--relay", choices=["auto", "on", "off"], default="off", help="per-domain relay of DEVICE-scope counters (P5); off by default")
        s.set_defaults(fn=fn)
    s = sub.add_parser("archs")
    s.set_defaults(fn=cmd_archs)
    a = ap.parse_args(argv)
    return a.fn(a)
