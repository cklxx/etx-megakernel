"""List the compile plan's exports as sorted NAME<TAB>KIND lines.

The selector is environment-independent. model.instances applies the explicit
A/B instructions to its phases; use that same final plan rather than duplicating
those instructions here. --shell supplies both build scripts with import and
emitter arguments. No import is added unless the final plan instantiates it.
"""
from __future__ import annotations

import argparse
import os
import shlex

from examples.llm.model import load_dims
from etx.importer.spec import OPERATORS
from etx.select.planner import plan
from etx.select.variants import REGISTRY


def exports(config):
    d = load_dims(config)
    selected = plan(d, "gfx942", None)
    from etx.ops import gemv
    kinds = {v.export: v.kind for v in REGISTRY}
    kinds.update({n: "dsl" for n in gemv.import_entries()})
    kinds.update({v.export: v.kind for v in selected.phases.values()})
    # A generic selection can be inspected, but cannot use the vLLM host ABI.
    if any(v.kind == "generic" for v in selected.phases.values()):
        return d, {v.export: v.kind for v in selected.phases.values()}, []
    from examples.vllm_llm.model import instances
    phases = instances(d)
    names = {p["export"] for p in phases}
    return d, {n: "imported" if n in OPERATORS else kinds[n] for n in sorted(names)}, phases


def build_settings(d, names, phases):
    imported = {n for n, kind in names.items() if kind == "imported"}
    generic = [n for n, kind in names.items() if kind == "generic"]
    if generic:
        raise SystemExit(f"vLLM host cannot launch generic exports: {sorted(generic)}")
    # Variant phases do not carry launch block sizes. Retain the existing
    # shape derivation only for these variable-block kernels, filtered by plan.
    blocks = {"rms_norm_2d": d.H // 8, "reshape_and_cache": d.NKV * d.HD // 8}
    if d.moe:
        blocks["silu_and_mul"] = min(d.MI // 8, 1024)
    slots = {}
    for entry in filter(None, os.environ.get("ETX_VL_SLOTS", "").split(",")):
        name, value = entry.split("=", 1)
        if name in imported:
            slots[name] = int(value)
    for p in phases:
        if p["role"] in ("moe_up", "moe_down", "moe_act") and p["export"] in imported:
            slots[p["export"]] = 1
    # The instance plan carries the attention/reduce grid. One prep task per
    # KV head requires one attention block/task and G reduce blocks/task.
    prep = next((p for p in phases if p["role"] == "prep"), None)
    if prep:
        for p in phases:
            if p["export"] not in imported:
                continue
            if p["role"] == "attn":
                slots[p["export"]] = 1
            elif p["role"] == "reduce":
                slots[p["export"]] = p["grid"][0] // prep["grid"][0]
    ops = set()
    for name, kind in names.items():
        if kind != "dsl":
            continue
        if name.startswith(("dsl_gemv", "dsl_moe")):
            ops.add("gemv")
        elif name == "dsl_topk":
            ops.add("topk")
        elif name == "dsl_attnprep":
            ops.add("attnprep")
        else:
            raise SystemExit(f"no tile emitter for planned DSL export {name}")
    return {
        "EXPORTS": ",".join(sorted(imported)),
        "SLOTS": ",".join(f"{n}={slots[n]}" for n in sorted(slots)),
        "DSLOPS": ",".join(sorted(ops)),
        "ETX_VL_DSL_EXT": str(int(d.ln_norm or d.bias_o or d.bias_lm)),
        "BLK": [arg for n in sorted(blocks) if n in imported for arg in ("--block", f"{n}={blocks[n]}")],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--kind", choices=("imported", "dsl", "generic"))
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    d, names, phases = exports(args.config)
    if args.shell:
        for key, value in build_settings(d, names, phases).items():
            if isinstance(value, list):
                print(f"{key}=(" + " ".join(map(shlex.quote, value)) + ")")
            else:
                print(f"{key}={shlex.quote(value)}")
        print("export ETX_VL_DSL_EXT")
    else:
        for name, kind in sorted(names.items()):
            if args.kind is None:
                print(f"{name}\t{kind}")
            elif kind == args.kind:
                print(name)


if __name__ == "__main__":
    main()
