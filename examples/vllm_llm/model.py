"""Qwen-family decode as an ETX graph of vLLM's own kernels, imported at the IR level.

Every grid is one launch of the kernel vLLM runs for that step of the model on MI300X at batch 1 with
custom ops on (vllm/model_executor/models/qwen3.py / qwen2.py, the ROCM_ATTN backend, wvSplitK for every
linear layer); the order is vLLM's stream order and every edge is a full barrier, so the megakernel runs
exactly the work vLLM's HIP graph runs, with the launches replaced by ETX events:

  layer L:  in-norm        rms_norm (L = 0) | fused_add_rms_norm(down_{L-1}, residual)
            qkv            wvSplitK   [+ bias for Qwen2]
            q/k norm       rms_norm 3-D over heads (Qwen3 only)
            rope           rotary_embedding (neox), in place on q and k
            cache          reshape_and_cache (k, v -> the paged cache)
            attn, reduce   paged_attention_rocm: ll4mi QKV kernel + reduction
            o_proj         wvSplitK
            post-norm      fused_add_rms_norm(o, residual)
            gate_up        wvSplitK      silu_and_mul        down   wvSplitK
  then fused_add_rms_norm, lm_head (wvSplitK), argmax.

Embedding lookup, the per-step position/slot/length scalars and argmax are ETX tiles (vLLM uses torch
kernels for those). The host fills one argument block per grid instance (vl_plan.txt lists them in order:
instance, export, layer, role, grid, byte offset); the tile finds its block at c->cst[0].
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from etx.frontends import hip_link  # noqa: E402
from etx.ir import Graph, Resource  # noqa: E402
from examples.llm.model import load_dims  # noqa: E402

CU = 304                      # MI300X compute units (vLLM passes cu_count to wvSplitK)
BS = 16                       # KV block size (vLLM default on ROCm)
PART = 256                    # paged attention partition size (fixed in paged_attention_custom_launcher)
THREADS = 1024                # megakernel workgroup = the largest imported block (wvSplitK 64 x 16)
MAXLEN = int(os.environ.get("ETX_VL_MAXLEN", "1072"))    # vLLM's max_model_len for the benchmark (prompt + 32 + 16)
# ETX_VL_PIN=1: the small launches between qkv and o_proj (q/k norm, rope, cache, attention, reduce) all run on
# XCD 0, so the events between them are XCD-local (a wait on L2 of one XCD) instead of device-wide
PIN_ROLES = {"q_norm", "k_norm", "rope", "cache", "attn", "reduce"} if os.environ.get("ETX_VL_PIN", "0") == "1" else set()
# ETX_VL_CHAIN=1: consecutive launches that each fit on one workgroup run as ONE task, back to back with a
# __syncthreads between them instead of an event (q/k norm, rope and the cache write; at most 4 per task)
CHAIN_ROLES = ("q_norm", "k_norm", "rope", "cache")
CHAIN = os.environ.get("ETX_VL_CHAIN", "0") == "1"
CHAIN_FILE = os.environ.get("ETX_VL_CHAINS", str(HERE / "build" / "chain_tiles.hip"))


def imports() -> dict:
    p = os.environ.get("ETX_VL_IMPORTS")
    if not p:
        raise SystemExit("ETX_VL_IMPORTS=<imports.json from etx.importer.vllm_kernels> is required")
    return json.loads(Path(p).read_text())


def mindiv(n: int, div1: int, div2: int) -> int:          # vLLM skinny_gemms.cu, the waves-per-group choice
    npr = div1 * div2
    rnds = []
    for _ in range(13):
        rnds.append((n + npr - 1) // npr); npr -= div1
    for i in range(12, -1, -1):
        if rnds[0] == rnds[i]:
            return div2 - i
    return 0


def wvsplitk_ytile(m: int) -> int:                           # WVSPLIT_TILE for gfx9, N = 1
    syt = (m + CU * 4 - 1) // (CU * 4)
    return 1 if syt <= 1 else 2


def instances(d) -> list[dict]:
    """The launches of one decode step, in vLLM's order."""
    parts = math.ceil(MAXLEN / PART)
    out = []

    def add(export, layer, role, grid, **kw):
        out.append({"export": export, "layer": layer, "role": role, "grid": grid, **kw})

    def gemv(layer, role, m, k):
        y = wvsplitk_ytile(m)
        add(f"wvsplitk_y{y}", layer, role, (CU, 1, 1), M=m, K=k, wvprgrp=mindiv(m, CU * y, 16))

    attn = {4: "attn_mfma4_g4", 6: "attn_mfma16_g6", 8: "attn_mfma16_g8"}[d.G]
    for L in range(d.L):
        add("rms_norm_2d" if L == 0 else "fused_add_rms_norm", L, "in_norm", (1, 1, 1))
        gemv(L, "qkv", d.NQKV, d.H)
        if d.qk_norm:
            add("rms_norm_3d", L, "q_norm", (d.NH, 1, 1))
            add("rms_norm_3d", L, "k_norm", (d.NKV, 1, 1))
        add("rope_neox", L, "rope", (1, 1, 1))
        add("reshape_and_cache", L, "cache", (1, 1, 1))
        add(attn, L, "attn", (1, parts, d.NKV))
        add("attn_reduce", L, "reduce", (d.NH, 1, 1))
        gemv(L, "o_proj", d.H, d.NH * d.HD)
        add("fused_add_rms_norm", L, "post_norm", (1, 1, 1))
        gemv(L, "gate_up", 2 * d.INTER, d.H)
        add("silu_and_mul", L, "act", (1, 1, 1))
        gemv(L, "down", d.H, d.INTER)
    add("fused_add_rms_norm", -1, "final_norm", (1, 1, 1))
    gemv(-1, "lm_head", d.V, d.H)
    return out


def plan_offsets(insts: list[dict], imp: dict) -> int:
    off = 0
    for i in insts:
        info = imp[i["export"]]
        off = (off + 15) // 16 * 16
        i["offset"] = off
        i["tasks"] = math.ceil(i["grid"][0] * i["grid"][1] * i["grid"][2] / info["slots"])
        off += info["arg_bytes"]
    return (off + 15) // 16 * 16


def build() -> Graph:
    d = load_dims()
    imp = imports()
    insts = instances(d)
    skip = set(filter(None, os.environ.get("ETX_VL_SKIP", "").split(",")))    # register analysis only: drop roles
    if skip:
        insts = [i for i in insts if i["role"] not in skip]
    total = plan_offsets(insts, imp)
    # the imported tiles use the import arena of etx/import.h (sized by the adapter); lds_bytes here is only
    # the ETX tiles' own LDS (argmax: one value and one index per thread)
    res = Resource(threads=THREADS, vgpr=128, agpr=0, lds_bytes=8 * THREADS + 256)
    g = Graph(f"vl_{d.name}")
    g.tensor("impargs", (total,), role="weight", bytes_per_elem=1)
    g.tensor("vl", (1,), role="weight", bytes_per_elem=8)         # placeholder: the native tiles read __constant__ params
    adapters = os.environ.get("ETX_VL_ADAPTERS", str(HERE / "build" / "imp_tiles.hip"))
    native = str(HERE / "vl_tiles.hip")

    g.etensor("E_pro", (1,), wait_count=1)
    g.call_device("prologue", (1,), hip_link(native, "vl_prologue"), resource=res, args=["vl"],
                  out_edges={"E_pro": "i->i"}, duration_us=2.0)
    prev = "E_pro"
    # ETX_VL_CHAIN: runs of CHAIN_ROLES within one layer become one grid of one task (at most 4, the consts)
    groups, cur = [], []
    for i in insts:
        chainable = CHAIN and i["role"] in CHAIN_ROLES
        if cur and (not chainable or cur[-1]["layer"] != i["layer"] or len(cur) == 4):
            groups.append(cur); cur = []
        if chainable:
            cur.append(i)
        else:
            groups.append([i])
    if cur:
        groups.append(cur)
    chains = {}
    for n, grp in enumerate(groups):
        ev = f"E_{n}"
        if len(grp) == 1:
            i = grp[0]
            g.etensor(ev, (1,), wait_count=i["tasks"])
            name = f"{i['role']}_{i['layer']}" if i["layer"] >= 0 else i["role"]
            pin = {"domain_map": "i->(0)"} if i["role"] in PIN_ROLES else {}
            g.call_device(name, (i["tasks"],), hip_link(adapters, f"etx_tile_{i['export']}"), resource=res, args=["impargs"],
                          consts=(i["offset"],), in_edges={prev: "i->(0)"}, out_edges={ev: "i->(0)"}, **pin,
                          duration_us=2.0 if not i["export"].startswith("wvsplitk") else max(2.0, i["M"] * i["K"] * 2 / 4.0e12 * 1e6))
        else:
            sig = "__".join(x["export"] for x in grp)
            chains[sig] = [x["export"] for x in grp]
            g.etensor(ev, (1,), wait_count=1)
            name = "chain_" + "_".join(x["role"] for x in grp) + f"_{grp[0]['layer']}"
            g.call_device(name, (1,), hip_link(CHAIN_FILE, f"etx_chain_{sig}"), resource=res, args=["impargs"],
                          consts=tuple(x["offset"] for x in grp), in_edges={prev: "i->(0)"}, out_edges={ev: "i->(0)"},
                          duration_us=2.0 * len(grp))
        prev = ev
    if chains:
        with open(CHAIN_FILE, "w") as f:
            f.write("// generated by examples/vllm_llm/model.py (ETX_VL_CHAIN=1): chained launches, one task each\n#pragma once\n")
            f.write(f'#include "{adapters}"\n')
            for sig, exps in chains.items():
                f.write(f'extern "C" __device__ __attribute__((always_inline)) void etx_chain_{sig}(const etx_ctx* c) {{\n')
                f.write("  const unsigned char* base = (const unsigned char*)c->args[0];\n")
                f.write("".join(f"  etx_all_blocks_{e}(base + c->cst[{k}]);\n" for k, e in enumerate(exps)))
                f.write("}\n")
    g.etensor("E_argmax", (1,), wait_count=1)
    g.call_device("argmax", (1,), hip_link(native, "vl_argmax"), resource=res, args=["vl"],
                  in_edges={prev: "i->(0)"}, out_edges={"E_argmax": "i->i"}, duration_us=3.0)

    plan = os.environ.get("ETX_VL_PLAN")
    if plan:
        with open(plan, "w") as f:
            f.write(f"# vl plan {d.name}: instance export layer role gx gy gz offset tasks M K wvprgrp; total {total} bytes\n")
            f.write(f"total {total} parts {math.ceil(MAXLEN / PART)} maxlen {MAXLEN}\n")
            for n, i in enumerate(insts):
                gx, gy, gz = i["grid"]
                f.write(f"{n} {i['export']} {i['layer']} {i['role']} {gx} {gy} {gz} {i['offset']} {i['tasks']} "
                        f"{i.get('M', 0)} {i.get('K', 0)} {i.get('wvprgrp', 0)}\n")
    return g


def bindings(**overrides) -> dict:
    return {}


def runtime(bindings, rng=None):
    return {}
