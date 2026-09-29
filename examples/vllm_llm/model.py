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
            or, for MoE layers (Qwen3-30B-A3B; fused_moe/fused_moe.py, TritonExperts at batch 1):
            router         wvSplitK (YTILE 1, 128 rows)
            topk           topk_softmax (softmax, top-8, renormalised)
            moe_up         Triton fused_moe_kernel, gate_up of the 8 experts (naive block assignment)
            act            silu_and_mul over the 8 rows
            moe_down       Triton fused_moe_kernel, down, times the routing weight
            moe_sum        moe_sum over the 8 rows, straight into the buffer the next norm reads
  then fused_add_rms_norm, lm_head (wvSplitK), argmax.

Two launches of vLLM's MoE layer are not in the graph: torch.full() of num_tokens_post_padded (8 * BLOCK_SIZE_M at
batch 1, a constant the host writes once) and the copy of moe_sum's output into the MoE custom op's out tensor (moe_sum
writes where the copy would put it).

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
MOE_BN = 64                   # fused_moe_kernel BLOCK_SIZE_N of vLLM's default config at batch 1 (examples/vllm_llm/triton)
MAXLEN = int(os.environ.get("ETX_VL_MAXLEN", "1072"))    # vLLM's max_model_len for the benchmark (prompt + 32 + 16)
# ETX_VL_PIN=1: the small launches between qkv and o_proj (q/k norm, rope, cache, attention, reduce) all run on
# XCD 0, so the events between them are XCD-local (a wait on L2 of one XCD) instead of device-wide
PIN_ROLES = {"q_norm", "k_norm", "rope", "qknr", "cache", "attn", "reduce"} if os.environ.get("ETX_VL_PIN", "0") == "1" else set()
# ETX_VL_CHAIN=1: consecutive launches that each fit on one workgroup run as ONE task, back to back with a
# __syncthreads between them instead of an event (q/k norm, rope and the cache write; at most 4 per task)
CHAIN_ROLES = ("q_norm", "k_norm", "rope", "qknr", "cache")
CHAIN = os.environ.get("ETX_VL_CHAIN", "0") == "1"
# ETX_VL_QKNR=1: q/k norm + rope as ONE launch of vLLM's fused_qk_norm_rope (tensorrt_llm::kernels::fusedQKNormRopeKernel,
# what vLLM runs with compilation_config pass_config enable_qk_norm_rope_fusion): one 32-lane warp per head, 8 per
# 256-thread block, in place on qkv; replaces rms_norm_3d x2 + rotary_embedding
# ETX_VL_QKNR=grid: the fused launch as its own grid (its blocks on several CUs), not chained with the cache write
QKNR = os.environ.get("ETX_VL_QKNR", "0") in ("1", "grid")
# ETX_VL_DSL=1 (simulation only, docs/MODEL-vllm-kernels.md section 12): the latency-bound small ops as ETX tile ops with
# their chains fused: q/k norm + rope + cache write + attention in one task per (partition, kv head); top-k as one tile;
# moe_sum folded into the next layer's in_norm. The streaming ops (GEMVs, fused_moe) stay vLLM's.
DSL = os.environ.get("ETX_VL_DSL", "0") == "1"
if os.environ.get("ETX_VL_QKNR", "0") == "grid":
    CHAIN_ROLES = tuple(r for r in CHAIN_ROLES if r != "qknr")
CHAIN_FILE = os.environ.get("ETX_VL_CHAINS", str(HERE / "build" / "chain_tiles.hip"))
# ETX_VL_XCD=1: the one-block launches between the GEMVs (in/post/final norm, silu_and_mul) run once per XCD on
# that XCD's own copies (residual, normed input, activation), and the GEMV that follows is a grid of
# (8 XCDs x 38) tasks each reading its XCD's copy: a device-wide handoff with 304 waiters becomes one with 8
# waiters plus an XCD-local one. The kernels are unchanged; only argument blocks and buffers are per XCD.
XCD = os.environ.get("ETX_VL_XCD", "0") == "1"
NX, WPX = 8, 38
REP_ROLES = {"in_norm", "post_norm", "act", "final_norm", "topk", "moe_sum"}
XGEMV_AFTER = {"qkv": "in_norm", "gate_up": "post_norm", "down": "act", "lm_head": "final_norm", "router": "post_norm"}
# MoE with ETX_VL_XCD: the expert path stays on one XCD per selected expert. top-k runs once per XCD (a replica); XCD x
# then runs the blocks of expert slot x of gate_up (fused_moe pid_m = pid // (N / BLOCK_SIZE_N) = x), of silu (row x) and
# of down, each waiting only on its own XCD's part of the previous launch; moe_sum runs once per XCD again. The kernels
# and their blocks are vLLM's; only which XCD runs which block, and what each block waits for, is ETX's.
XAFF_ROLES = {"moe_up", "moe_act", "moe_down"}
# ETX_VL_REQUESTS=2: two requests in one megakernel, one workgroup per CU. Their launches alternate in the graph (request
# 0's launch n, request 1's launch n, ...), so every worker's static queue interleaves them and a CU request 0 leaves
# idle (its 1-8-task launches, the attention block) runs request 1's tasks. Buffers, KV caches and argument blocks are
# per request; weights are shared. (docs/MODEL-vllm-kernels.md section 8)
REQUESTS = int(os.environ.get("ETX_VL_REQUESTS", "1"))


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
        if DSL:
            add("dsl_attn", L, "attn", (1, parts, d.NKV))          # prep + cache + attention, one block per (partition, kv head)
        elif d.qk_norm and QKNR:
            add("qknorm_rope", L, "qknr", (math.ceil((d.NH + d.NKV) / 8), 1, 1))
        else:
            if d.qk_norm:
                add("rms_norm_3d", L, "q_norm", (d.NH, 1, 1))
                add("rms_norm_3d", L, "k_norm", (d.NKV, 1, 1))
            add("rope_neox", L, "rope", (1, 1, 1))
        if not DSL:
            add("reshape_and_cache", L, "cache", (1, 1, 1))
            add(attn, L, "attn", (1, parts, d.NKV))
        add("attn_reduce", L, "reduce", (d.NH, 1, 1))
        gemv(L, "o_proj", d.H, d.NH * d.HD)
        add("fused_add_rms_norm", L, "post_norm", (1, 1, 1))
        if d.moe:
            gemv(L, "router", d.E, d.H)
            add("dsl_topk" if DSL else "topk_softmax_e128", L, "topk", (1, 1, 1))
            add("fused_moe_up", L, "moe_up", (d.TOPK * math.ceil(2 * d.MI / MOE_BN), 1, 1))
            add("silu_and_mul", L, "moe_act", (d.TOPK, 1, 1))
            add("fused_moe_down", L, "moe_down", (d.TOPK * math.ceil(d.H / MOE_BN), 1, 1))
            if not DSL:                                            # DSL: the next in_norm reads the 8 expert rows itself
                add("moe_sum_k8", L, "moe_sum", (1, 1, 1))
            continue
        gemv(L, "gate_up", 2 * d.INTER, d.H)
        add("silu_and_mul", L, "act", (1, 1, 1))
        gemv(L, "down", d.H, d.INTER)
    add("fused_add_rms_norm", -1, "final_norm", (1, 1, 1))
    gemv(-1, "lm_head", d.V, d.H)
    return out


def plan_offsets(insts: list[dict], imp: dict, start: int = 0) -> int:
    """Byte offsets of every argument block; replicated launches (ETX_VL_XCD) get one block per XCD at `stride`,
    and copies (src, dst, bytes: 32 B per XCD) when the kernel works in place on its XCD's copy."""
    off = start
    for n, i in enumerate(insts):
        info = imp[i["export"]]
        i["tasks"] = math.ceil(i["grid"][0] * i["grid"][1] * i["grid"][2] / info["slots"])
        stride = (info["arg_bytes"] + 15) // 16 * 16
        i["rep"] = NX if XCD and (i["role"] in REP_ROLES or i["role"] in XAFF_ROLES or
                                   (i["role"] in XGEMV_AFTER and n > 0 and insts[n - 1]["role"] == XGEMV_AFTER[i["role"]])) else 1
        i["stride"] = stride
        off = (off + 15) // 16 * 16
        i["offset"] = off
        off += stride * i["rep"]
        i["copyoff"] = -1
        # in-place norms copy the shared input into their XCD's copy first; after moe_sum (itself a replica writing its
        # XCD's copy) there is nothing to copy
        after_moe = n > 0 and insts[n - 1]["role"] == "moe_sum"
        if (i["rep"] > 1 and i["role"] in ("in_norm", "post_norm", "final_norm") and not (i["role"] == "in_norm" and i["layer"] == 0)
                and not after_moe):
            i["copyoff"] = off
            off += 32 * NX
    return (off + 15) // 16 * 16


def build() -> Graph:
    d = load_dims()
    imp = imports()
    base = instances(d)
    skip = set(filter(None, os.environ.get("ETX_VL_SKIP", "").split(",")))    # register analysis only: drop roles
    if skip:
        if XCD or REQUESTS > 1:
            raise SystemExit("ETX_VL_SKIP cannot be combined with ETX_VL_XCD or ETX_VL_REQUESTS (a replica or its GEMV could be dropped alone)")
        base = [i for i in base if i["role"] not in skip]
    per_req, total = [], 0
    for r in range(REQUESTS):                                  # each request its own argument blocks
        ins = [dict(i, req=r) for i in base]
        total = plan_offsets(ins, imp, total)
        per_req.append(ins)
    # the imported tiles use the import arena of etx/import.h (sized by the adapter); lds_bytes here is only
    # the ETX tiles' own LDS (argmax: one value and one index per thread)
    res = Resource(threads=THREADS, vgpr=128, agpr=0, lds_bytes=8 * THREADS + 256)
    g = Graph(f"vl_{d.name}" + (f"_r{REQUESTS}" if REQUESTS > 1 else ""))
    g.tensor("impargs", (total,), role="weight", bytes_per_elem=1)
    g.tensor("vl", (1,), role="weight", bytes_per_elem=8)         # placeholder: the native tiles read __constant__ params
    adapters = os.environ.get("ETX_VL_ADAPTERS", str(HERE / "build" / "imp_tiles.hip"))
    native = str(HERE / "vl_tiles.hip")
    pre = (lambda r: f"r{r}_") if REQUESTS > 1 else (lambda r: "")

    prev = {}
    for r in range(REQUESTS):
        g.etensor(f"{pre(r)}E_pro", (1,), wait_count=1)
        g.call_device(f"{pre(r)}prologue", (1,), hip_link(native, "vl_prologue"), resource=res, args=["vl"], consts=(r,),
                      out_edges={f"{pre(r)}E_pro": "i->i"}, duration_us=2.0)
        prev[r] = f"{pre(r)}E_pro"
    # ETX_VL_CHAIN: runs of CHAIN_ROLES within one layer become one grid of one task (at most 4, the consts)
    groups_r = []
    for ins in per_req:
        groups, cur = [], []
        for i in ins:
            chainable = CHAIN and i["role"] in CHAIN_ROLES
            if cur and (not chainable or cur[-1]["layer"] != i["layer"] or len(cur) == 4):
                groups.append(cur); cur = []
            if chainable:
                cur.append(i)
            else:
                groups.append([i])
        if cur:
            groups.append(cur)
        groups_r.append(groups)
    chains = {}
    order = []                                                 # graph (and plan-file) order: requests alternate
    for n in range(len(groups_r[0])):
        for r in range(REQUESTS):
            order.append((r, n))
    for r, n in order:
        groups = groups_r[r]
        grp = groups[n]
        P = pre(r)
        ev = f"{P}E_{n}"
        if len(grp) == 1 and grp[0]["rep"] > 1 and grp[0]["role"] in REP_ROLES:
            i = grp[0]
            g.etensor(ev, (NX,), wait_count=1)
            name = P + (f"{i['role']}_{i['layer']}_x" if i["layer"] >= 0 else f"{i['role']}_x")
            g.call_device(name, (NX,), hip_link(adapters, f"etx_tile_rep_{i['export']}", pre_wait=f"etx_pre_x_{i['export']}"),
                          resource=res, args=["impargs"],
                          consts=(i["offset"], i["stride"], i["copyoff"]), in_edges={prev[r]: "x->(0)"}, out_edges={ev: "x->x"},
                          domain_map="x->x", duration_us=3.0)
        elif len(grp) == 1 and grp[0]["rep"] > 1:
            # a grid split over the XCDs, (XCDs, tasks per XCD) pinned xw->x, each part waiting on its XCD's part of the
            # previous launch: the GEMV after a replica (38 per XCD) or an XCD-affine MoE launch (one expert slot per XCD)
            i = grp[0]
            per = i["tasks"] // NX
            if per * NX != i["tasks"] or (i["role"] not in XAFF_ROLES and per != WPX):
                raise SystemExit(f"{i['export']}: {i['tasks']} tasks do not split over {NX} XCDs as {i['role']} needs")
            nxt = groups[n + 1][0] if n + 1 < len(groups) else None
            local_next = nxt is not None and len(groups[n + 1]) == 1 and nxt["role"] in XAFF_ROLES and nxt["rep"] > 1
            if local_next:        # the next launch's part on this XCD waits for this XCD's part only
                g.etensor(ev, (NX,), wait_count=per)
                out = {ev: "xw->x"}
            else:
                g.etensor(ev, (1,), wait_count=i["tasks"])
                out = {ev: "xw->(0)"}
            name = P + (f"{i['role']}_{i['layer']}" if i["layer"] >= 0 else i["role"])
            dur = max(2.0, i["M"] * i["K"] * 2 / 4.0e12 * 1e6) if i["export"].startswith("wvsplitk") else 3.0
            g.call_device(name, (NX, per), hip_link(adapters, f"etx_tile_x_{i['export']}", pre_wait=f"etx_pre_x_{i['export']}"),
                          resource=res, args=["impargs"],
                          consts=(i["offset"], i["stride"], per), in_edges={prev[r]: "xw->x"}, out_edges=out,
                          domain_map="xw->x", duration_us=dur)
        elif len(grp) == 1:
            i = grp[0]
            g.etensor(ev, (1,), wait_count=i["tasks"])
            name = P + (f"{i['role']}_{i['layer']}" if i["layer"] >= 0 else i["role"])
            pin = {"domain_map": "i->(0)"} if i["role"] in PIN_ROLES else {}
            g.call_device(name, (i["tasks"],), hip_link(adapters, f"etx_tile_{i['export']}", pre_wait=f"etx_pre_{i['export']}"),
                          resource=res, args=["impargs"],
                          consts=(i["offset"],), in_edges={prev[r]: "i->(0)"}, out_edges={ev: "i->(0)"}, **pin,
                          duration_us=2.0 if not i["export"].startswith("wvsplitk") else max(2.0, i["M"] * i["K"] * 2 / 4.0e12 * 1e6))
        else:
            sig = "__".join(x["export"] for x in grp)
            chains[sig] = [x["export"] for x in grp]
            g.etensor(ev, (1,), wait_count=1)
            name = P + "chain_" + "_".join(x["role"] for x in grp) + f"_{grp[0]['layer']}"
            pin = {"domain_map": "i->(0)"} if any(x["role"] in PIN_ROLES for x in grp) else {}
            g.call_device(name, (1,), hip_link(CHAIN_FILE, f"etx_chain_{sig}", pre_wait=f"etx_pre_chain_{sig}"), resource=res, args=["impargs"],
                          consts=tuple(x["offset"] for x in grp), in_edges={prev[r]: "i->(0)"}, out_edges={ev: "i->(0)"},
                          duration_us=2.0 * len(grp), **pin)
        prev[r] = ev
    if chains:
        with open(CHAIN_FILE, "w") as f:
            f.write("// generated by examples/vllm_llm/model.py (ETX_VL_CHAIN=1): chained launches, one task each\n#pragma once\n")
            f.write(f'#include "{adapters}"\n')
            for sig, exps in chains.items():
                f.write(f'extern "C" __device__ __attribute__((always_inline)) void etx_chain_{sig}(const etx_ctx* c) {{\n')
                f.write("  const unsigned char* base = etx_imp_argblock(c->args, 0);\n")
                f.write("".join(f"  etx_all_blocks_{e}(base + c->cst[{k}]);\n" for k, e in enumerate(exps)))
                f.write("}\n")
                # before the wait: every part's argument block into the scalar cache
                f.write(f'extern "C" __device__ __attribute__((always_inline)) void etx_pre_chain_{sig}(const etx_ctx* c) {{\n')
                f.write("".join(f"  etx_pre_{e}_at(c, c->cst[{k}]);\n" for k, e in enumerate(exps)))
                f.write("}\n")
    for r in range(REQUESTS):
        g.etensor(f"{pre(r)}E_argmax", (1,), wait_count=1)
        g.call_device(f"{pre(r)}argmax", (1,), hip_link(native, "vl_argmax"), resource=res, args=["vl"], consts=(r,),
                      in_edges={prev[r]: "i->(0)"}, out_edges={f"{pre(r)}E_argmax": "i->i"}, duration_us=3.0)

    plan = os.environ.get("ETX_VL_PLAN")
    if plan:
        insts = [i for r, n in order for i in groups_r[r][n]]
        with open(plan, "w") as f:
            f.write(f"# vl plan {d.name}: instance export layer role gx gy gz offset tasks M K wvprgrp rep stride copyoff req; total {total} bytes\n")
            f.write(f"total {total} parts {math.ceil(MAXLEN / PART)} maxlen {MAXLEN} requests {REQUESTS}\n")
            for n, i in enumerate(insts):
                gx, gy, gz = i["grid"]
                f.write(f"{n} {i['export']} {i['layer']} {i['role']} {gx} {gy} {gz} {i['offset']} {i['tasks']} "
                        f"{i.get('M', 0)} {i.get('K', 0)} {i.get('wvprgrp', 0)} {i['rep']} {i['stride']} {i['copyoff']} {i['req']}\n")
    return g


def bindings(**overrides) -> dict:
    return {}


def runtime(bindings, rng=None):
    return {}
