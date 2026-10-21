#!/usr/bin/env python3
"""Operation-level cost model of vLLM's decode kernels on MI300X, in vLLM's form and in ETX's two forms
(docs/qwen/MODEL-vllm-kernels.md).

Every kernel at batch 1 is a chain of dependent steps, so one block's time is a sum over its critical path:

  T_block = S * t_s              scalar round trips (kernel arguments; ETX adds its descriptor / argument-table ones)
          + V * t_v              dependent vector round trips of fresh data (activations written by the previous kernel)
          + stream               weights / KV: max(bandwidth bound, latency bound)   (Little's law)
                                   bandwidth bound = active CUs x bytes per block / BW
                                   latency bound   = trips x (t_v + per-trip exposed work)
                                   trips = bytes per block / bytes in flight per CU; the exposed work per trip is the
                                   fused_moe tile's LDS write / barrier / read gap (0.33 us) plus, in ETX's form, the
                                   software barriers of that tile (a barrier inside the K loop is on the critical path)
          + bar * t_bar          hardware barriers (__syncthreads, measured 77 ns)
          + swbar * t_swbar      ETX's software slot barriers (LDS counter + poll; ~0.26 us from the instruction count)
          + shfl * t_shfl        LDS cross-lane steps (ds_bpermute, measured 56 ns)
          + dpp * t_dpp          in-register cross-lane steps
          + instr * t_i          the rest of the instructions one wave issues on the critical path

  vLLM launch   = F + T_block(vLLM)                       F: the kernel boundary of a HIP graph (dispatch, end release)
  ETX unfused   = F + T_block(ETX) + A_unf * t_s          A_unf: descriptor, argument table, argument block
  ETX task      = T_block(ETX) + A_task * t_s             the hand-off is the simulator's (etx.sim.xcd)

The operation counts are the inventories (examples/vllm_llm/ops/*.md, from the source and the assembly); the unknown
latencies F, t_s, t_v, t_i are fitted to vLLM's own launches only, so everything about ETX's forms is a prediction.

  python3 -m examples.vllm_llm.kmodel [--measured results_2026-09-28/measured.json] [--json out.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import dataclass, field, replace

BW0 = 4.2e6         # bytes/us, achievable HBM rate of many streaming CUs (hw_bench stream, WG medians); refitted
HW = dict(t_bar=0.077, t_shfl=0.056, t_dpp=0.004, t_swbar=0.26)


@dataclass
class Kern:
    role: str
    S: float = 2            # scalar round trips on the critical path (vLLM form)
    V: float = 1            # dependent vector round trips of fresh data
    bytes_block: float = 0  # bytes one streaming block reads (weights, KV)
    mall_block: float = 0   # of bytes_block, this many come from the Infinity Cache (MALL), not HBM:
                            # the KV cache (104 MB) fits in MALL (256 MB); attn streams at MALL rate
                            # (docs/qwen/BUDGET-2026-10-01.md: 2.16 MB/layer in 6.28 us = 344 GB/s)
    inflight: float = 0     # bytes one active CU keeps in flight while streaming
    active: int = 1         # CUs (blocks) that stream
    gap: float = 0          # exposed work per trip not overlapped with memory (fused_moe tile: 0.33 us)
    sw_per_trip: float = 0  # software barriers per trip in ETX's form (on the critical path of every trip)
    bar: float = 0          # hardware barriers per block (vLLM form)
    sw_etx: float = 0       # barriers that become software slot barriers in ETX's form
    shfl: float = 0
    dpp: float = 0
    instr: float = 200
    etx_extra_V: float = 0  # ETX form only: extra dependent vector round trips (fused_moe prologue)
    chain_rounds: int = 1   # rounds of blocks when the whole launch runs on one workgroup (a chain part)
    note: str = ""



# Inventories (examples/vllm_llm/ops/hip.md section 13, ops/fused_moe.md): per block, at batch 1, context 1024
MOE = [
    Kern("in_norm", S=2, V=2, bar=2, shfl=10, instr=250, note="fused_add_rms_norm 1x1024, 256 threads with data"),
    Kern("qkv", S=4, V=1, bytes_block=73.6e3, inflight=36e3, active=285, bar=1, dpp=16, instr=245, note="wvSplitK 5120x2048, 9 waves stream"),
    Kern("q_norm", S=4, V=2, bar=2, sw_etx=2, shfl=10, instr=350, note="rms_norm_3d 32 heads x 16 threads"),
    Kern("k_norm", S=4, V=2, bar=2, sw_etx=2, shfl=10, instr=350, note="rms_norm_3d 4 heads"),
    Kern("rope", S=3, V=5, instr=400, note="rotary neox, 2 B scalar loads, 4 passes over q + 1 over k"),
    Kern("cache", S=3, V=9, instr=350, note="reshape_and_cache, V copy 8 serial 2 B load/store rounds"),
    Kern("qknr", S=2, V=2, shfl=9, instr=250, chain_rounds=2,
         note="fused_qk_norm_rope (ETX_VL_QKNR): from the source, not yet the ISA: qkv and position loads in parallel, "
              "cos/sin after the position; 5-step 32-lane reduce + 4 rope exchanges; 5 blocks = 2 rounds of 4 slots"),
    Kern("attn", S=4, V=7, bar=7, sw_etx=7, shfl=4, instr=720, note="ll4mi mfma16 GQA 8, 4 of 5 partitions, 2 scratch reloads"),
    Kern("reduce", S=3, V=1, bar=1, sw_etx=1, shfl=12, instr=430, note="attention reduce, 32 heads"),
    Kern("o_proj", S=4, V=1, bytes_block=65.5e3, inflight=16e3, active=256, bar=1, dpp=16, instr=350, note="wvSplitK 2048x4096, 4 waves stream"),
    Kern("post_norm", S=2, V=2, bar=2, shfl=10, instr=250, note="fused_add_rms_norm"),
    Kern("router", S=4, V=1, bytes_block=16.4e3, inflight=16e3, active=32, bar=1, dpp=8, instr=195, note="wvSplitK 128x2048 YTILE 1, 32 blocks with rows"),
    Kern("topk", S=2, V=5, shfl=40, instr=1500, note="topkGating, 16 lanes, 8 x (argmax + 4 bpermute steps), renorm re-read"),
    Kern("moe_up", S=2, V=2, bytes_block=262e3, inflight=16.4e3, active=192, gap=0.33, sw_per_trip=2, bar=31, sw_etx=31, instr=0, etx_extra_V=3,
         note="Triton fused_moe gate_up, 256 KB per program, one real row of 16"),
    Kern("moe_act", S=2, V=1, instr=340, note="silu 8 rows x 96 threads, expf + IEEE divide"),
    Kern("moe_down", S=2, V=3, bytes_block=98.3e3, inflight=16.4e3, active=256, gap=0.33, sw_per_trip=2, bar=11, sw_etx=11, instr=0, etx_extra_V=3,
         note="Triton fused_moe down, 96 KB per program, x routing weight"),
    Kern("moe_sum", S=2, V=1, instr=300, note="moe_sum 8 x 16 B in flight"),
]
Q15 = [
    Kern("in_norm", S=2, V=2, bar=2, shfl=10, instr=250),
    Kern("qkv", S=4, V=2, bytes_block=24.6e3, inflight=16e3, active=256, bar=1, dpp=16, instr=265, note="+ bias round trip"),
    Kern("rope", S=3, V=3, instr=250),
    Kern("cache", S=3, V=9, instr=350),
    Kern("attn", S=4, V=7, bar=7, sw_etx=7, shfl=4, instr=720, note="GQA 6"),
    Kern("reduce", S=3, V=1, bar=1, sw_etx=1, shfl=12, instr=430),
    Kern("o_proj", S=4, V=1, bytes_block=24.6e3, inflight=16e3, active=192, bar=1, dpp=16, instr=245),
    Kern("post_norm", S=2, V=2, bar=2, shfl=10, instr=250),
    Kern("gate_up", S=4, V=1, bytes_block=181e3, inflight=60e3, active=304, bar=1, dpp=32, instr=400),
    Kern("act", S=2, V=2, instr=340),
    Kern("down", S=4, V=2, bytes_block=143e3, inflight=16e3, active=192, bar=1, dpp=16, instr=620),
]
# ETX_VL_DSL (section 12, estimates from the chains, not from ISA): the fused attention task = the fused q/k norm + rope
# chain (2 V: qkv || position, then cos/sin) + the cache write of its k/v rows (1 V, no serial 2 B copy) + paged attention
# over one partition (7 V, 7 hardware barriers); top-k = one 128-logit softmax + 8 argmax rounds over 64 lanes; in_norm
# reads the 8 expert rows (+1 V) instead of a separate moe_sum
DSL_MOE = [replace(k) for k in MOE if k.role not in ("q_norm", "k_norm", "rope", "cache", "attn", "topk", "moe_sum", "in_norm")] + [
    Kern("in_norm", S=2, V=3, bar=2, shfl=10, instr=300, note="fused_add_rms_norm reading the 8 expert rows (moe_sum folded in)"),
    Kern("attn", S=3, V=10, bytes_block=128e3, mall_block=128e3, inflight=16e3, active=20, bar=9, shfl=13, instr=1100,
         note="DSL: q/k norm + rope + cache + attention of one (partition, kv head)"),
    Kern("topk", S=2, V=2, shfl=24, instr=400, note="DSL: softmax + top-8 over 128 logits, 64 lanes"),
]
# + the streaming GEMVs generated by the DSL (section 13): 64 KB in flight per workgroup, no per-trip gap, one barrier
# for the reduction; bytes and active CUs as vLLM's launches (the lf microbenchmark: 262 KB x 192 WGs at 64 KB in flight
# 3.61 TB/s, 98 KB x 256 3.53, 98 KB x 296 3.57)
DSL_GEMV = [replace(k) for k in DSL_MOE if k.role not in ("qkv", "o_proj", "router", "moe_up", "moe_down")] + [
    Kern("qkv", S=2, V=1, bytes_block=69e3, inflight=64e3, active=304, bar=1, shfl=6, instr=300, note="DSL GEMV 5120 x 2048"),
    Kern("o_proj", S=2, V=1, bytes_block=55e3, inflight=64e3, active=304, bar=1, shfl=6, instr=300, note="DSL GEMV 2048 x 4096"),
    Kern("router", S=2, V=1, bytes_block=16.4e3, inflight=16.4e3, active=32, bar=1, shfl=6, instr=200, note="DSL GEMV 128 x 2048"),
    Kern("moe_up", S=2, V=2, bytes_block=262e3, inflight=64e3, active=192, bar=1, shfl=6, instr=600, note="DSL expert GEMV, 64 rows of one expert (top-k ids: +1 V)"),
    Kern("moe_down", S=2, V=2, bytes_block=98.3e3, inflight=64e3, active=256, bar=1, shfl=6, instr=400, note="DSL expert GEMV x routing weight"),
]
# phase 1 (ETX_VL_DSL=2): DSL GEMVs, top-k and the moe_sum fold; attention as today (fused q/k norm + rope, cache, vLLM's attention)
DSL_P1 = [k for k in DSL_GEMV if k.role != "attn"] + [k for k in MOE if k.role in ("attn", "cache", "qknr")]
# ETX_VL_DSL_GEMV=1 alone (the first build): vLLM's small kernels as today, the DSL GEMVs, no moe_act
GEMV_ONLY = [k for k in MOE if k.role not in ("qkv", "o_proj", "router", "moe_up", "moe_down", "moe_act")] + \
            [k for k in DSL_GEMV if k.role in ("qkv", "o_proj", "router", "moe_up", "moe_down")]
BUILD1 = [k for k in GEMV_ONLY if k.role != "topk"] + [k for k in DSL_MOE if k.role == "topk"]   # + etx.ops.topk
# + ETX_VL_DSL_ATTN=head: etx.ops.attnprep per kv head (9 waves: row load, cos/sin after the position, stores), vLLM's
# attention and reduce as they are
BUILD2 = [k for k in BUILD1 if k.role not in ("qknr", "cache")] + [Kern("prep", S=4, V=3, shfl=8, instr=300, note="DSL attention prep of one kv head")]
# + ETX_VL_DSL_FOLD: the consumers' prologues: qkv reads 8 expert rows + residual and norms (+3 V, 2 barriers, 10 shfl),
# router norms o + residual (+2 V), moe_up computes the top-k from the logits (+2 V, 24 shfl)
_b2 = {k.role: k for k in BUILD2}
BUILD3 = [k for k in BUILD2 if k.role not in ("qkv", "router", "moe_up", "in_norm", "post_norm", "topk", "moe_sum")] + [
    replace(_b2["qkv"], V=_b2["qkv"].V + 3, bar=_b2["qkv"].bar + 2, shfl=_b2["qkv"].shfl + 10, instr=_b2["qkv"].instr + 250, note="DSL GEMV + folded in_norm"),
    replace(_b2["router"], V=_b2["router"].V + 2, bar=_b2["router"].bar + 2, shfl=_b2["router"].shfl + 10, instr=_b2["router"].instr + 250, note="DSL GEMV + folded post_norm"),
    replace(_b2["moe_up"], V=_b2["moe_up"].V + 2, shfl=_b2["moe_up"].shfl + 24, instr=_b2["moe_up"].instr + 400, note="DSL expert GEMV + folded top-k"),
    _b2["in_norm"]]
# the cost entry of a task from what its type links (plan.json "symbol"), so any mix of imported kernels and DSL ops
# costs itself without a hand-picked model name (etx_tile_[rep_|x_]<export>): DSL exports map to their entries, the
# fused q/k norm + rope to "qknr", everything else to the vLLM entry of the role
def kern_for_symbol(symbol: str, role: str, base: list) -> "Kern | None":
    exp = re.sub(r"^etx_tile_(rep_|x_)?", "", symbol)
    if symbol.endswith("_fold"):
        return {k.role: k for k in BUILD3}.get(role)
    dsl = {"dsl_gemv": {k.role: k for k in DSL_GEMV}, "dsl_moe_up": {k.role: k for k in DSL_GEMV}, "dsl_moe_down": {k.role: k for k in DSL_GEMV},
           "dsl_topk": {k.role: k for k in DSL_MOE}, "dsl_attnprep": {k.role: k for k in BUILD2}, "dsl_attn": {k.role: k for k in DSL_MOE}}
    if exp in dsl:
        return dsl[exp].get(role)
    return {k.role: k for k in base}.get(role)


MODELS = {"qwen3-30b-a3b": MOE, "qwen2.5-1.5b": Q15, "qwen3-30b-a3b-dsl": DSL_MOE, "qwen3-30b-a3b-dslgemv": DSL_GEMV,
          "qwen3-30b-a3b-dslp1": DSL_P1, "qwen3-30b-a3b-gemvonly": GEMV_ONLY, "qwen3-30b-a3b-build1": BUILD1,
          "qwen3-30b-a3b-build2": BUILD2, "qwen3-30b-a3b-build3": BUILD3}


@dataclass
class Params:
    F: float = 1.6
    t_s: float = 0.4
    t_v: float = 0.6
    t_i: float = 0.002
    A_unf: float = 0.5      # ETX unfused: us for the descriptor + argument-table round trips; fitted, reported as such
    A_task: float = 1.0     # ETX task: us from ready to the kernel's first own load (acquire, argument table and block
                            # read cold after the L2 invalidate): fitted to ETX's small tasks, reported as such
    bw: float = BW0
    hw: dict = field(default_factory=lambda: dict(HW))


def stream_time(k: Kern, p: Params, etx: bool) -> tuple[float, str]:
    """trips x (memory time of one trip + the work of a trip that is not overlapped with memory). One trip's memory
    time is the loaded latency: max(idle round trip, bytes in flight on the GPU / BW) (Little's law); trips are
    fractional (bytes per block / bytes in flight), since CUs are not in step."""
    if not k.bytes_block:
        return 0.0, ""
    trips = k.bytes_block / k.inflight
    mem = max(p.t_v, k.active * k.inflight / p.bw)
    exposed = k.gap + (k.sw_per_trip * (p.hw["t_swbar"] - p.hw["t_bar"]) if etx else 0.0)
    return trips * (mem + exposed), ("bandwidth" if mem > p.t_v else "latency")


def terms(k: Kern, p: Params, form: str) -> dict:
    etx = form != "vllm"
    st, bound = stream_time(k, p, etx)
    sw_outside = max(0.0, k.sw_etx - (k.sw_per_trip * (k.bytes_block / k.inflight if k.inflight else 0))) if etx else 0.0
    d = {
        "front": p.F if form in ("vllm", "etx_unf") else 0.0,
        "etx_args": (p.A_unf if form == "etx_unf" else p.A_task if form == "etx_task" else 0.0),
        "scalar": k.S * p.t_s,
        "chain": (k.V + (k.etx_extra_V if etx else 0)) * p.t_v,
        "stream": st,
        "sync": (k.bar - (k.sw_etx if etx else 0)) * p.hw["t_bar"] * (0 if k.bytes_block and k.gap else 1)
                + sw_outside * p.hw["t_swbar"] + k.shfl * p.hw["t_shfl"] + k.dpp * p.hw["t_dpp"],
        "issue": k.instr * p.t_i,
    }
    d["total"] = sum(d.values())
    d["bound"] = bound
    return d


def _err(pts, p, form):
    import math
    return sum((math.log(max(terms(k, p, form)["total"], 1e-3)) - math.log(v)) ** 2 for k, v in pts)


def _pts(measured, form, key=None):
    pts = []
    for model, ks in MODELS.items():
        for k in ks:
            v = measured.get(model, {}).get(form, {}).get(k.role)
            if isinstance(v, dict):
                v = v.get(key)
            if v:
                pts.append((k, v))
    return pts


def _frange(a, b, step):
    n = int(round((b - a) / step))
    return [round(a + i * step, 4) for i in range(n + 1) if a + i * step <= b + 1e-9]


def fit(measured: dict) -> Params:
    """Step 1: F, t_s, t_v, t_i on vLLM's launches (both models). Step 2, ETX-specific: A_unf (extra scalar round
    trips of the unfused launch) on ETX's unfused launches of the kernels without software barriers, A_task (ready ->
    first own load) on ETX's fused small tasks. Everything else about ETX's forms is predicted."""
    pts = _pts(measured, "vllm")
    best = None
    for F, t_s, t_v, t_i, bw in itertools.product(_frange(0.8, 3.0, 0.1), [0.0, 0.1, 0.2, 0.3], _frange(0.2, 1.5, 0.05),
                                                  [0.0005, 0.001, 0.0015, 0.002, 0.003], _frange(3.6e6, 5.2e6, 0.2e6)):
        p = Params(F=round(F, 2), t_s=t_s, t_v=round(t_v, 2), t_i=t_i, bw=bw)
        e = _err(pts, p, "vllm")
        if best is None or e < best[0]:
            best = (e, p)
    p = best[1]
    unf = [(k, v) for k, v in _pts(measured, "etx_unf") if not k.sw_etx]
    p.A_unf = min(_frange(0, 3.0, 0.05), key=lambda a: _err(unf, replace(p, A_unf=a), "etx_unf"))
    small = [(k, v) for k, v in _pts(measured, "etx_task", "body_median") if not k.sw_etx and not k.bytes_block]
    p.A_task = min(_frange(0, 3.01, 0.1), key=lambda a: _err(small, replace(p, A_task=a), "etx_task"))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", default="results_2026-09-28/measured.json")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    m = json.load(open(a.measured))
    p = fit(m)
    print(f"fitted on vLLM's launches: F {p.F} us, t_s {p.t_s} us, t_v {p.t_v} us, t_i {p.t_i * 1e3:.1f} ns/instr; "
          f"BW {p.bw / 1e6:.1f} TB/s; ETX-specific fits: A_unf {p.A_unf:.2f} us, A_task {p.A_task:.1f} us; fixed: {p.hw}")
    out = {"params": p.__dict__, "models": {}}
    for model, ks in MODELS.items():
        mm = m.get(model, {})
        print(f"\n{model}: predicted / measured (us); ETX task = body only (the hand-off is the simulator's)")
        print(f"  {'role':10s} {'vLLM':>13s} {'ETX unfused':>15s} {'ETX task':>15s}   ETX task terms")
        rows = []
        for k in ks:
            tv, tu, tt = (terms(k, p, f) for f in ("vllm", "etx_unf", "etx_task"))
            mv = mm.get("vllm", {}).get(k.role)
            mu = mm.get("etx_unf", {}).get(k.role)
            task = mm.get("etx_task", {})
            mt = (task.get(k.role) or {}).get("body_median")
            f = lambda pr, me: f"{pr:6.2f}/{me:6.2f}" if me else f"{pr:6.2f}/  -   "
            ex = " ".join(f"{n}={v:.2f}" for n, v in tt.items() if n not in ("total", "bound") and v >= 0.05) + (f" [{tt['bound']}]" if tt["bound"] else "")
            print(f"  {k.role:10s} {f(tv['total'], mv):>13s} {f(tu['total'], mu):>15s} {f(tt['total'], mt):>15s}   {ex}")
            rows.append({"role": k.role, "note": k.note, "pred": {"vllm": tv, "etx_unf": tu, "etx_task": tt},
                         "meas": {"vllm": mv, "etx_unf": mu, "etx_task": mt}})
        sv = sum(r["pred"]["vllm"]["total"] for r in rows); su = sum(r["pred"]["etx_unf"]["total"] for r in rows)
        mv = sum(r["meas"]["vllm"] or 0 for r in rows); mu = sum(r["meas"]["etx_unf"] or 0 for r in rows)
        print(f"  {'layer':10s} vLLM {sv:6.1f}/{mv:6.1f}   ETX unfused {su:6.1f}/{mu:6.1f}   (roles with a vLLM counterpart in the vLLM sum)")
        out["models"][model] = rows
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=lambda o: o.__dict__)


if __name__ == "__main__":
    main()
