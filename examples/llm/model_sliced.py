"""The same decoder as examples/llm/model.py, built as one slice per XCD (design: docs/PLAN-fusion-win.md, section 2).

Every layer is cut into X = 8 slices pinned to the XCDs with `domain_map`, the way fleet-mi300x's
Chiplet tasks are.  Inside a slice only XCD-local events are used; the slices meet at two device-wide
events per layer (after o_proj, after down / the expert down).  Partial sums are folded in the next
phase's prologue instead of a barrier (fleet's FOLD_PARTIALS, the design's recompute-for-events).

  XCD k owns query heads [qh0_k, qh0_k + nqh_k) and computes the K/V rows of the KV heads they need
  (duplicated across XCDs that share a KV head; they write identical values).
  qkv_s    (X, W)        rows of XCD k's q/k/v slice, split over the W workers     -> E_qkv[x]   local
  post_q_s (X, HMAX)     q-norm + RoPE per owned head                               -> E_post[x]  local
  post_kv_s(X, KVMAX, 2) k-norm + RoPE + KV-cache write per needed KV head          -> E_post[x]  local
  attn_s   (X, HMAX, NC) one owned head x one chunk                                 -> E_attn[x,j] local
  merge_s  (X, HMAX)     combine the chunks of one head                             -> E_merge[x] local
  oproj_s  (X, W)        o_part[x] = Wo[:, cols of x's heads] . o[x's heads]        -> E_o        DEVICE
  gateup_s (X, W)        fold xa = xb + sum o_part (w==0 writes its slice of xa); h rows of slice x -> E_gu[x] local
  down_s   (X, W)        d_part[x] = Wd[:, inter slice x] . h[inter slice x]        -> E_down     DEVICE
  next qkv_s prologue    fold xin = xa + sum d_part (w==0 writes its slice of xb)
MoE layers: router_s (X, W) every XCD computes all E logits (no barrier); egu_s / edn_s (X, W) handle
the top-k slots s = x, x+X, ... ; d_part is per slot and the fold sums the slots in ascending expert id.
"""
from __future__ import annotations

import math
import os

from etx.frontends import hip_link
from etx.ir import Graph, Resource

from examples.llm import model as M

T = "examples/llm/tiles.hip"
X = 8


def split(n: int, parts: int, unit: int = 1) -> list[tuple[int, int]]:
    """(start, count) per part: n/unit units split evenly, the remainder to the first parts."""
    u = n // unit
    base, rem = divmod(u, parts)
    out, s = [], 0
    for k in range(parts):
        c = base + (1 if k < rem else 0)
        out.append((s * unit, c * unit)); s += c
    return out


def slices(d: M.Dims) -> dict:
    """The per-XCD tables the tiles, the packer and the host all derive from the dims."""
    q = split(d.NH, X)
    kv = []
    for qh0, nqh in q:
        if nqh == 0:
            kv.append((0, 0))
        else:
            kv0, kv1 = qh0 // d.G, (qh0 + nqh - 1) // d.G
            kv.append((kv0, kv1 - kv0 + 1))
    inter = split(d.INTER, X, unit=128) if not d.moe else [(0, 0)] * X    # K of the down slices stays streamable (K % 128 == 0)
    hs = split(d.H, X, unit=8)
    return {"q": q, "kv": kv, "inter": inter, "h": hs,
            "HMAX": max(c for _, c in q), "KVMAX": max(c for _, c in kv)}


def build() -> Graph:
    d = M.load_dims()
    p = M.partition(d)
    S = slices(d)
    W = M.workers(d) // X
    ctx, chunk = M._ctx_chunk()
    NC = ctx // chunk
    res = Resource(threads=256, vgpr=128, agpr=0, lds_bytes=M.lds_bytes(d))
    g = Graph(f"llm_sliced_{d.name}")
    g.tensor("llm", (1,), role="weight", bytes_per_elem=8)
    HMAX, KVMAX = S["HMAX"], S["KVMAX"]

    PF = {"llm_s_qkv": "llm_s_qkv_pf", "llm_s_oproj": "llm_s_oproj_pf", "llm_s_gateup": "llm_s_gateup_pf", "llm_s_down": "llm_s_down_pf",
          "llm_s_egu": "llm_s_egu_pf"} if os.environ.get("ETX_LLM_PREFETCH", "1") != "0" else {}

    def grid(name, shape, sym, L, ins, outs, dur, dmap, wmap=None):
        kw = {"domain_map": dmap}
        if wmap:
            kw["worker_map"] = wmap
        return g.call_device(name, shape, hip_link(T, sym, prefetch=PF.get(sym)), resource=res, args=["llm"], consts=(L,),
                             in_edges=ins, out_edges=outs, duration_us=dur, duration_cv=0.05, **kw)

    def us(nbytes):
        return max(1.0, nbytes / (X * W) / (M.HBM_BPS / 304) * 1e6)

    g.etensor("E_embed", (1,), wait_count=1)
    g.call_device("embed", (1,), hip_link(T, "llm_embed"), resource=res, args=["llm"], consts=(-1,),
                  out_edges={"E_embed": "i->i"}, duration_us=1.0)
    e_in, e_map = "E_embed", "xw->(0)"
    for L in range(d.L):
        # one fold task per XCD is the only device-scope consumer: it sums the partials once into an
        # XCD-local vector; the 37 other workers wait on a local event and read that (measured: folding
        # in every task cost 35 us per phase, 304 tasks each re-fetching 147 KB after their L2 invalidate)
        g.etensor(f"E_fold_{L}", (X,), wait_count=1)
        g.etensor(f"E_qkv_{L}", (X,), wait_count=W)
        g.etensor(f"E_post_{L}", (X,), wait_count=HMAX + 2 * KVMAX)
        g.etensor(f"E_attn_{L}", (X,), wait_count=NC)
        g.etensor(f"E_merge_{L}", (X,), wait_count=HMAX)
        g.etensor(f"E_o_{L}", (1,), wait_count=X * W)
        g.etensor(f"E_folo_{L}", (X,), wait_count=1)
        g.etensor(f"E_down_{L}", (1,), wait_count=X * W)
        grid(f"fold_{L}", (X, 1), "llm_s_fold_d", L, {e_in: e_map}, {f"E_fold_{L}": "xw->x"}, 2.0, "xw->x")
        grid(f"qkv_{L}", (X, W), "llm_s_qkv", L, {f"E_fold_{L}": "xw->x"}, {f"E_qkv_{L}": "xw->x"}, us(d.NQKV * d.H * 2), "xw->x", "xw->w")
        grid(f"post_q_{L}", (X, HMAX), "llm_s_post_q", L, {f"E_qkv_{L}": "xj->x"}, {f"E_post_{L}": "xj->x"}, 1.0, "xj->x")
        grid(f"post_kv_{L}", (X, KVMAX, 2), "llm_s_post_kv", L, {f"E_qkv_{L}": "xjs->x"}, {f"E_post_{L}": "xjs->x"}, 1.0, "xjs->x")
        grid(f"attn_{L}", (X, NC), "llm_s_attn", L, {f"E_post_{L}": "xc->x"}, {f"E_attn_{L}": "xc->x"}, 6.0, "xc->x")
        grid(f"merge_{L}", (X, HMAX), "llm_s_merge", L, {f"E_attn_{L}": "xj->x"}, {f"E_merge_{L}": "xj->x"}, 1.0, "xj->x")
        grid(f"oproj_{L}", (X, W), "llm_s_oproj", L, {f"E_merge_{L}": "xw->x"}, {f"E_o_{L}": "xw->(0)"}, us(d.H * d.NH * d.HD * 2), "xw->x", "xw->w")
        grid(f"folo_{L}", (X, 1), "llm_s_fold_o", L, {f"E_o_{L}": "xw->(0)"}, {f"E_folo_{L}": "xw->x"}, 2.0, "xw->x")
        if d.moe:
            g.etensor(f"E_r_{L}", (X,), wait_count=W)
            g.etensor(f"E_egu_{L}", (X,), wait_count=W)
            grid(f"router_{L}", (X, W), "llm_s_router", L, {f"E_folo_{L}": "xw->x"}, {f"E_r_{L}": "xw->x"}, us(X * d.E * d.H * 2), "xw->x", "xw->w")
            grid(f"egu_{L}", (X, W), "llm_s_egu", L, {f"E_r_{L}": "xw->x"}, {f"E_egu_{L}": "xw->x"}, us(d.TOPK * 2 * d.MI * d.H * 2), "xw->x", "xw->w")
            grid(f"edn_{L}", (X, W), "llm_s_edn", L, {f"E_egu_{L}": "xw->x"}, {f"E_down_{L}": "xw->(0)"}, us(d.TOPK * d.H * d.MI * 2), "xw->x", "xw->w")
        else:
            g.etensor(f"E_gu_{L}", (X,), wait_count=W)
            grid(f"gateup_{L}", (X, W), "llm_s_gateup", L, {f"E_folo_{L}": "xw->x"}, {f"E_gu_{L}": "xw->x"}, us(2 * d.INTER * d.H * 2), "xw->x", "xw->w")
            grid(f"down_{L}", (X, W), "llm_s_down", L, {f"E_gu_{L}": "xw->x"}, {f"E_down_{L}": "xw->(0)"}, us(d.H * d.INTER * 2), "xw->x", "xw->w")
        e_in, e_map = f"E_down_{L}", "xw->(0)"
    n_lm = math.ceil(d.V / p["lm"])
    g.etensor("E_lmf", (1,), wait_count=1)
    g.etensor("E_lm", (1,), wait_count=n_lm)
    g.etensor("E_argmax", (1,), wait_count=1)
    g.call_device("lmfold", (1,), hip_link(T, "llm_s_lmfold"), resource=res, args=["llm"], consts=(d.L,),
                  in_edges={e_in: "i->(0)"}, out_edges={"E_lmf": "i->i"}, duration_us=2.0)
    g.call_device("lmhead", (n_lm,), hip_link(T, "llm_s_lmhead"), resource=res, args=["llm"], consts=(-1,),
                  in_edges={"E_lmf": "i->(0)"}, out_edges={"E_lm": "i->(0)"}, duration_us=M._us(d.V * d.H * 2, n_lm))
    g.call_device("argmax", (1,), hip_link(T, "llm_argmax"), resource=res, args=["llm"], consts=(-1,),
                  in_edges={"E_lm": "i->i"}, out_edges={"E_argmax": "i->i"}, duration_us=2.0)
    return g


def bindings(**overrides) -> dict:
    return {}


def runtime(bindings, rng=None):
    return {}


if __name__ == "__main__":
    import json
    d = M.load_dims()
    print(json.dumps({"W": M.workers(d) // X, **{k: v for k, v in slices(d).items()}}))
