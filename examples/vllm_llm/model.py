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
from examples.vllm_llm import folds  # noqa: E402

# ETX_VL_CU: workgroups a DSL dense phase spreads over (one task per workgroup). 304 = every CU, which is what
# vLLM's wvSplitK uses, but hw_bench stream (2026-10-01) shows 304 is oversubscribed: at the gate_up shape the
# per-workgroup rate collapses from ~31 KB/us (<= 74 workgroups) to 8.9 KB/us at 296, and the aggregate optimum
# sits near 192. Must be a multiple of NX (8) so each XCD gets the same number of workers.
# ETX_VL_IMPORTED=1: every phase uses vLLM's imported kernel instead of a DSL tile. The comparison, not a default:
# measured 2026-10-01 round 24 at 6.435 ms against 4.937, and with imported kernels fusion still LOSES (-0.171),
# because an imported kernel's loads come from vLLM's source and its task must keep the acquire that invalidates L2.
IMPORTED = os.environ.get("ETX_VL_IMPORTED", "0") == "1"


def _on(name: str, universal: bool = True) -> bool:
    """A rule that is on unless this build is the all-imported comparison or someone passes `=0`."""
    v = os.environ.get(name)
    if v is not None:
        return v == "1"
    return universal and not IMPORTED


# UNIVERSAL (see docs/OPTIMISATIONS.md): on for every model, `=0` only to reproduce a measurement.
CU = int(os.environ.get("ETX_VL_CU", "256"))    # MI300X has 304 compute units
BS = 16                       # KV block size (vLLM default on ROCm)
PART = 256                    # paged attention partition size (fixed in paged_attention_custom_launcher).
# Sets only parts = ceil(MAXLEN / PART) = gridDim.y; the partition SIZE is a hardcoded T_PAR_SIZE = 256 in the
# kernel (vllm/csrc/rocm/attention.cu:362), and a partition past the sequence early-exits (:366-369). So parts
# * 256 must cover seq_len or the tail is never attended and reduce reads unwritten partials: 256 is the minimum
# correct value at MAXLEN 1072, 128 only adds early-exiting tasks, 512 is incorrect. Not a tuning knob
# (results_2026-10-01/round11/FINDING.md).
THREADS = 1024                # megakernel workgroup = the largest imported block (wvSplitK 64 x 16)
MOE_BN = 64                   # fused_moe_kernel BLOCK_SIZE_N of vLLM's default config at batch 1 (examples/vllm_llm/triton)
MAXLEN = int(os.environ.get("ETX_VL_MAXLEN", "1072"))    # vLLM's max_model_len for the benchmark (prompt + 32 + 16)
# ETX_VL_PIN=1: the small launches between qkv and o_proj (q/k norm, rope, cache, attention, reduce) all run on
# XCD 0, so the events between them are XCD-local (a wait on L2 of one XCD) instead of device-wide
PIN_ROLES = {"q_norm", "k_norm", "rope", "qknr", "cache", "attn", "reduce"} if os.environ.get("ETX_VL_PIN", "0") == "1" else set()
# ETX_VL_CHAIN=1: consecutive launches that each fit on one workgroup run as ONE task, back to back with a
# __syncthreads between them instead of an event (q/k norm, rope and the cache write; at most 4 per task)
CHAIN_ROLES = ("q_norm", "k_norm", "rope", "qknr", "cache")
CHAIN = _on("ETX_VL_CHAIN", universal=True)
# ETX_VL_QKNR=1: q/k norm + rope as ONE launch of vLLM's fused_qk_norm_rope (tensorrt_llm::kernels::fusedQKNormRopeKernel,
# what vLLM runs with compilation_config pass_config enable_qk_norm_rope_fusion): one 32-lane warp per head, 8 per
# 256-thread block, in place on qkv; replaces rms_norm_3d x2 + rotary_embedding
# ETX_VL_QKNR=grid: the fused launch as its own grid (its blocks on several CUs), not chained with the cache write
QKNR = os.environ.get("ETX_VL_QKNR", "0") in ("1", "grid")
# ETX_VL_DSL=1 (simulation only, docs/MODEL-vllm-kernels.md section 12): the latency-bound small ops as ETX tile ops with
# their chains fused: q/k norm + rope + cache write + attention in one task per (partition, kv head); top-k as one tile;
# moe_sum folded into the next layer's in_norm. The streaming ops (GEMVs, fused_moe) stay vLLM's.
DSL = os.environ.get("ETX_VL_DSL", "0") in ("1", "2")
# ETX_VL_DSL_GEMV=1: the streaming GEMVs generated by etx.ops.gemv (64 KB in flight per CU) in place of wvSplitK and the
# Triton fused_moe launches; silu fused into the expert gate_up epilogue (no moe_act launch). Tiles: ETX_VL_DSL_TILES.
DSL_GEMV = _on("ETX_VL_DSL_GEMV")
DSL_TOPK = _on("ETX_VL_DSL_TOPK")       # the router's softmax + top-k as etx.ops.topk
# ETX_VL_DSL_ATTN=head: the attention block by kv head: etx.ops.attnprep (q/k norm + rope + cache write of one head) on
# XCD h, the head's attention partitions and its reduce on XCD h too, so those two hand-offs are XCD-local; attention
# and reduce stay vLLM's kernels (attention at one block per task, ETX_VL_SLOTS=<attn>=1; reduce's slots must equal G)
# DERIVED: `dsl_attnprep` fuses the q/k norm, the rope and the cache write of one kv head, so it applies only to a
# model that HAS a q/k norm and whose G + 2 heads fit in the workgroup's waves. Defaulting it to "head" unconditionally
# made Qwen2.5-1.5B (no q/k norm) refuse to plan at all. `_attn_head(d)` asks the shapes; the env still forces either way.
_ATTN_ENV = os.environ.get("ETX_VL_DSL_ATTN")


def _attn_head(d) -> bool:
    if _ATTN_ENV is not None:
        return _ATTN_ENV == "head"
    return not IMPORTED and bool(d.qk_norm) and d.G + 2 <= THREADS // 64


DSL_ATTNHEAD = _ATTN_ENV == "head"      # module-level view; instances() re-derives it per model below
# ETX_VL_DSL_FOLD=1 (simulation so far): the one-block producers folded into their consumers' prologues: every qkv task
# sums the 8 expert rows + residual and norms them itself, every router task norms o + residual, every moe_up task
# recomputes the top-k from the logits (deterministic), the next layer reads the expert rows: no in_norm (L > 0),
# post_norm, topk or moe_sum launches (the design's rematerialisation: redundant parallel work for a serial event)
# DERIVED by default (examples/vllm_llm/folds.py): a fold deletes a one-block producer phase and pays for it with
# rematerialised work in every consumer task, so whether it wins depends on the model's shapes, not on the phase.
# `ETX_VL_DSL_FOLD=1/0` still forces it either way for an A/B; unset means "ask the cost model".
# A fold is a DSL-tile capability (the folded prologue lives in dsl_gemv_fold), so the all-imported build has none.
_FOLD_ENV = os.environ.get("ETX_VL_DSL_FOLD")
DSL_FOLD = _FOLD_ENV == "1" if _FOLD_ENV is not None else not IMPORTED
# ETX_VL_FOLD_TOPK=0 keeps the norm folds but restores the one top-k launch per layer: the fold made all 448 expert
# tasks of a layer recompute the same 128-logit softmax + top-8 on wave 0 while 15 waves waited at the barrier, and
# the ablation measured that at 14.1 us/layer = 0.676 ms/token (results_2026-10-01/round7). One launch costs the
# phase floor plus a hand-off, about 4.4 us/layer.
# Derived too: the top-k fold loses on Qwen3-30B-A3B (448 expert tasks re-reading the same 128 logits uncached) but
# a model with few experts or few expert tasks could flip it, so the comparison is made from the shapes.
_FOLD_TOPK_ENV = os.environ.get("ETX_VL_FOLD_TOPK")
# ETX_VL_TOUCH=1: the dense DSL GEMVs get the megakernel's touch hook (etx.ops.gemv dsl_touch_plain): while wave 0 waits
# for the task's event, the other waves pull the task's first weight trip into L2 (docs/MODEL-vllm-kernels.md section 18)
TOUCH = _on("ETX_VL_TOUCH")
# ETX_VL_COH=1: the DSL tiles access their activations device-coherently (etx.ops.coh, built with -DETX_DSL_COH), so
# the megakernel emits no acquire buffer_inv for them and the caches survive the phase boundary (docs/RESULTS-2026-09-30.md)
COH = _on("ETX_VL_COH") and not IMPORTED   # imported tiles cannot have it
# ETX_VL_ATTN_WIDE=1: the attention partitions spread over all NX XCDs instead of sitting on the kv head's own
# one. With NKV = 4 and 8 XCDs the head-affine block leaves XCDs 4-7 idle; the price is that prep -> attn ->
# reduce stop being XCD-local, so their events rise to DEVICE scope (p2_affinity derives it).
ATTN_WIDE = os.environ.get("ETX_VL_ATTN_WIDE", "0") == "1"
# ETX_VL_ARGPF=1: wave 1 prefetches the NEXT static task's argument block while thread 0 runs the arrive.
# ETX_VL_BAR2=1: the static loop's broadcast rides the post-drain barrier (two __syncthreads per task, not four).
# Both are kernel-emission switches (etx/codegen/kernel.py), carried on the graph; default off.
ARGPF = os.environ.get("ETX_VL_ARGPF", "0") == "1"
BAR2 = os.environ.get("ETX_VL_BAR2", "0") == "1"
# ETX_DEV_RESET=1: the kernel rewrites its own per-step state at the end of the step, so the host enqueues nothing.
# Read HERE, at plan time, so the emitter and plan_data.h's ETX_EMIT_DEV_RESET agree by construction; the final
# hipcc gets no define for it (build.sh used to, and dropped it whenever ETX_VL_COH was also set -- round 19).
DEV_RESET = os.environ.get("ETX_DEV_RESET", "0") == "1"
DSL_FILE = os.environ.get("ETX_VL_DSL_TILES", str(HERE / "build" / "dsl_tiles.hip"))
DSL_ATTN = os.environ.get("ETX_VL_DSL", "0") == "1"       # =2: attention stays vLLM's (fused q/k norm + rope grid, cache, attn)
if os.environ.get("ETX_VL_QKNR", "0") == "grid" or os.environ.get("ETX_VL_DSL", "0") == "2":
    CHAIN_ROLES = tuple(r for r in CHAIN_ROLES if r != "qknr")
CHAIN_FILE = os.environ.get("ETX_VL_CHAINS", str(HERE / "build" / "chain_tiles.hip"))
# ETX_VL_XCD=1: the one-block launches between the GEMVs (in/post/final norm, silu_and_mul) run once per XCD on
# that XCD's own copies (residual, normed input, activation), and the GEMV that follows is a grid of
# (8 XCDs x 38) tasks each reading its XCD's copy: a device-wide handoff with 304 waiters becomes one with 8
# waiters plus an XCD-local one. The kernels are unchanged; only argument blocks and buffers are per XCD.
XCD = _on("ETX_VL_XCD", universal=True)
NX = 8
if CU % NX:
    raise SystemExit(f"ETX_VL_CU={CU} must be a multiple of {NX} (one worker count per XCD)")
WPX = CU // NX                # workers per XCD of an XCD-affine dense phase (304 -> 38, 192 -> 24)
REP_ROLES = {"in_norm", "post_norm", "act", "final_norm", "topk", "moe_sum"}
# ETX_VL_NOREP=<roles>: roles kept at one task set instead of one replica per XCD. The replica exists so each XCD's
# consumers wait on an XCD-local event, which was worth its publication cost before ETX_VL_COH; with coherent reads a
# consumer no longer invalidates, so 8 coherent copies against 1 copy plus a device-scope read is a measurement, not
# an obvious choice (round 13 priced the top-k replica's body as one write-through round trip, not its arithmetic).
_NOREP = {r for r in os.environ.get("ETX_VL_NOREP", "").split(",") if r}
# host.hip addresses the top-k copy as topk_w + rr * TOPK (its "this XCD's top-k copy"), so de-replicating topk alone
# leaves XCDs 1-7 reading a copy nobody writes: measured 2026-10-01 as 19533 values beyond 2 ulps, correct-looking
# build, wrong answer. Refuse rather than miscompile. The arithmetic says it could not pay anyway: the 8 replicas run
# concurrently, one per XCD, so the phase costs one body either way, while the consumers' wait would rise from
# XCD-local to device scope (1.00 us by the budget's hand-off figure).
if "topk" in _NOREP:
    raise SystemExit("ETX_VL_NOREP=topk needs host.hip to point every consumer at copy 0; see model.py")
REP_ROLES -= _NOREP
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
# ETX_VL_BATCH=1: the dense GEMVs are emitted ONCE for all requests, their tile reading REQUESTS activation columns per
# weight read (etx/ops/gemv.py, DSL_NB). The attention block and the MoE stay replicated, so a batched dense node waits
# on every request's previous event and every request's chain continues from its single output event.
BATCH = os.environ.get("ETX_VL_BATCH", "0") == "1" and REQUESTS > 1
BATCH_ROLES = {"qkv", "o_proj", "router", "lm_head"}
# Whole-layer batching (round 33): topk and the MoE batch too, so the only join left per layer is the batched
# o_proj waiting on every request's reduce (attention is per-sequence and cannot batch). topk: one x-form replica
# set whose wave w does request w's column (etx/ops/topk.py). MoE: the slot index runs over requests x slots,
# S = s * REQUESTS + rq, slot s of every request on XCD s as before; each task serves ONE column, so this removes
# the joins, not the duplicated expert weight stream -- the union dedup is a further step (at B=2 it would save
# only ~3% of expert bytes; the join is the point here). Requires the standalone top-k (no fold: the fold would
# recompute per task per column, round 7's mistake squared), which fold_topk already derives for this model.
BATCH_MOE_ROLES = {"topk", "moe_up", "moe_down"}


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


def fold_topk(d) -> bool:
    """Fold the top-k into the expert tasks? Derived from the shapes unless ETX_VL_FOLD_TOPK forces it.

    On Qwen3-30B-A3B the fold loses by 5.5 us/layer: 448 expert tasks each re-read the same 128 logits with uncached
    accesses, which do not share a cache line, and each redoes a TOPK-deep selection whose every step is a full wave
    reduction. A model with fewer experts or fewer expert tasks can flip it, so folds.py does the comparison.
    """
    if _FOLD_TOPK_ENV is not None:
        return _FOLD_TOPK_ENV == "1"
    return folds.decide(d, CU, NX).get("topk", False)


def instances(d) -> list[dict]:
    """The launches of one decode step, in vLLM's order."""
    parts = math.ceil(MAXLEN / PART)
    FOLD_TOPK = fold_topk(d)
    global DSL_ATTNHEAD
    DSL_ATTNHEAD = _attn_head(d)
    out = []

    def add(export, layer, role, grid, **kw):
        out.append({"export": export, "layer": layer, "role": role, "grid": grid, **kw})

    def gemv(layer, role, m, k):
        if DSL_GEMV:                                          # rows per task = ceil(M / CU): at 304, 17 (qkv), 7 (o_proj), 1 (router)
            fold = DSL_FOLD and (role in ("router", "lm_head") or (role == "qkv" and layer > 0))
            add("dsl_gemv_fold" if fold else "dsl_gemv", layer, role, (CU, 1, 1), M=m, K=k, wvprgrp=0)
            return
        y = wvsplitk_ytile(m)
        add(f"wvsplitk_y{y}", layer, role, (CU, 1, 1), M=m, K=k, wvprgrp=mindiv(m, CU * y, 16))

    attn = {4: "attn_mfma4_g4", 6: "attn_mfma16_g6", 8: "attn_mfma16_g8"}[d.G]
    for L in range(d.L):
        if not (DSL_FOLD and L > 0):
            add("rms_norm_2d" if L == 0 else "fused_add_rms_norm", L, "in_norm", (1, 1, 1))
        gemv(L, "qkv", d.NQKV, d.H)
        if DSL_ATTN:
            # prep + cache + attention in one block per (partition, kv head). NOTE (round 9, 2026-10-01): this shape
            # cannot be built correctly as written, and it is not the way to shorten the attention block.
            #  * dsl_attnprep transforms qkv IN PLACE (reads row, rms-norms + ropes it, writes the same row), so it is
            #    not idempotent: any decomposition that runs it in more than one task per kv head applies norm+rope
            #    twice and is simply wrong. That also rules out Pass 3's prologue recomputation (p3_event_elim, which
            #    gates on grid.pure -- prep is not marked pure, so P3 never fires on it today; marking it pure would
            #    silently miscompile, since P3 checks purity by declaration, not by inspecting in-place writes).
            #  * Making it idempotent (Q to a separate buffer) costs parts x prep per head -- 5 x 6.06 = 30 us/layer
            #    here -- to delete one ~1.2 us hand-off: round 7's folded-top-k mistake at 25x the ratio.
            #  * Letting only partition 0 do the prep does not remove the edge, it relabels it: partitions 1..parts-1
            #    would wait on task (0, h), whose event is published after its WHOLE body, so they wait prep + attn
            #    (6.06 + 1.91) instead of prep (6.06). Strictly worse by attn's body.
            # The serialization is semantic: attn(p, h) needs all G q rows plus the k/v cache rows of head h, and those
            # are a function of this step's qkv. Shortening the block means shortening prep's own dependent chain
            # (8.8 hops measured, ~3-4 necessary), not re-drawing the task boundaries.
            add("dsl_attn", L, "attn", (1, parts, d.NKV))
        elif DSL_ATTNHEAD:
            if not d.qk_norm or d.G + 2 > THREADS // 64:
                raise SystemExit("ETX_VL_DSL_ATTN=head needs q/k norm and G + 2 waves per workgroup (etx/ops/attnprep.py)")
            add("dsl_attnprep", L, "prep", (d.NKV, 1, 1), G=d.G, parts=parts)
        elif d.qk_norm and (QKNR or DSL):
            add("qknorm_rope", L, "qknr", (math.ceil((d.NH + d.NKV) / 8), 1, 1))
        else:
            if d.qk_norm:
                add("rms_norm_3d", L, "q_norm", (d.NH, 1, 1))
                add("rms_norm_3d", L, "k_norm", (d.NKV, 1, 1))
            add("rope_neox", L, "rope", (1, 1, 1))
        if not DSL_ATTN and not DSL_ATTNHEAD:
            add("reshape_and_cache", L, "cache", (1, 1, 1))
        if not DSL_ATTN:
            add(attn, L, "attn", (1, parts, d.NKV), parts=parts)
        add("attn_reduce", L, "reduce", (d.NH, 1, 1), G=d.G)
        gemv(L, "o_proj", d.H, d.NH * d.HD)
        if not DSL_FOLD:
            add("fused_add_rms_norm", L, "post_norm", (1, 1, 1))
        if d.moe:
            gemv(L, "router", d.E, d.H)
            if not DSL_FOLD or not FOLD_TOPK:
                add("dsl_topk" if (DSL or DSL_TOPK) else "topk_softmax_e128", L, "topk", (1, 1, 1))
            if XCD and d.TOPK != NX:                          # the XCD-affine expert path puts slot s on XCD s
                raise SystemExit(f"ETX_VL_XCD needs TOPK == {NX} (top-k slot per XCD), model has {d.TOPK}")
            if DSL_GEMV:                                      # 32 gate + 32 up rows (256 KB) per task; 64 down rows (96 KB)
                add("dsl_moe_up_fold" if DSL_FOLD else "dsl_moe_up", L, "moe_up", (d.TOPK * (d.MI // 32), 1, 1), M=d.MI, K=d.H)
                add("dsl_moe_down_fold" if DSL_FOLD else "dsl_moe_down", L, "moe_down", (d.TOPK * (d.H // 64), 1, 1), M=d.H, K=d.MI)
            else:
                add("fused_moe_up", L, "moe_up", (d.TOPK * math.ceil(2 * d.MI / MOE_BN), 1, 1))
                add("silu_and_mul", L, "moe_act", (d.TOPK, 1, 1))
                add("fused_moe_down", L, "moe_down", (d.TOPK * math.ceil(d.H / MOE_BN), 1, 1))
            if not DSL and not DSL_FOLD:                           # DSL: the next in_norm reads the 8 expert rows itself
                add("moe_sum_k8", L, "moe_sum", (1, 1, 1))
            continue
        gemv(L, "gate_up", 2 * d.INTER, d.H)
        add("silu_and_mul", L, "act", (1, 1, 1))
        gemv(L, "down", d.H, d.INTER)
    if not DSL_FOLD:
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
    if ARGPF:
        g.codegen["argpf"] = True
    if BAR2:
        g.codegen["bar2"] = True
    if DEV_RESET:
        g.codegen["dev_reset"] = True
    # The fold decision is DERIVED (fold_topk), so the host must learn it from the plan, never from an env var it
    # reads independently: build.sh gated the tile's define on ETX_VL_FOLD_TOPK while the plan derived the opposite,
    # and the expert tiles recomputed a top-k the plan had already given its own phase -- 0.63 ms, measured round 30.
    g.codegen["fold_topk"] = fold_topk(d)
    # Compile-time decisions the tiles and the host must share. They are published through the manifest
    # (etx/codegen/manifest.py) and MUST NOT also be passed as -D: build.sh re-deriving the same decision is how
    # ETX_VL_NB (round 14), ETX_DEV_RESET (rounds 19/20) and the top-k fold (round 30) each produced a build that
    # looked healthy and was wrong. COH in particular was derived twice -- here and in build.sh's own default
    # expression -- which agreed only by hand-matching.
    g.codegen["coh"] = COH
    g.codegen["nb"] = REQUESTS if BATCH else 1
    g.codegen["requests"] = REQUESTS
    g.codegen["uc"] = os.environ.get("ETX_VL_UC", "0") == "1"
    g.tensor("impargs", (total,), role="weight", bytes_per_elem=1)
    g.tensor("vl", (1,), role="weight", bytes_per_elem=8)         # placeholder: the native tiles read __constant__ params
    adapters = os.environ.get("ETX_VL_ADAPTERS", str(HERE / "build" / "imp_tiles.hip"))

    def src_of(i):                                             # DSL ops link their generator's tiles, imports the adapter's
        return DSL_FILE if i["export"].startswith("dsl_") else adapters

    def touch_of(i, x):                                        # the touch hook of the dense DSL GEMVs (plain or x form)
        return f"etx_touch_{'x_' if x else ''}{i['export']}" if TOUCH and i["export"] in ("dsl_gemv", "dsl_gemv_fold") else None

    def coh_of(i):                                             # the DSL tiles are the coherent ones (their source is ours)
        return "coherent" if COH and i["export"].startswith("dsl_") else "imported"
    native = str(HERE / "vl_tiles.hip")
    pre = (lambda r: f"r{r}_") if REQUESTS > 1 else (lambda r: "")

    prev = {}
    prev_x = {}                                                # the previous event is per XCD ((NX,) shape)
    for r in range(REQUESTS):
        g.etensor(f"{pre(r)}E_pro", (1,), wait_count=1)
        g.call_device(f"{pre(r)}prologue", (1,), hip_link(native, "vl_prologue"), resource=res, args=["vl"], consts=(r,),
                      out_edges={f"{pre(r)}E_pro": "i->i"}, duration_us=2.0)
        prev[r] = f"{pre(r)}E_pro"; prev_x[r] = False
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
    FT = fold_topk(d)

    def is_batched(grp):
        if not BATCH or len(grp) != 1:
            return False
        i = grp[0]
        if i["role"] in BATCH_ROLES and i["export"].startswith("dsl_gemv"):
            return True
        if FT or not (DSL or DSL_TOPK):                              # the batched MoE path needs the standalone top-k's ids/wts
            return False
        if i["role"] == "topk" and i["export"] == "dsl_topk":
            return True
        return i["role"] in ("moe_up", "moe_down") and i["export"].startswith("dsl_moe")

    for r, n in order:
        groups = groups_r[r]
        grp = groups[n]
        bat = is_batched(grp)
        if bat and r > 0:                                      # one node serves every request
            grp[0]["skip"] = True
            continue
        P = "" if bat else pre(r)
        ev = f"{P}E_{n}"
        if bat:
            grp[0]["nb"] = REQUESTS

        def ins_of(m):                                         # in-edges: every request's previous event when batched
            return {prev[rr]: m for rr in range(REQUESTS)} if bat else {prev[r]: m}

        def advance(e, x):
            if bat:
                for rr in range(REQUESTS):
                    prev[rr] = e; prev_x[rr] = x
            else:
                prev[r] = e; prev_x[r] = x
        if len(grp) == 1 and grp[0]["rep"] > 1 and grp[0]["role"] in REP_ROLES:
            i = grp[0]
            g.etensor(ev, (NX,), wait_count=1)
            name = P + (f"{i['role']}_{i['layer']}_x" if i["layer"] >= 0 else f"{i['role']}_x")
            g.call_device(name, (NX,), hip_link(src_of(i), f"etx_tile_rep_{i['export']}", pre_wait=f"etx_pre_x_{i['export']}", coherence=coh_of(i), arg_bytes=i.get("arg_bytes", 0)),
                          resource=res, args=["impargs"],
                          consts=(i["offset"], i["stride"], i["copyoff"]), in_edges=ins_of("x->(0)"), out_edges={ev: "x->x"},
                          domain_map="x->x", duration_us=3.0)
            prev_x[r] = True
        elif len(grp) == 1 and grp[0]["rep"] > 1:
            # a grid split over the XCDs, (XCDs, tasks per XCD) pinned xw->x, each part waiting on its XCD's part of the
            # previous launch: the GEMV after a replica (38 per XCD) or an XCD-affine MoE launch (one expert slot per XCD)
            i = grp[0]
            if bat and i["role"] in ("moe_up", "moe_down"):    # batched MoE: requests x slots tasks, one node
                i["tasks"] *= REQUESTS                         # (a batched dense GEMV that is also a replica -- layer-0
                i["grid"] = (i["grid"][0] * REQUESTS, i["grid"][1], i["grid"][2])   # qkv -- keeps its task count: NB columns share tasks)
            per = i["tasks"] // NX
            if per * NX != i["tasks"] or (i["role"] not in XAFF_ROLES and per != WPX):
                raise SystemExit(f"{i['export']}: {i['tasks']} tasks do not split over {NX} XCDs as {i['role']} needs")
            nxt = groups[n + 1][0] if n + 1 < len(groups) else None
            local_next = nxt is not None and len(groups[n + 1]) == 1 and nxt["role"] in XAFF_ROLES and nxt["rep"] > 1
            if local_next:        # the next launch's part on this XCD waits for this XCD's part only
                g.etensor(ev, (NX,), wait_count=per)
                out = {ev: "xw->x"}; this_x = True
            else:
                g.etensor(ev, (1,), wait_count=i["tasks"])
                out = {ev: "xw->(0)"}; this_x = False
            name = P + (f"{i['role']}_{i['layer']}" if i["layer"] >= 0 else i["role"])
            dur = max(2.0, i["M"] * i["K"] * 2 / 4.0e12 * 1e6) if i["export"].startswith("wvsplitk") else 3.0
            g.call_device(name, (NX, per), hip_link(src_of(i), f"etx_tile_x_{i['export']}", pre_wait=f"etx_pre_x_{i['export']}", touch=touch_of(i, True), coherence=coh_of(i), arg_bytes=i.get("arg_bytes", 0)),
                          resource=res, args=["impargs"],
                          consts=(i["offset"], i["stride"], per), in_edges=ins_of("xw->x" if prev_x[r] else "xw->(0)"), out_edges=out,
                          domain_map="xw->x", duration_us=dur)
            prev_x[r] = this_x
        elif len(grp) == 1 and DSL_ATTNHEAD and grp[0]["role"] in ("prep", "attn", "reduce"):
            # the attention block by kv head: prep h -> attention (partition p, head h) -> reduce h, all on XCD h
            i = grp[0]
            name = P + f"{i['role']}_{i['layer']}"
            link = hip_link(src_of(i), f"etx_tile_{i['export']}", pre_wait=f"etx_pre_{i['export']}", coherence=coh_of(i), arg_bytes=i.get("arg_bytes", 0))
            if i["role"] == "prep":
                nkv = i["tasks"]
                g.etensor(ev, (nkv,), wait_count=1)
                g.call_device(name, (nkv,), link, resource=res, args=["impargs"], consts=(i["offset"],),
                              in_edges={prev[r]: "i->(0)"}, out_edges={ev: "i->i"}, domain_map="i->i", duration_us=3.0)
            elif i["role"] == "attn":
                parts = i["parts"]; nkv = i["tasks"] // parts
                if nkv * parts != i["tasks"]:
                    raise SystemExit(f"ETX_VL_DSL_ATTN=head needs one attention block per task (ETX_VL_SLOTS={i['export']}=1)")
                g.etensor(ev, (nkv,), wait_count=parts)
                dm = f"i->(i%{NX})" if ATTN_WIDE else f"i->(i/{parts})"   # placement only; the edges stay per kv head
                g.call_device(name, (i["tasks"],), link, resource=res, args=["impargs"], consts=(i["offset"],),
                              in_edges={prev[r]: f"i->(i/{parts})"}, out_edges={ev: f"i->(i/{parts})"}, domain_map=dm, duration_us=8.0)
            else:
                nkv = i["tasks"]
                if i["grid"][0] != nkv * i["G"]:
                    raise SystemExit(f"ETX_VL_DSL_ATTN=head needs the reduce's slots to equal G = {i['G']} (one kv head per task)")
                g.etensor(ev, (1,), wait_count=nkv)
                g.call_device(name, (nkv,), link, resource=res, args=["impargs"], consts=(i["offset"],),
                              in_edges={prev[r]: "i->i"}, out_edges={ev: "i->(0)"}, domain_map="i->i", duration_us=4.0)
        elif len(grp) == 1:
            i = grp[0]
            g.etensor(ev, (1,), wait_count=i["tasks"])
            name = P + (f"{i['role']}_{i['layer']}" if i["layer"] >= 0 else i["role"])
            pin = {"domain_map": "i->(0)"} if i["role"] in PIN_ROLES else {}
            g.call_device(name, (i["tasks"],), hip_link(src_of(i), f"etx_tile_{i['export']}", pre_wait=f"etx_pre_{i['export']}", touch=touch_of(i, False), coherence=coh_of(i), arg_bytes=i.get("arg_bytes", 0)),
                          resource=res, args=["impargs"],
                          consts=(i["offset"],), in_edges=ins_of("i->(0)"), out_edges={ev: "i->(0)"}, **pin,
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
        advance(ev, prev_x[r] if (len(grp) == 1 and grp[0]["rep"] > 1) else False)
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
            f.write(f"# vl plan {d.name}: instance export layer role gx gy gz offset tasks M K wvprgrp rep stride copyoff req nb; total {total} bytes\n")
            f.write(f"total {total} parts {math.ceil(MAXLEN / PART)} maxlen {MAXLEN} requests {REQUESTS}\n")
            for n, i in enumerate(insts):
                if i.get("skip"):                    # ETX_VL_BATCH: request 0's node runs this group for everyone
                    continue
                gx, gy, gz = i["grid"]
                f.write(f"{n} {i['export']} {i['layer']} {i['role']} {gx} {gy} {gz} {i['offset']} {i['tasks']} "
                        f"{i.get('M', 0)} {i.get('K', 0)} {i.get('wvprgrp', 0)} {i['rep']} {i['stride']} {i['copyoff']} {i['req']} {i.get('nb', 1)}\n")
    return g


def bindings(**overrides) -> dict:
    return {}


def runtime(bindings, rng=None):
    return {}
