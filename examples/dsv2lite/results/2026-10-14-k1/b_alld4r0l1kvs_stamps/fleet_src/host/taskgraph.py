#!/usr/bin/env python3
"""Fleet task-graph builder for batch-1 decode (docs/design.md §3, §4).

Builds the immutable, ahead-of-time task queues the persistent kernel runs:
descriptors are generated once on the host, validated as a DAG, assigned to
per-XCD worker queues, and emitted as a flat binary the kernel reads without
synchronisation.

Per MoE layer the graph (v0.11, two global events) is

    QKV[x] ──xcd──▶ AT[x] ──xcd──▶ MG[x] ──xcd──▶ OP[x] ──global──▶ RT[x] (folds the partials)
    RT[x] ──xcd──▶ GU[x] ──xcd──▶ DN[x] ──global──▶ next layer's QKV (folds the partials)

Compared with the v0.9 graph (six global events) the differences are:
  * q/kv_a is head-aligned: XCD k computes the q rows of heads 2k, 2k+1 and
    its own copy of kv_a, so attention waits on an XCD-local event;
  * the merge + W_UV of a head is spread over kv_chunks tasks on the head's
    XCD (one CU cannot stream 128 KB of W_UV fast enough alone);
  * the router runs once per XCD as a Chiplet-task (256 KB of weights,
    replicated 8x, spread over 37 CUs), so the experts wait on an XCD-local
    event and pick the top-k themselves;
  * by default there is no REDUCE task: every consumer folds the 8 partials it needs in
    its prologue (FOLD_PARTIALS on the next q/kv_a and lm_head; the router
    always folds the o_proj partials) and worker (0,0) publishes the result
    to the other x buffer;
  * o_proj is K-split per XCD (its own heads' o slice x 256 columns of W_o),
    so merge → o_proj is XCD-local;
  * gate_up → down can be tiled by K-chunk of h (--k-chunk); off by default,
    measured no gain because down runs on the same workers as gate_up.

Experimental --local-fold and --local-kva insert explicit per-XCD assembly
tasks after baseline worker assignment. Both default off; existing graph bytes
and all q_c publisher / attention worker assignments stay unchanged.

Two granularities are reported. A *logical task* is a node of the graph above.
A *descriptor* is what a worker dequeues: a CU-task is one descriptor, a
Chiplet-task fans out to one descriptor per worker on each XCD it spans (37
per XCD), each carrying its worker id.

    python3 src/host/taskgraph.py --report
    python3 src/host/taskgraph.py --emit build/taskgraph.bin
"""
from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from pathlib import Path

XCDS = 8              # MI300X chiplets
CUS_PER_XCD = 38
BLOCKS_PER_CU = 1     # must match FLEET_BLOCKS_PER_CU in fleet_runtime.h
BLOCKS_PER_XCD = CUS_PER_XCD * BLOCKS_PER_CU
WORKERS_PER_XCD = BLOCKS_PER_XCD - 1   # 1 scheduler + 37 workers
WAVES = 4             # waves per workgroup, must match kWaves in gemv.h
QABS_SLICES = 2       # publishers per head for q_c, must match kQAbsSlices in fleet_types.h
EXPERT_K_CHUNK = 2048  # rows of h per gate_up→down tile, kExpertKChunk in fleet_types.h;
                       # >= moe_inter = one chunk (tiling measured no gain: same workers)
DESCRIPTOR_BYTES = 64
# Must match struct TaskDescriptor in src/runtime/fleet_runtime.h field for
# field; tests/test_descriptor_layout.py checks that it does.
PACK_FORMAT = "<Hhhh hhhh hhhh hhhh i 28x"
PACK_FIELDS = ["kind", "layer", "xcd", "worker", "wait_event", "signal_event",
               "signal_scope", "n_split", "head", "kv_chunk", "expert_slot",
               "wait_scope", "wait_count", "local_event", "flags",
               "signal_xcd_count", "index"]


class TaskKind(IntEnum):
    """Mirrors the task levels in §3. Values are baked into the descriptor."""
    QKV_FUSED = 0      # q rows of this XCD's heads ‖ kv_a, Chiplet-task per XCD
    ATTENTION = 1      # MLA, CU-task, head-affine, kv-post + q-absorb in prologue
    MERGE_UV = 2       # merge the head's partials + 1/kv_chunks of W_UV, CU-task
    O_PROJ = 3         # o_proj + residual, Chiplet-task x8
    NORM_ROUTER = 4    # RMSNorm + router logits, Chiplet-task per XCD
    EXPERT_GATE_UP = 5  # gate_up + SiLU⊙up, Chiplet-task per XCD
    EXPERT_DOWN = 6    # down x routing weight, Chiplet-task per XCD
    REDUCE = 7         # opt-in local assembly; head 0/1 = fp32/bf16 fold, 2 = kv_a
    DENSE_GATE_UP = 8  # layer 0 only
    DENSE_DOWN = 9     # layer 0 only
    EMBED = 10
    LM_HEAD = 11       # Chiplet-task x8, fold + final norm in prologue
    ARGMAX = 12
    PREFETCH = 13      # idle-worker weight prefetch during attention (--prefetch)
    Q_ABSORB = 14      # publish one row slice of a head's absorbed q_c (--qc-published)


class Flags(IntFlag):
    """Descriptor flags (TaskDescriptor.flags)."""
    NONE = 0
    FOLD_PARTIALS = 1   # prologue folds the 8 expert partials into the residual
    SIGNAL_LAST = 2     # signal signal_event only if this task is the last of
                        # n_split to bump local_event (XCD-local counter)
    CHUNK_SIGNAL = 8    # gate_up: every wave bumps local_event + c as soon as
                        # its rows of K-chunk c are stored (kv_chunk chunks)
    CHUNK_WAIT = 16     # down: waits local_event + c (c >= 1) inside the body,
                        # before consuming K-chunk c; chunk 0 is wait_event
    KVA_SHARED = 32     # q/kv_a: kv_a rows are split over all 8 XCDs into one
                        # buffer (not replicated); attention: read that buffer
    WORKER_GROUP = 64   # the task's rows are split over the n_split workers
                        # starting at worker `head` of the XCD, not all 37
    ROUTING_CACHED = 128  # expert down: the same worker just ran this layer's
                          # gate_up, whose top-k is still in its LDS — skip
                          # select_experts (measured 4.9 us per task)
    TOPK_PUBLISH = 256    # router: the last of the XCD's 37 workers to arrive on
                          # local_event publishes the top-k for the experts
    TOPK_READ = 512       # expert: read that published top-k instead of redoing
                          # softmax + top-k per task
    OPROJ_ROW_SPLIT = 2048  # o_proj splits W_o's output rows over the 8 XCDs against
                          # the full o[2048]; the merge boundary becomes global and
                          # the router prologue has no partials to fold
    LOCAL_FOLD = 4096     # explicit per-XCD residual assembly, then local reader staging
    LOCAL_KVA = 8192      # explicit per-XCD gather of the single projected kv_a
    PUB_WAIT = 1024       # the task waits on local_event inside its body, after the
                          # work that does not depend on the publication, and then
                          # reads the published buffer instead of recomputing it


# Implementation status per task kind, reported because the task spec asks for
# "Fleet-native operations, remaining fallbacks" as a deliverable metric.
IMPLEMENTATION: dict[str, str] = {
    "EMBED": "fleet-verified",
    "QKV_FUSED": "fleet-verified",
    "ATTENTION": "fleet-verified",
    "O_PROJ": "fleet-verified",
    "MERGE_UV": "fleet-verified",
    "NORM_ROUTER": "fleet-verified",
    "EXPERT_GATE_UP": "fleet-verified",
    "EXPERT_DOWN": "fleet-verified",
    "DENSE_GATE_UP": "fleet-verified",
    "DENSE_DOWN": "fleet-verified",
    "LM_HEAD": "fleet-verified",
    "ARGMAX": "fleet-verified",
    "PREFETCH": "fleet-experimental",
    "Q_ABSORB": "fleet-experimental",
    "REDUCE": "fleet-unverified",
}


class Scope(IntEnum):
    """Synchronisation scope of an event (§4)."""
    NONE = 0
    XCD_LOCAL = 1      # counter on the producing XCD, no scheduler hop
    GLOBAL = 2         # cross-XCD: counter + mirroring by the per-XCD scheduler


@dataclass
class Task:
    kind: TaskKind
    layer: int
    xcd: int
    wait_event: int | None     # event id this task blocks on
    signal_event: int | None   # event id it contributes to on completion
    signal_scope: Scope = Scope.NONE
    head: int = -1             # attention only
    kv_chunk: int = -1         # split-KV chunk; experts: number of K chunks
    expert_slot: int = -1      # 0..top_k-1 routed, or shared half
    n_split: int = 1           # attention: number of KV chunks of this head
    worker: int = -1           # pinned for Chiplet-tasks, else assign_workers()
    index: int = -1
    wait_scope: Scope = Scope.NONE   # filled by finalize() from the event table
    wait_count: int = 0              # producers of wait_event, per epoch
    local_event: int | None = None   # SIGNAL_LAST: the XCD-local arrival counter
    flags: Flags = Flags.NONE
    signal_xcd_count: int = 0        # producers of signal_event on this XCD (global
                                     # events: the last of them flushes L2 and adds
                                     # the count to the global counter); finalize()
    logical: int = -1                # graph node this descriptor belongs to

    def pack(self) -> bytes:
        """64-byte descriptor; layout is fixed so the kernel can index it."""
        return struct.pack(
            PACK_FORMAT,
            int(self.kind), self.layer, self.xcd, self.worker,
            -1 if self.wait_event is None else self.wait_event,
            -1 if self.signal_event is None else self.signal_event,
            int(self.signal_scope), self.n_split,
            self.head, self.kv_chunk, self.expert_slot,
            int(self.wait_scope), self.wait_count,
            -1 if self.local_event is None else self.local_event,
            int(self.flags), self.signal_xcd_count, self.index,
        )


@dataclass
class Graph:
    tasks: list[Task] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    n_logical: int = 0

    def new_event(self, scope: Scope, producers: int, label: str) -> int:
        eid = len(self.events)
        self.events.append({"id": eid, "scope": int(scope),
                            "producers": producers, "label": label})
        return eid

    def add(self, t: Task) -> Task:
        t.index = len(self.tasks)
        self.tasks.append(t)
        return t

    def cu_task(self, t: Task) -> Task:
        """A single-workgroup task."""
        t.logical = self.n_logical
        self.n_logical += 1
        return self.add(t)

    def chiplet_task(self, proto: Task, xcds, workers=None) -> None:
        """Chiplet-tasks: one logical task per XCD in `xcds`, each a
        descriptor per worker of that XCD (or of `workers`, a sub-range:
        WORKER_GROUP, the rows are then split over that group only)."""
        group = list(range(WORKERS_PER_XCD)) if workers is None else list(workers)
        extra = {}
        if workers is not None:
            extra = {"flags": proto.flags | Flags.WORKER_GROUP,
                     "head": group[0], "n_split": len(group)}
        for xcd in xcds:
            lid = self.n_logical
            self.n_logical += 1
            for w in group:
                t = Task(**{**proto.__dict__, **extra, "xcd": xcd, "worker": w,
                            "logical": lid})
                self.add(t)


def chiplet_producers(n_xcds: int = XCDS) -> int:
    return n_xcds * WORKERS_PER_XCD


def build_qkv(g: Graph, layer: int, prev_event: int | None, fold: bool,
              cfg: dict) -> list[int]:
    """Head-aligned q rows + kv_a, one Chiplet-task per XCD. Default: every
    XCD computes its own copy of all 576 kv_a rows and signals an XCD-local
    event (attention never waits on another chiplet; 0.51 GB/token of
    redundant reads). `kva_shared`: the 576 rows are split over the 8 XCDs
    into one buffer and the event is global (296 producers). `fold`: the
    previous layer was MoE, so the prologue folds its 8 expert partials.
    Returns the per-XCD wait events for attention (8 copies of the global one
    when shared)."""
    flags = Flags.FOLD_PARTIALS if fold else Flags.NONE
    if cfg.get("kva_shared"):
        e = g.new_event(Scope.GLOBAL, chiplet_producers(), f"L{layer}.qkv")
        g.chiplet_task(Task(TaskKind.QKV_FUSED, layer, -1, prev_event, e, Scope.GLOBAL,
                            flags=flags | Flags.KVA_SHARED), range(XCDS))
        return [e] * XCDS
    events = []
    for xcd in range(XCDS):
        e = g.new_event(Scope.XCD_LOCAL, WORKERS_PER_XCD, f"L{layer}.qkv.x{xcd}")
        g.chiplet_task(Task(TaskKind.QKV_FUSED, layer, xcd, prev_event, e,
                            Scope.XCD_LOCAL, flags=flags), [xcd])
        events.append(e)
    return events


def build_attention(g: Graph, layer: int, e_qkv: list[int], cfg: dict) -> list[int]:
    """Attention, then the merge + W_UV spread over kv_chunks tasks per head
    (each takes 128 / kv_chunks rows of W_UV: one CU cannot stream the 128 KB
    fast enough on its own). Returns the per-XCD merge events: XCD k's heads'
    o slice is all its K-split o_proj needs, so merge → o_proj is XCD-local."""
    heads, kv_chunks = cfg["heads"], cfg["kv_chunks"]
    heads_per_xcd = heads // XCDS

    # --oproj-row-split: o_proj takes a slice of W_o's output ROWS, so every
    # XCD needs all 16 heads' o and this boundary has to be one global event.
    # That is the cost side of the A/B; the saving is the fold it removes.
    row_split = cfg.get("oproj_row_split", False)
    if row_split:
        e_merge = [g.new_event(Scope.GLOBAL, heads * kv_chunks,
                               f"L{layer}.merge")] * XCDS
    else:
        e_merge = [g.new_event(Scope.XCD_LOCAL, heads_per_xcd * kv_chunks,
                               f"L{layer}.merge.x{x}") for x in range(XCDS)]
    # q_c publication (--qc-published): every KV chunk of a head used to
    # stream all 131 KB of W_UK to build the same q_c. Publishers run on the
    # workers that would otherwise idle through this phase (32 of 37 take an
    # attention task), and the readers wait for them *after* kv_post, so the
    # wait is paid against work already done.
    published = cfg.get("qc_published", True)
    e_qabs = []
    if published:
        for h in range(heads):
            xcd = h // heads_per_xcd
            e = g.new_event(Scope.XCD_LOCAL, QABS_SLICES, f"L{layer}.qabs.h{h}")
            e_qabs.append(e)
            for sl in range(QABS_SLICES):
                g.cu_task(Task(TaskKind.Q_ABSORB, layer, xcd, e_qkv[xcd], e,
                               Scope.XCD_LOCAL, head=h, kv_chunk=sl, n_split=QABS_SLICES))

    e_attn = []
    for h in range(heads):
        xcd = h // heads_per_xcd
        e = g.new_event(Scope.XCD_LOCAL, kv_chunks, f"L{layer}.attn.h{h}")
        e_attn.append(e)
        att_flags = Flags.KVA_SHARED if cfg.get("kva_shared") else Flags.NONE
        if published:
            att_flags |= Flags.PUB_WAIT
        for chunk in range(kv_chunks):
            g.cu_task(Task(TaskKind.ATTENTION, layer, xcd, e_qkv[xcd], e,
                           Scope.XCD_LOCAL, head=h, kv_chunk=chunk, n_split=kv_chunks,
                           local_event=e_qabs[h] if published else None,
                           flags=att_flags))
    for h in range(heads):
        xcd = h // heads_per_xcd
        for chunk in range(kv_chunks):
            g.cu_task(Task(TaskKind.MERGE_UV, layer, xcd, e_attn[h], e_merge[xcd],
                           Scope.GLOBAL if row_split else Scope.XCD_LOCAL,
                           head=h, kv_chunk=chunk, n_split=kv_chunks))
    return e_merge


# Idle-worker prefetch (§12). Per XCD and layer the attention phase uses 16
# CU-tasks (2 heads x kv_chunks) and the merge another 16, round-robined by
# assign_workers over the 37 workers, so 5 workers sit idle through both
# phases and 16 (the attention workers) through the merge. Emitted right
# after the merge tasks, the prefetch CU-tasks land in that order: the first
# 5 on the idle workers (3 slices each, ~the attention time), the next 16 on
# the attention workers (1 slice each, ~the merge time). They wait on the
# XCD's q/kv_a event and signal nothing.
PREFETCH_IDLE = 5
PREFETCH_IDLE_SLICES = 3
PREFETCH_ATTN = 16


def build_prefetch(g: Graph, layer: int, e_qkv: list[int]) -> None:
    per_xcd = PREFETCH_IDLE * PREFETCH_IDLE_SLICES + PREFETCH_ATTN
    n_split = XCDS * per_xcd
    for xcd in range(XCDS):
        first = xcd * per_xcd
        for i in range(PREFETCH_IDLE):
            g.cu_task(Task(TaskKind.PREFETCH, layer, xcd, e_qkv[xcd], None, Scope.NONE,
                           head=PREFETCH_IDLE_SLICES, kv_chunk=first, n_split=n_split))
            first += PREFETCH_IDLE_SLICES
        for i in range(PREFETCH_ATTN):
            g.cu_task(Task(TaskKind.PREFETCH, layer, xcd, e_qkv[xcd], None, Scope.NONE,
                           head=1, kv_chunk=first, n_split=n_split))
            first += 1
        assert first == (xcd + 1) * per_xcd


def build_o_proj(g: Graph, layer: int, e_merge: list[int],
                 row_split: bool = False) -> int:
    """o_proj, two ways, both reading 1 MB of W_o per XCD.

    K-split (default, §12): XCD k multiplies its own heads' 256 columns of
    W_o against its o slice, so it waits on the XCD-LOCAL merge and the layer
    keeps ONE global event. It pays for that with 8 fp32 partials that every
    router worker folds in its prologue.

    Row-split (--oproj-row-split): XCD k takes 256 of W_o's 2048 output rows
    against the full o[2048] and writes x through EPI_RESIDUAL, so there are
    no partials and the router prologue is a plain stage. It pays with a
    GLOBAL merge boundary (two global events per layer) and a K of 2048
    instead of 256, which drops o_proj out of the half-wave GEMV path.

    Which side wins is a measurement, not an argument: see STATUS.md."""
    e_oproj = g.new_event(Scope.GLOBAL, chiplet_producers(), f"L{layer}.o_proj")
    flags = Flags.OPROJ_ROW_SPLIT if row_split else Flags.NONE
    for xcd in range(XCDS):
        g.chiplet_task(Task(TaskKind.O_PROJ, layer, xcd, e_merge[xcd], e_oproj,
                            Scope.GLOBAL, flags=flags), [xcd])
    return e_oproj


def build_moe_layer(g: Graph, layer: int, prev_event: int | None,
                    cfg: dict, fold: bool) -> int:
    """Emit one MoE layer; returns the event the next layer must wait on."""
    top_k = cfg["top_k"]

    e_qkv = build_qkv(g, layer, prev_event, fold, cfg)
    e_merge = build_attention(g, layer, e_qkv, cfg)
    if cfg.get("prefetch"):
        build_prefetch(g, layer, e_qkv)
    e_oproj = build_o_proj(g, layer, e_merge, cfg.get("oproj_row_split", False))

    # post-attention norm + router, once per XCD as a Chiplet-task: every
    # worker norms x (8 KB, redundant) and computes ~2 router rows into this
    # XCD's logits; the expert tasks pick the top-k in their prologue. 256 KB
    # of router weights read 8 times buys an XCD-local event and 37 CUs of
    # bandwidth instead of one.
    # the last of the XCD's 37 router workers has seen all 64 logits: it turns
    # them into the top-k once and publishes it, then signals e_router, so any
    # expert task released by e_router sees the publication
    published = cfg.get("topk_published", False)
    e_router = []
    for xcd in range(XCDS):
        e = g.new_event(Scope.XCD_LOCAL, WORKERS_PER_XCD, f"L{layer}.norm.x{xcd}")
        pub = g.new_event(Scope.XCD_LOCAL, WORKERS_PER_XCD,
                          f"L{layer}.topk.x{xcd}") if published else None
        nr_flags = Flags.TOPK_PUBLISH if published else Flags.NONE
        if cfg.get("oproj_row_split", False):
            nr_flags |= Flags.OPROJ_ROW_SPLIT    # prologue stages instead of folding
        g.chiplet_task(Task(TaskKind.NORM_ROUTER, layer, xcd, e_oproj, e, Scope.XCD_LOCAL,
                            local_event=pub, n_split=WORKERS_PER_XCD,
                            flags=nr_flags), [xcd])
        e_router.append(e)

    # experts: 8 balanced units = top_k routed + shared split in two.
    # gate_up → down at tile granularity (§12): h[inter] is cut into K chunks
    # of EXPERT_K_CHUNK rows, one XCD-local event each, consecutive ids. Rows
    # are interleaved over the workers, so every wave finishes chunk c at
    # about (c+1)/n of the phase and signals it then; down consumes chunk c
    # while gate_up is still streaming chunk c+1. Producers are counted per
    # wave (37 x 4), which is what lets a wave signal without a block barrier.
    # `split_workers` G > 0: gate_up runs on workers [0, G) and down on
    # [G, 37) of every XCD (WORKER_GROUP), so the K-chunk tiling can actually
    # overlap them — with one group for both, a worker's own gate_up rows
    # are in front of its down task and nothing overlaps (measured).
    read_flag = Flags.TOPK_READ if cfg.get("topk_published", False) else Flags.NONE
    n_chunks = -(-cfg["moe_inter"] // cfg.get("k_chunk", EXPERT_K_CHUNK))
    G = cfg.get("split_workers", 0)
    gu_workers = range(0, G) if G else None
    dn_workers = range(G, WORKERS_PER_XCD) if G else None
    n_gu = G if G else WORKERS_PER_XCD
    n_dn = (WORKERS_PER_XCD - G) if G else WORKERS_PER_XCD
    e_chunk0 = []
    for xcd in range(XCDS):
        ids = [g.new_event(Scope.XCD_LOCAL, n_gu * WAVES,
                           f"L{layer}.gate_up.x{xcd}.k{c}") for c in range(n_chunks)]
        assert ids == list(range(ids[0], ids[0] + n_chunks))
        e_chunk0.append(ids[0])
    # down: the 8 partials are folded by the next layer's q/kv_a workers (or
    # lm_head's) in their prologue, FOLD_PARTIALS — no REDUCE task.
    e_down = g.new_event(Scope.GLOBAL, XCDS * n_dn, f"L{layer}.down")
    for xcd in range(XCDS):
        slot = xcd if xcd < top_k else -(xcd - top_k + 1)  # <0 marks shared half
        g.chiplet_task(Task(TaskKind.EXPERT_GATE_UP, layer, xcd, e_router[xcd],
                            None, Scope.NONE, expert_slot=slot,
                            local_event=e_chunk0[xcd], kv_chunk=n_chunks,
                            flags=Flags.CHUNK_SIGNAL | read_flag), [xcd], workers=gu_workers)
    for xcd in range(XCDS):
        slot = xcd if xcd < top_k else -(xcd - top_k + 1)
        # without a worker split the down descriptor directly follows the
        # same worker's gate_up descriptor, so the routing is still in LDS
        dn_flags = Flags.CHUNK_WAIT | read_flag | (Flags.NONE if G else Flags.ROUTING_CACHED)
        g.chiplet_task(Task(TaskKind.EXPERT_DOWN, layer, xcd, e_chunk0[xcd],
                            e_down, Scope.GLOBAL, expert_slot=slot,
                            local_event=e_chunk0[xcd], kv_chunk=n_chunks,
                            flags=dn_flags), [xcd], workers=dn_workers)
    return e_down


def build_dense_layer(g: Graph, layer: int, prev_event: int | None,
                      cfg: dict, fold: bool) -> int:
    """Layer 0: same attention path, dense MLP instead of experts. The dense
    gate_up → down boundary needs the full h[10944], so it is global; the
    residual add is done in place by dense_down, so nothing to fold after."""
    e_qkv = build_qkv(g, layer, prev_event, fold, cfg)
    e_merge = build_attention(g, layer, e_qkv, cfg)
    if cfg.get("prefetch"):
        build_prefetch(g, layer, e_qkv)
    e_oproj = build_o_proj(g, layer, e_merge, cfg.get("oproj_row_split", False))

    # the dense layer has no experts, so nothing to publish
    e_norm = []
    for xcd in range(XCDS):
        e = g.new_event(Scope.XCD_LOCAL, WORKERS_PER_XCD, f"L{layer}.norm.x{xcd}")
        g.chiplet_task(Task(TaskKind.NORM_ROUTER, layer, xcd, e_oproj, e, Scope.XCD_LOCAL,
                            flags=(Flags.OPROJ_ROW_SPLIT
                                   if cfg.get("oproj_row_split", False)
                                   else Flags.NONE)), [xcd])
        e_norm.append(e)

    e_gate_up = g.new_event(Scope.GLOBAL, chiplet_producers(),
                            f"L{layer}.dense_gate_up")
    for xcd in range(XCDS):
        g.chiplet_task(Task(TaskKind.DENSE_GATE_UP, layer, xcd, e_norm[xcd],
                            e_gate_up, Scope.GLOBAL), [xcd])

    e_out = g.new_event(Scope.GLOBAL, chiplet_producers(), f"L{layer}.dense_down")
    g.chiplet_task(Task(TaskKind.DENSE_DOWN, layer, -1, e_gate_up, e_out,
                        Scope.GLOBAL), range(XCDS))
    return e_out


def build_graph(cfg: dict) -> Graph:
    g = Graph()
    e = g.new_event(Scope.GLOBAL, 1, "embed")
    g.cu_task(Task(TaskKind.EMBED, -1, 0, None, e, Scope.GLOBAL))

    prev_moe = False
    for layer in range(cfg["layers"]):
        if layer < cfg["first_k_dense"]:
            e = build_dense_layer(g, layer, e, cfg, fold=prev_moe)
            prev_moe = False
        else:
            e = build_moe_layer(g, layer, e, cfg, fold=prev_moe)
            prev_moe = True

    e_lm = g.new_event(Scope.GLOBAL, chiplet_producers(), "lm_head")
    g.chiplet_task(Task(TaskKind.LM_HEAD, -1, -1, e, e_lm, Scope.GLOBAL,
                        flags=Flags.FOLD_PARTIALS if prev_moe else Flags.NONE),
                   range(XCDS))
    e_arg = g.new_event(Scope.GLOBAL, 1, "argmax")
    g.cu_task(Task(TaskKind.ARGMAX, -1, 0, e_lm, e_arg, Scope.GLOBAL))
    return g


def finalize(g: Graph) -> None:
    """Copy each waited event's scope and producer count onto the waiter, and
    each global event's per-XCD producer count onto its producers (a
    SIGNAL_LAST group counts once: only its last member signals)."""
    for t in g.tasks:
        if t.wait_event is None:
            continue
        ev = g.events[t.wait_event]
        t.wait_scope = Scope(ev["scope"])
        t.wait_count = ev["producers"]
    per_xcd: dict[tuple[int, int], int] = {}
    seen_groups: set[tuple[int, int]] = set()
    for t in g.tasks:
        if t.signal_event is None:
            continue
        if t.flags & Flags.SIGNAL_LAST:
            key = (t.signal_event, t.local_event)
            if key in seen_groups:
                continue
            seen_groups.add(key)
        per_xcd[(t.signal_event, t.xcd)] = per_xcd.get((t.signal_event, t.xcd), 0) + 1
    for t in g.tasks:
        if t.signal_event is not None:
            t.signal_xcd_count = per_xcd[(t.signal_event, t.xcd)]


def validate(g: Graph) -> list[str]:
    """DAG checks that would otherwise show up as a deadlocked kernel."""
    errors: list[str] = []
    signalled: dict[int, int] = {}
    last_groups: dict[tuple[int, int], int] = {}   # (signal, local) -> members
    for t in g.tasks:
        if t.flags & Flags.CHUNK_SIGNAL:
            if t.local_event is None or t.kv_chunk < 1:
                errors.append(f"task {t.index}: CHUNK_SIGNAL without chunk events")
                continue
            for c in range(t.kv_chunk):
                signalled[t.local_event + c] = signalled.get(t.local_event + c, 0) + WAVES
        if t.flags & Flags.CHUNK_WAIT:
            if t.local_event is None or t.wait_event != t.local_event or t.kv_chunk < 1:
                errors.append(f"task {t.index}: CHUNK_WAIT must wait on its chunk 0 event")
                continue
            e0 = g.events[t.local_event]
            for c in range(1, t.kv_chunk):
                ev = g.events[t.local_event + c]
                if ev["scope"] != e0["scope"] or ev["producers"] != e0["producers"]:
                    errors.append(f"task {t.index}: chunk event {ev['id']} differs "
                                  f"from chunk 0 ({e0['id']})")
        if t.signal_event is None:
            continue
        if t.flags & Flags.SIGNAL_LAST:
            if t.local_event is None:
                errors.append(f"task {t.index}: SIGNAL_LAST without local_event")
                continue
            key = (t.signal_event, t.local_event)
            last_groups[key] = last_groups.get(key, 0) + 1
            signalled[t.local_event] = signalled.get(t.local_event, 0) + 1
        else:
            signalled[t.signal_event] = signalled.get(t.signal_event, 0) + 1
    for (sig, loc), members in last_groups.items():
        # the group counts as ONE producer of the global event, and its size
        # must equal the local counter's producer count and each member's n_split
        signalled[sig] = signalled.get(sig, 0) + 1
        want = g.events[loc]["producers"]
        if members != want:
            errors.append(f"event {loc} ({g.events[loc]['label']}): {members} tasks "
                          f"bump it, table says {want}")
        if g.events[loc]["scope"] != int(Scope.XCD_LOCAL):
            errors.append(f"event {loc}: last-arrival counter must be XCD-local")
    for t in g.tasks:
        if t.flags & Flags.SIGNAL_LAST and t.n_split != g.events[t.local_event]["producers"]:
            errors.append(f"task {t.index}: n_split {t.n_split} != producers of "
                          f"its local event")

    for t in g.tasks:
        if t.flags & Flags.PUB_WAIT:
            if t.local_event is None:
                errors.append(f"task {t.index}: PUB_WAIT without a local_event")
            else:
                ev = g.events[t.local_event]
                if ev["producers"] != QABS_SLICES or ev["scope"] != int(Scope.XCD_LOCAL):
                    errors.append(f"task {t.index}: PUB_WAIT on event {ev['id']} with "
                                  f"{ev['producers']} producers, scope {ev['scope']}; "
                                  f"expected {QABS_SLICES} XCD-local")
    # A count-one local event is a payload publication, not permission to skip
    # the original global acquire. Validate both sides and the buffer selector.
    publishers = {t.signal_event: t for t in g.tasks if t.kind == TaskKind.REDUCE}
    source_kinds = {}
    attention_workers = {}
    for t in g.tasks:
        if t.kind in (TaskKind.Q_ABSORB, TaskKind.ATTENTION):
            attention_workers.setdefault((t.layer, t.xcd), set()).add(t.worker)
        if t.signal_event is not None:
            source_kinds.setdefault(t.signal_event, set()).add(t.kind)
    for t in g.tasks:
        local_flags = t.flags & (Flags.LOCAL_FOLD | Flags.LOCAL_KVA)
        if t.kind == TaskKind.REDUCE:
            want_flag = Flags.LOCAL_KVA if t.head == 2 else Flags.LOCAL_FOLD
            want_source = {0: TaskKind.O_PROJ, 1: TaskKind.EXPERT_DOWN,
                           2: TaskKind.QKV_FUSED}.get(t.head)
            if (want_source is None or local_flags != want_flag
                    or t.wait_event is None or t.signal_event is None):
                errors.append(f"task {t.index}: invalid local assembly descriptor")
                continue
            incoming, outgoing = g.events[t.wait_event], g.events[t.signal_event]
            if (incoming["scope"] != int(Scope.GLOBAL)
                    or source_kinds.get(t.wait_event) != {want_source}):
                errors.append(f"task {t.index}: local assembly needs its original GLOBAL payload")
            if (outgoing["scope"] != int(Scope.XCD_LOCAL) or outgoing["producers"] != 1
                    or t.signal_scope != Scope.XCD_LOCAL):
                errors.append(f"task {t.index}: local assembly must publish one XCD-local arrival")
            if t.head == 2 and t.worker in attention_workers.get((t.layer, t.xcd), set()):
                errors.append(f"task {t.index}: local kv_a gather steals an attention/q_c worker")
            continue
        if not local_flags:
            continue
        variant = (0 if t.kind == TaskKind.NORM_ROUTER else
                   1 if t.kind in (TaskKind.QKV_FUSED, TaskKind.LM_HEAD) else
                   2 if t.kind == TaskKind.ATTENTION else -1)
        flag = Flags.LOCAL_KVA if variant == 2 else Flags.LOCAL_FOLD
        pub = publishers.get(t.wait_event)
        if (variant < 0 or local_flags != flag or pub is None
                or pub.head != variant or pub.layer != t.layer or pub.xcd != t.xcd
                or t.wait_scope != Scope.XCD_LOCAL or t.wait_count != 1):
            errors.append(f"task {t.index}: local reader lacks its matching XCD publication")
        if variant == 1 and not t.flags & Flags.FOLD_PARTIALS:
            errors.append(f"task {t.index}: local expert-fold reader lost FOLD_PARTIALS")
        if variant == 2 and (not t.flags & Flags.KVA_SHARED or not t.flags & Flags.PUB_WAIT):
            errors.append(f"task {t.index}: local kv_a reader lost shared projection/q_c publication")
    # Row-split o_proj multiplies against the WHOLE o[2048], so the merge that
    # releases it must be GLOBAL. Left XCD-local the graph still looks locally
    # consistent -- each XCD waits on a counter only it signals -- while every
    # o_proj reads 7/8 of its input before the other XCDs have written it.
    for t in g.tasks:
        if (t.flags & Flags.OPROJ_ROW_SPLIT) and t.kind == TaskKind.O_PROJ:
            if t.wait_event is None:
                errors.append(f"task {t.index}: row-split o_proj has no wait event")
            elif g.events[t.wait_event]["scope"] != int(Scope.GLOBAL):
                errors.append(f"task {t.index}: row-split o_proj waits on event "
                              f"{t.wait_event} (scope "
                              f"{g.events[t.wait_event]['scope']}); needs GLOBAL, "
                              f"or it reads o before the other XCDs wrote it")
    for t in g.tasks:
        if t.flags & Flags.TOPK_PUBLISH:
            if t.local_event is None or t.n_split != WORKERS_PER_XCD:
                errors.append(f"task {t.index}: TOPK_PUBLISH without a full-XCD arrival counter")
            else:
                signalled[t.local_event] = signalled.get(t.local_event, 0) + 1
    for ev in g.events:
        got, want = signalled.get(ev["id"], 0), ev["producers"]
        if got != want:
            errors.append(f"event {ev['id']} ({ev['label']}): {got} producers "
                          f"signal it, event table says {want} — kernel would hang")
    # Per-XCD shares of each global event must add up to its producer count:
    # the kernel adds a share to the global counter once per XCD.
    # A set per (event, xcd): every descriptor of a chiplet task carries the
    # same share, and one that disagrees used to be hidden by the 36 after it.
    shares: dict[int, dict[int, set]] = {}
    for t in g.tasks:
        if t.signal_event is not None and t.signal_scope == Scope.GLOBAL:
            shares.setdefault(t.signal_event, {}).setdefault(t.xcd, set()).add(
                t.signal_xcd_count)
    for e, by_xcd in shares.items():
        label = g.events[e]["label"]
        for xcd, vals in sorted(by_xcd.items()):
            if len(vals) > 1:
                errors.append(f"event {e} ({label}): XCD {xcd} descriptors disagree "
                              f"on their per-XCD share {sorted(vals)}")
        got = {x: max(v) for x, v in by_xcd.items()}
        want = g.events[e]["producers"]
        if sum(got.values()) != want:
            errors.append(f"event {e} ({label}): per-XCD shares "
                          f"{got} do not sum to {want}")

    prev_in_queue: dict[tuple[int, int], Task] = {}
    for t in g.tasks:
        if t.flags & Flags.ROUTING_CACHED:
            q = prev_in_queue.get((t.xcd, t.worker))
            if q is None or q.kind != TaskKind.EXPERT_GATE_UP or q.layer != t.layer:
                errors.append(f"task {t.index}: ROUTING_CACHED but the worker's previous "
                              f"descriptor is not this layer's gate_up")
        prev_in_queue[(t.xcd, t.worker)] = t
    for t in g.tasks:
        if t.wait_event is not None:
            ev = g.events[t.wait_event]
            if t.wait_count != ev["producers"] or int(t.wait_scope) != ev["scope"]:
                errors.append(f"task {t.index}: wait-side fields "
                              f"({t.wait_scope.name}, {t.wait_count}) disagree "
                              f"with event {ev['id']} — finalize() not run?")
        if t.worker < 0 or t.worker >= WORKERS_PER_XCD or t.xcd < 0 or t.xcd >= XCDS:
            errors.append(f"task {t.index} has no valid (xcd, worker): "
                          f"({t.xcd}, {t.worker})")

    # A task may only wait on an event every one of whose producers appears
    # earlier in the queue; otherwise AOT ordering can deadlock.
    last_producer: dict[int, int] = {}
    for t in g.tasks:
        if t.signal_event is not None:
            last_producer[t.signal_event] = max(
                last_producer.get(t.signal_event, -1), t.index)
    for t in g.tasks:
        if t.flags & Flags.CHUNK_SIGNAL:
            for c in range(t.kv_chunk):
                e = t.local_event + c
                last_producer[e] = max(last_producer.get(e, -1), t.index)
    for t in g.tasks:
        waits = [] if t.wait_event is None else [t.wait_event]
        if t.flags & Flags.CHUNK_WAIT:
            waits += [t.local_event + c for c in range(1, t.kv_chunk)]
        if t.flags & Flags.PUB_WAIT:
            waits.append(t.local_event)
        for e in waits:
            if last_producer.get(e, 1 << 30) > t.index:
                errors.append(f"task {t.index} ({t.kind.name}) waits on event "
                              f"{e} produced later — cycle")

    # XCD-local events must not cross chiplets.
    xcds_of_event: dict[int, set[int]] = {}
    for t in g.tasks:
        for e in (t.signal_event, t.local_event):
            if e is not None:
                xcds_of_event.setdefault(e, set()).add(t.xcd)
        if t.flags & Flags.CHUNK_SIGNAL:
            for c in range(1, t.kv_chunk):
                xcds_of_event.setdefault(t.local_event + c, set()).add(t.xcd)
    for t in g.tasks:
        if t.wait_event is None:
            continue
        ev = g.events[t.wait_event]
        if ev["scope"] == int(Scope.XCD_LOCAL):
            producers = xcds_of_event.get(t.wait_event, set())
            if producers - {t.xcd}:
                errors.append(f"task {t.index} on XCD {t.xcd} waits on XCD-local "
                              f"event {t.wait_event} signalled from {sorted(producers)}")
    for e, xs in xcds_of_event.items():
        if g.events[e]["scope"] == int(Scope.XCD_LOCAL) and len(xs) > 1:
            errors.append(f"XCD-local event {e} ({g.events[e]['label']}) is produced "
                          f"from XCDs {sorted(xs)}")
    return errors


def assign_workers(g: Graph) -> None:
    """Round-robin the CU-tasks within each XCD; Chiplet-task descriptors
    already carry their worker."""
    cursor = [0] * XCDS
    for t in g.tasks:
        if t.worker >= 0:
            continue
        t.worker = cursor[t.xcd] % WORKERS_PER_XCD
        cursor[t.xcd] += 1


def add_local_assembly(g: Graph, cfg: dict) -> None:
    """Insert optional publishers AFTER baseline CU assignment.

    Pinned assembly tasks must not advance the round-robin cursor: that would
    displace q_c publishers onto attention workers and erase v0.18's overlap.
    Existing event IDs and descriptor order are preserved. New count-one local
    events publish distinct per-XCD buffers after the original global acquire.
    Their storage can be reused only after all previous readers finish: global
    qkv/o_proj/down boundaries already enforce that lifetime across layers.
    """
    local_fold, local_kva = cfg.get("local_fold", False), cfg.get("local_kva", False)
    if not (local_fold or local_kva):
        return
    if local_fold and (cfg.get("oproj_row_split", False)
                       or not cfg.get("kva_shared", False)):
        raise ValueError("--local-fold requires shared kv_a and K-split o_proj")
    if local_kva and (cfg["heads"] != 16 or cfg["kv_chunks"] != 16
                      or not cfg.get("qc_published", True)
                      or not cfg.get("kva_shared", False)):
        raise ValueError("--local-kva requires 16 heads, --kv-chunks 16, "
                         "q_c publication and shared kv_a")

    # There is exactly one free worker per XCD: 37 - 32 attention - 4 q_c.
    # Include the initial embed and all preceding CU tasks by inspecting the
    # completed baseline assignment, rather than guessing its rotating cursor.
    idle = {}
    if local_kva:
        busy = {}
        for t in g.tasks:
            if t.kind in (TaskKind.Q_ABSORB, TaskKind.ATTENTION):
                busy.setdefault((t.layer, t.xcd), set()).add(t.worker)
        for key, workers in busy.items():
            free = set(range(WORKERS_PER_XCD)) - workers
            if len(free) != 1:
                raise ValueError("--local-kva needs exactly one idle attention worker "
                                 f"per XCD; layer/XCD {key} has {len(free)}")
            idle[key] = free.pop()

    originals, g.tasks = g.tasks, []
    publications = {}
    for t in originals:
        variant = None
        if local_fold:
            if t.kind == TaskKind.NORM_ROUTER:
                variant = 0
            elif t.kind in (TaskKind.QKV_FUSED, TaskKind.LM_HEAD) \
                    and t.flags & Flags.FOLD_PARTIALS:
                variant = 1
        if local_kva and t.kind == TaskKind.ATTENTION:
            variant = 2
        if variant is not None:
            key = (t.layer, variant)
            flag = Flags.LOCAL_KVA if variant == 2 else Flags.LOCAL_FOLD
            if key not in publications:
                assert t.wait_event is not None
                assert g.events[t.wait_event]["scope"] == Scope.GLOBAL
                events = []
                for xcd in range(XCDS):
                    label = (f"L{t.layer}.local_kva.x{xcd}" if variant == 2 else
                             f"L{t.layer}.local_fold{variant}.x{xcd}")
                    event = g.new_event(Scope.XCD_LOCAL, 1, label)
                    worker = idle[(t.layer, xcd)] if variant == 2 else 0
                    g.cu_task(Task(TaskKind.REDUCE, t.layer, xcd, t.wait_event,
                                   event, Scope.XCD_LOCAL, head=variant,
                                   worker=worker, flags=flag))
                    events.append(event)
                publications[key] = events
            t.wait_event = publications[key][t.xcd]
            t.flags |= flag
        g.add(t)


def build(cfg: dict) -> Graph:
    g = build_graph(cfg)
    assign_workers(g)
    add_local_assembly(g, cfg)
    finalize(g)
    return g


def emit(g: Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(t.pack() for t in g.tasks))
    side = path.with_suffix(path.suffix + ".events.txt")
    side.write_text("".join(f"{e['id']} {e['scope']} {e['producers']} {e['label']}\n"
                            for e in g.events))
    print(f"wrote {len(g.tasks)} descriptors to {path} "
          f"({len(g.tasks) * DESCRIPTOR_BYTES / 1024:.0f} KB) and {side.name}")


def report(g: Graph, cfg: dict) -> None:
    from collections import Counter
    logical_kinds = Counter()
    seen = set()
    for t in g.tasks:
        if t.logical not in seen:
            seen.add(t.logical)
            logical_kinds[t.kind.name] += 1
    desc_kinds = Counter(t.kind.name for t in g.tasks)
    globals_ = sum(1 for e in g.events if e["scope"] == int(Scope.GLOBAL))
    locals_ = sum(1 for e in g.events if e["scope"] == int(Scope.XCD_LOCAL))

    moe = [t for t in g.tasks if t.layer == cfg["first_k_dense"]]
    moe_logical = len({t.logical for t in moe})
    moe_events = {t.signal_event for t in moe if t.signal_event is not None}
    moe_events |= {t.local_event for t in moe if t.local_event is not None}
    moe_events |= {t.local_event + c for t in moe if t.flags & Flags.CHUNK_SIGNAL
                   for c in range(t.kv_chunk)}
    moe_global = sum(1 for e in moe_events if g.events[e]["scope"] == int(Scope.GLOBAL))
    moe_local = len(moe_events) - moe_global

    print(f"logical tasks: {g.n_logical}   descriptors: {len(g.tasks)}   "
          f"events: {len(g.events)} ({globals_} global, {locals_} XCD-local)")
    print(f"\nper MoE layer: {moe_logical} logical tasks, {len(moe)} descriptors, "
          f"{moe_global} global + {moe_local} XCD-local events produced "
          f"(kv_chunks={cfg['kv_chunks']})")
    print("\ntask kinds (logical / descriptors):")
    for k, n in sorted(logical_kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {n:6d} / {desc_kinds[k]:6d}")

    per_xcd = [sum(1 for t in g.tasks if t.xcd == x) for x in range(XCDS)]
    print(f"\nper-XCD descriptor counts: {per_xcd}")
    print(f"  imbalance: {max(per_xcd) - min(per_xcd)} descriptors "
          f"({(max(per_xcd)/min(per_xcd) - 1)*100:.1f}%)")
    per_worker = Counter((t.xcd, t.worker) for t in g.tasks)
    print(f"  per-worker queue length: min {min(per_worker.values())}, "
          f"max {max(per_worker.values())}")
    print(f"\ndescriptor memory: {len(g.tasks) * DESCRIPTOR_BYTES / 1024:.1f} KB "
          f"({len(g.tasks)} x {DESCRIPTOR_BYTES} B)")

    by_status: dict[str, int] = {}
    for kind, n in logical_kinds.items():
        st = IMPLEMENTATION.get(kind, "fallback")
        by_status[st] = by_status.get(st, 0) + n
    total = sum(by_status.values())
    print("\nFleet-native vs fallback (by logical task count):")
    for status in ("fleet-verified", "fleet-unverified", "fallback"):
        n = by_status.get(status, 0)
        print(f"  {status:<18} {n:6d}  {100*n/total:5.1f}%")
    fb = sorted(k for k in logical_kinds if IMPLEMENTATION.get(k, "fallback") == "fallback")
    if fb:
        print(f"  still on torch: {', '.join(fb)}")

    errors = validate(g)
    print(f"\nvalidation: {'PASS' if not errors else f'{len(errors)} ERRORS'}")
    for e in errors[:10]:
        print(f"  ! {e}")


def load_cfg(config: Path, kv_chunks: int, prefetch: bool = False,
             k_chunk: int = EXPERT_K_CHUNK, kva_shared: bool = True,
             split_workers: int = 0, topk_published: bool = False,
             qc_published: bool = True, oproj_row_split: bool = False,
             local_fold: bool = False, local_kva: bool = False) -> dict:
    c = json.loads(config.read_text())
    return {
        "prefetch": prefetch,
        "k_chunk": k_chunk,
        "kva_shared": kva_shared,
        "split_workers": split_workers,
        "topk_published": topk_published,
        "qc_published": qc_published,
        "oproj_row_split": oproj_row_split,
        "local_fold": local_fold,
        "local_kva": local_kva,
        "layers": c["num_hidden_layers"],
        "heads": c["num_attention_heads"],
        "top_k": c["num_experts_per_tok"],
        "first_k_dense": c["first_k_dense_replace"],
        "moe_inter": c["moe_intermediate_size"],
        "kv_chunks": kv_chunks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[2] / "reference" / "dsv2lite_config.json")
    ap.add_argument("--kv-chunks", type=int, default=8,
                    help="split-KV factor: attention tasks per head (8 measured best)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--emit", type=Path, help="write packed descriptors here")
    ap.add_argument("--prefetch", action="store_true",
                    help="idle-worker weight prefetch tasks during attention (§12)")
    ap.add_argument("--k-chunk", type=int, default=EXPERT_K_CHUNK,
                    help="rows of h per gate_up->down tile; must match kExpertKChunk "
                         "unless it is >= moe_inter (one chunk = no tiling)")
    ap.add_argument("--kva-replicated", action="store_true",
                    help="every XCD computes its own copy of the 576 kv_a rows behind an "
                         "XCD-local event (the v0.10-v0.14 graph) instead of the default "
                         "split over the 8 XCDs behind one global event")
    ap.add_argument("--qc-per-task", action="store_true",
                    help="every KV chunk streams all 131 KB of W_UK to build its own q_c "
                         "(the pre-v0.18 graph); the default publishes it once per head, "
                         "measured 3.593 vs 3.650 ms over three alternating pairs")
    ap.add_argument("--topk-published", action="store_true",
                    help="the router's last arriver publishes the top-k and the expert tasks "
                         "read it (64 B); measured 0.4%% slower than the default, where every "
                         "expert task redoes softmax + top-k in parallel (STATUS.md)")
    ap.add_argument("--oproj-row-split", action="store_true",
                    help="o_proj splits W_o's 2048 output rows over the 8 XCDs against the "
                         "full o[2048] and writes x directly, removing the 8 fp32 partials "
                         "the router prologue folds; costs a GLOBAL merge boundary (two "
                         "global events per layer) and K 256 -> 2048")
    ap.add_argument("--local-fold", action="store_true",
                    help="experimental: assemble residual folds once per XCD and fan "
                         "out locally (default off; shared kv_a and K-split o_proj only)")
    ap.add_argument("--local-kva", action="store_true",
                    help="experimental: gather the single projected kv_a into each "
                         "XCD's own buffer (default off; 16 KV chunks and q_c publication)")
    ap.add_argument("--split-workers", type=int, default=0,
                    help="G: expert gate_up on workers [0,G) and down on [G,37) per XCD, "
                         "so --k-chunk tiling can overlap them")
    a = ap.parse_args()

    cfg = load_cfg(a.config, a.kv_chunks, a.prefetch, a.k_chunk, not a.kva_replicated,
                   a.split_workers, a.topk_published, not a.qc_per_task,
                   a.oproj_row_split, a.local_fold, a.local_kva)
    try:
        g = build(cfg)
    except ValueError as exc:
        ap.error(str(exc))

    if a.emit:
        emit(g, a.emit)
    if a.report or not a.emit:
        report(g, cfg)
    raise SystemExit(1 if validate(g) else 0)


if __name__ == "__main__":
    main()
