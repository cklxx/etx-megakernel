"""Persistent-kernel emitter (L5).

Emits one translation unit per kernel instance:
  * a dispatch switch over task types calling the frontend-provided tile bodies
  * per-type wait code generated from the in-edge maps (targets enumerated in C)
  * per-type arrive code from the out-edge maps, with pushes for dynamic events
  * the worker loop: static head if ready -> pop local/global -> spin on head
HIP and CUDA share the template; every hardware-specific line comes from
etx_lowering.h.  The plan (queues, descriptors, event layout, consumer lists)
is emitted separately as JSON for the host launcher.
"""
from __future__ import annotations

from . import manifest

import json
import os
from typing import Any

from ..ir.edgemap import EdgeMap
from ..ir.types import Scope, TaskGrid
from ..passes.plan import Plan


def _scope_macro(plan: Plan, event: str) -> str:
    return plan.events[event].scope.name


def _runtime_ptrs(plan: Plan, g: TaskGrid) -> dict[str, str]:
    ptrs = {}
    for m in list(g.in_edges.values()) + list(g.out_edges.values()):
        for rt in m.runtime:
            ptrs[rt] = f"((const int32_t*)p.args[{_arg_index(plan, rt)}])"
    return ptrs


def _arg_table(plan: Plan) -> dict[str, int]:
    """Tensor name -> argument index, in graph order. Cached on the plan object itself.

    The cache was a module dict keyed by id(plan). CPython reuses the id of a freed object, so a new plan could get
    the table of a dead plan with other tensors (tests/test_selection_manifest.py, then test_feature_manifest.py:
    KeyError 'impargs'). The cache now lives and dies with its plan, and it is rebuilt if the tensors change."""
    names = tuple(plan.graph.tensors)
    cached = plan.__dict__.get("_etx_arg_table")
    if cached is None or cached[0] != names:
        cached = (names, {n: i for i, n in enumerate(names)})
        plan.__dict__["_etx_arg_table"] = cached
    return cached[1]


def _arg_index(plan: Plan, tensor: str) -> int:
    return _arg_table(plan)[tensor]


def _gather_width(plan: Plan, tensor: str) -> str:
    t = plan.graph.tensors[tensor]
    if len(t.shape) >= 2:
        return f"p.shape[{plan.graph.symbols.index(t.shape[1])}]" if isinstance(t.shape[1], str) and t.shape[1] in plan.graph.symbols else str(t.shape[1])
    return "1"


def _symbols(plan: Plan) -> dict[str, str]:
    return {s: f"p.shape[{i}]" for i, s in enumerate(plan.graph.symbols)}


def _runtime_count_c(plan: Plan, expr: str) -> str:
    """'expert_counts[i]' -> '((const int32_t*)p.args[k])[i]' (rank-1 events; loop variable i)."""
    import re as _re
    def repl(m):
        return f"((const int32_t*)p.args[{_arg_index(plan, m.group(1))}])["
    return _re.sub(r"\b(\w+)\[", repl, expr)


def _relayed(plan: Plan, ep) -> bool:
    return plan.relay and ep.scope == Scope.DEVICE


def _wait_code(plan: Plan, g: TaskGrid) -> str:
    lines = []
    if any(_relayed(plan, plan.events[ev]) for ev in g.in_edges):
        lines.append("      bool etx_lok = true;   // every relayed wait was covered by its domain relay's acquire")
    coord_vars = [f"t.coord[{i}]" for i in range(len(g.grid))]
    for ev, m in g.in_edges.items():
        ep = plan.events[ev]
        sc = ep.scope.name
        width = _gather_width(plan, m.gather_tensor) if m.kind == "gather" else ""
        lines.append(f"      // wait {ev} via {m.text!r} ({sc}{', relay mirror' if _relayed(plan, ep) else ''})")
        if _relayed(plan, ep):
            lines.append(f"#define ETX_TARGET(idx) etx_lok &= etx_wait_DEVICE_mirror(p, {ep.offset} + (idx), domain)")
        else:
            lines.append(f"#define ETX_TARGET(idx) etx_wait_{sc}(p.events + ETX_EVW({ep.offset} + (idx)), p.ctrl_abort)")
        lines.append(m.to_c(coord_vars, f"p.ev_shape[{list(plan.events).index(ev)}]", "ETX_TARGET",
                            _runtime_ptrs(plan, g), width, _symbols(plan)).rstrip())
        lines.append("#undef ETX_TARGET")
    return "\n".join(lines)


def _arrive_code(plan: Plan, g: TaskGrid, mode: str) -> str:
    lines = []
    coord_vars = [f"t.coord[{i}]" for i in range(len(g.grid))]
    for ev, m in g.out_edges.items():
        ep = plan.events[ev]
        sc = ep.scope.name
        ev_id = list(plan.events).index(ev)
        width = _gather_width(plan, m.gather_tensor) if m.kind == "gather" else ""
        lines.append(f"      // arrive {ev} via {m.text!r} ({sc})")
        has_dynamic_consumers = any(k[0] == ev for k in plan.ev_consumers)
        # DEVICE scope: last-arriver flush per domain (one L2 write-back per domain per coordinate)
        shares = {v for row in ep.share.values() for v in row if v > 0}
        nf = "_nf" if ep.producers_coherent else ""        # all producers store coherently: the payload is already past L2
        if ep.scope == Scope.DEVICE and not ep.runtime_init and len(shares) == 1:
            arrive = f"etx_arrive_flush_DEVICE_s{nf}(p, {ep.offset} + (idx), domain, {next(iter(shares))})"   # share as an immediate
        elif ep.scope == Scope.DEVICE:
            arrive = f"etx_arrive_flush_DEVICE{nf}(p, {ep.offset} + (idx), domain)"
        else:
            arrive = f"etx_arrive_{sc}(p.events + ETX_EVW({ep.offset} + (idx)))"
        if not has_dynamic_consumers:       # the producer's own mode is irrelevant: a static producer must still push dynamic consumers
            lines.append(f"#define ETX_TARGET(idx) (void){arrive}")
        else:
            lines.append(f"#define ETX_TARGET(idx) do {{ if ({arrive} == 0) etx_push_consumers(p, {ev_id}, (idx), domain); }} while (0)")
        lines.append(m.to_c(coord_vars, f"p.ev_shape[{ev_id}]", "ETX_TARGET", _runtime_ptrs(plan, g), width, _symbols(plan)).rstrip())
        lines.append("#undef ETX_TARGET")
    return "\n".join(lines)


def _setup_consts(g: TaskGrid) -> list[str]:
    if not g.consts:
        return []
    cs = list(g.consts)[:4] + [0] * (4 - min(4, len(g.consts)))
    return [f"      ctx.cst[0] = {cs[0]}; ctx.cst[1] = {cs[1]}; ctx.cst[2] = {cs[2]}; ctx.cst[3] = {cs[3]};"]


def _setup_body(plan: Plan, g: TaskGrid, symbols: list[str]) -> list[str]:
    """Pass-3 prologue contexts and the body symbol of one task type (shared by the fused and unfused kernels)."""
    out = []
    coord_vars = [f"t.coord[{i}]" for i in range(len(g.grid))]
    for j, (pro, mtext) in enumerate(g.prologue):
        ig = plan.inlined[pro]
        exprs = EdgeMap.parse(mtext).coord_exprs_c(coord_vars, _symbols(plan))
        out.append(f"      // prologue {j}: recompute {pro} for this task (Pass 3)")
        out.append(f"      pctx[{j}] = ctx; pctx[{j}].args = p.type_args + {plan.type_ids[pro]} * p.max_args; psym[{j}] = {symbols.index(ig.body.symbol)};   // {ig.body.symbol}")
        pc = list(ig.consts)[:4] + [0] * (4 - min(4, len(ig.consts)))
        out.append(f"      pctx[{j}].cst[0] = {pc[0]}; pctx[{j}].cst[1] = {pc[1]}; pctx[{j}].cst[2] = {pc[2]}; pctx[{j}].cst[3] = {pc[3]};")
        for d in range(4):
            out.append(f"      pctx[{j}].coord[{d}] = {exprs[d] if d < len(exprs) else 0};")
    out.append(f"      sym = {symbols.index(g.body.symbol)};   // {g.body.symbol}")
    return out


def emit_kernel(plan: Plan, device: int = 0) -> str:
    isolate_poller = os.environ.get("ETX_PREFETCH_ISOLATE_POLLER", "0") == "1"
    m = plan.machine
    is_hip = m.vendor == "amd"
    grids = [g for g in plan.graph.grids if g.device == device]
    # Opt-in shared dispatch: a prefetch sibling, like a tile body, needs one
    # call site rather than one per layer to keep inlining/register use bounded.
    shared_prefetch = bool(plan.graph.codegen.get("prefetch_dispatch")) and plan.options.prefetch and \
        m.capabilities.get("async_copy_to_lds", "none") != "none"
    prefetch_symbols = list(dict.fromkeys(g.body.prefetch for g in grids if g.body.prefetch)) if shared_prefetch else []
    threads = max([g.resource.threads for g in grids] + [64])
    lds = max([g.resource.lds_bytes + g.resource.prefetch_bytes for g in grids] + [0])
    out: list[str] = []
    out.append(f"// generated by etx: graph {plan.graph.name}, arch {m.name}, device {device}")
    out.append("// tile bodies are linked from the frontend (link mode); this file owns the persistent loop only")
    out.append("#ifndef ETX_EV_STRIDE   // event counter spacing in words, from the machine model (must match plan_data.h)")
    out.append(f"#define ETX_EV_STRIDE {int(m.lowering.get('event_stride_words', 1))}")
    out.append("#endif")
    out.extend(manifest.guard_block(plan.graph.codegen))   # before any tile source reads a feature macro
    out.append("#include \"etx/abi.h\"")
    out.append("#include \"etx_lowering.h\"")
    out.append("#include \"etx/primitives.h\"")
    out.append("")
    symbols: list[str] = []                     # distinct body symbols: one call site each (keeps inlining sane)
    for g in list(grids) + list(plan.inlined.values()):
        if g.body.symbol not in symbols:
            symbols.append(g.body.symbol)
    if plan.options.inline_tiles:
        out.append("// single-TU build: tile sources included so the bodies can inline into the dispatch")
        for src in dict.fromkeys(g.body.source for g in grids if g.body.source and g.body.kind in ("hip_link", "cuda_link")):
            out.append(f"#include \"{src}\"")
        if plan.graph.codegen.get("coh_l1"):
            # coherence mode l1 (ETX_COHERENCE): a DSL tile source emitted before etx/ops/coh.py had the agent-scope
            # variant would build and run correctly with system-scope loads -- under the wrong arm's name. Refuse it.
            out.append("#if defined(ETX_DSL_COH_HELPERS) && !defined(ETX_DSL_COH_L1_IMPL)")
            out.append('#error "coherence mode l1: the DSL tile source predates the agent-scope accessors '
                       '(etx/ops/coh.py); re-emit it with etx.ops.emit"')
            out.append("#endif")
    else:
        for g in grids:
            out.append(f"extern \"C\" __device__ void {g.body.symbol}(const etx_ctx*);")
            if g.body.prefetch:
                out.append(f"extern \"C\" __device__ void {g.body.prefetch}(const etx_ctx*);")
            if g.body.pre_wait:
                out.append(f"extern \"C\" __device__ void {g.body.pre_wait}(const etx_ctx*);")
            if g.body.touch:
                out.append(f"extern \"C\" __device__ void {g.body.touch}(const etx_ctx*);")
        for name, ig in plan.inlined.items():
            out.append(f"extern \"C\" __device__ void {ig.body.symbol}(const etx_ctx*);   // inlined producer {name}")
    for g in grids:
        for pro, m in g.prologue:
            out.append(f"// {g.name}: prologue recomputes {pro} via {m!r} (event eliminated by Pass 3)")
    out.append("")
    out.append("static __device__ __forceinline__ void etx_call_body(int sym, const etx_ctx* ctx) {")
    out.append("  switch (sym) {")
    for i, s in enumerate(symbols):
        out.append(f"    case {i}: {s}(ctx); break;")
    out.append("    default: break;")
    out.append("  }")
    out.append("}")
    out.append("")
    out.append(f"#define ETX_THREADS {threads}")
    out.append(f"#define ETX_LDS_USED {lds}")
    n_types = len(plan.type_ids)
    if prefetch_symbols:
        pf_ids = [-1] * n_types
        pf_consts = [[0] * 4 for _ in range(n_types)]
        for g in grids:
            if g.body.prefetch:
                pf_ids[plan.type_ids[g.name]] = prefetch_symbols.index(g.body.prefetch)
                pf_consts[plan.type_ids[g.name]] = list(g.consts)[:4] + [0] * (4 - min(4, len(g.consts)))
        out.append(f"static __device__ const int etx_prefetch_symbol[{n_types}] = {{{', '.join(map(str, pf_ids))}}};")
        out.append(f"static __device__ const int etx_prefetch_consts[{n_types}][4] = {{" +
                   ", ".join("{" + ", ".join(map(str, cs)) + "}" for cs in pf_consts) + "};")
        out.append("static __device__ __forceinline__ void etx_call_prefetch(int sym, const etx_ctx* ctx) {")
        out.append("  switch (sym) {")
        for i, symbol in enumerate(prefetch_symbols):
            out.append(f"    case {i}: {symbol}(ctx); break;")
        out.extend(["    default: break;", "  }", "}", ""])
    dyn_flags = ["0"] * n_types
    for g in plan.graph.grids:
        dyn_flags[plan.type_ids[g.name]] = "1" if plan.modes.get(g.name, "static") != "static" else "0"
    out.append(f"static __device__ const unsigned char etx_type_is_dynamic[{max(1, n_types)}] = {{{', '.join(dyn_flags) or '0'}}};")
    out.append("")
    static_only = all(t.mode == "static" for t in plan.tasks if t.device == device)
    bar2 = bool(plan.graph.codegen.get("bar2")) and static_only     # the two-barrier loop is a static-loop rewrite
    argpf = bool(plan.graph.codegen.get("argpf")) and static_only
    dev_reset = bool(plan.graph.codegen.get("dev_reset"))           # the kernel resets its own per-step state
    if argpf:
        # ETX_VL_ARGPF: wave 1 pulls the NEXT static task's argument block through the scalar path while thread 0
        # runs the current task's arrive (the block depends on no event; with a coherent build no acquire invalidate
        # follows, so the K$/L2 lines survive into the next task's argument reads). Per-type offset and size; 0 = skip.
        off, stride, nb = ["0"] * max(1, n_types), ["0"] * max(1, n_types), ["0"] * max(1, n_types)
        for g in plan.graph.grids:
            if g.body.arg_bytes and g.args:
                cs = list(g.consts)
                off[plan.type_ids[g.name]] = str(cs[0] if cs else 0)
                # per-domain replicas keep their blocks `stride` apart (cst[1]); the prefetch reads this worker's
                stride[plan.type_ids[g.name]] = str(cs[1] if len(cs) > 1 else 0)
                nb[plan.type_ids[g.name]] = str(g.body.arg_bytes)
        out.append("#ifdef ETX_HAS_IMPORTS   // ETX_VL_ARGPF: next-task argument prefetch (offset into the pool, per-domain stride, bytes)")
        out.append(f"static __device__ const int32_t etx_argpf_off[{max(1, n_types)}] = {{{', '.join(off)}}};")
        out.append(f"static __device__ const int32_t etx_argpf_str[{max(1, n_types)}] = {{{', '.join(stride)}}};")
        out.append(f"static __device__ const int32_t etx_argpf_bytes[{max(1, n_types)}] = {{{', '.join(nb)}}};")
        out.append("#endif")
        out.append("")
    # non-blocking readiness probe per task type (same edge maps as the wait code)
    out.append("static __device__ __forceinline__ bool etx_deps_ready(const etx_params& p, int32_t tid) {")
    out.append("  const etx_task t = p.descs[tid];")
    out.append("  switch (t.type) {")
    for g in grids:
        if not g.in_edges:
            continue
        out.append(f"    case {plan.type_ids[g.name]}: {{")
        coord_vars = [f"t.coord[{i}]" for i in range(len(g.grid))]
        for ev, m in g.in_edges.items():
            ep = plan.events[ev]
            width = _gather_width(plan, m.gather_tensor) if m.kind == "gather" else ""
            out.append(f"#define ETX_TARGET(idx) if (ETX_POLL_{ep.scope.name}(p.events + ETX_EVW({ep.offset} + (idx))) > 0) return false")
            out.append(m.to_c(coord_vars, f"p.ev_shape[{list(plan.events).index(ev)}]", "ETX_TARGET",
                              _runtime_ptrs(plan, g), width, _symbols(plan)).rstrip())
            out.append("#undef ETX_TARGET")
        out.append("      return true; }")
    out.append("    default: return true;")
    out.append("  }")
    out.append("}")
    out.append("")
    # Three phases, so every tile body has exactly ONE call site (two if it is also a Pass-3 prologue):
    # (1) per-type switch: immediates, waits, acquire, pick the body symbol; (2) one etx_call_body;
    # (3) per-type switch: releases and arrives. Emitting the body call inside each type's case made
    # 219 call sites per body on the DeepSeek port, so the compiler stopped inlining and paid the call
    # ABI (400 B/lane of scratch, vs fleet's 48 with one call site per body).
    max_pro = max([len(g.prologue) for g in grids] + [0])
    if bar2:
        # BAR2 (two-barrier loop): thread 0 advances the static queue into the OTHER parity slot before the
        # post-drain barrier, so that barrier doubles as the next task's broadcast and the loop-top barrier
        # disappears (two __syncthreads per task instead of four)
        out.append("static __device__ __forceinline__ void etx_run_task(const etx_params& p, const etx_task t, uint32_t worker, uint32_t domain, int32_t tid,")
        out.append("                                                     int32_t& cursor, const int32_t cend, int32_t* s_tid2, etx_task* s_task2, const int par) {")
    else:
        out.append("static __device__ __forceinline__ void etx_run_task(const etx_params& p, const etx_task t, uint32_t worker, uint32_t domain, int32_t tid) {")
    out.append("  __shared__ __align__(16) unsigned char etx_lds[ETX_LDS_USED > 0 ? ETX_LDS_USED : 16];")
    out.append("  uint64_t* tr = p.trace_time ? p.trace_time + (size_t)tid * 4 : nullptr;")
    out.append("  if (tr && threadIdx.x == 0) tr[0] = (uint64_t)ETX_TIMER();")
    out.append("  etx_ctx ctx; for (int i = 0; i < 4; ++i) ctx.coord[i] = t.coord[i];")
    out.append("  ctx.shape = p.shape; ctx.args = p.type_args + t.type * p.max_args; ctx.events = p.events; ctx.ev_offset = p.ev_offset; ctx.ev_shape = (const int32_t*)p.ev_shape;")
    out.append("  ctx.domain = domain; ctx.worker = worker; ctx.lds = etx_lds; ctx.cst[0] = ctx.cst[1] = ctx.cst[2] = ctx.cst[3] = 0;")
    if prefetch_symbols:
        out.append("  const int pf_sym = etx_prefetch_symbol[t.type];")
        out.append("  if (pf_sym >= 0) {")
        out.append("    for (int i = 0; i < 4; ++i) ctx.cst[i] = etx_prefetch_consts[t.type][i];")
        pf_guard = "if (threadIdx.x >= 64) " if isolate_poller else ""
        out.append(f"    {pf_guard}etx_call_prefetch(pf_sym, &ctx);   // Pass 7: immutable weights before this task's event wait")
        out.append("  }")
    out.append("  int sym = -1;")
    if max_pro:
        out.append(f"  etx_ctx pctx[{max_pro}]; int psym[{max_pro}];")
        out.append(f"  for (int j = 0; j < {max_pro}; ++j) psym[j] = -1;")
    out.append("  switch (t.type) {   // phase 1: immediates, waits, acquire")
    for g in grids:
        mode = plan.modes.get(g.name, "static")
        out.append(f"    case {plan.type_ids[g.name]}: {{ // {g.name} [{mode}] grid={g.grid}")
        out.extend(_setup_consts(g))
        if not shared_prefetch and g.body.prefetch and plan.options.prefetch and plan.machine.capabilities.get("async_copy_to_lds", "none") != "none":
            if isolate_poller:
                out.append(f"      if (threadIdx.x >= 64) {g.body.prefetch}(&ctx);   // isolate polling wave 0")
            else:
                out.append(f"      {g.body.prefetch}(&ctx);   // lever 3: weights do not depend on events; warm them before waiting")
        if g.body.pre_wait and g.in_edges:
            out.append("#ifndef ETX_NO_PRE_WAIT")
            if isolate_poller:
                # Exactly one non-polling wave warms the immutable argument block.
                # A one-wave launch has no spare wave and omits this optional hook.
                out.append(f"      if (threadIdx.x >= 64 && threadIdx.x < 128) {g.body.pre_wait}(&ctx);   // wave 1; wave 0 only polls")
            else:
                out.append(f"      if (threadIdx.x < 64) {g.body.pre_wait}(&ctx);   // before the wait: does not depend on the event")
            out.append("#endif")
        if g.body.touch and g.in_edges:
            # the waves that are not waiting are idle until the barrier after this switch: they pull the task's first
            # weight trips into L2 meanwhile (the weights depend on no event; the loads complete at that barrier, which
            # wave 0 reaches after its wait, so their latency overlaps the wait, not the arrive or the body)
            out.append("#ifndef ETX_NO_TOUCH")
            out.append(f"      if (threadIdx.x >= 64) {g.body.touch}(&ctx);   // waves 1..: this task's first weight trips into L2 while wave 0 waits")
            out.append("#endif")
        if g.in_edges:
            out.append("      if (threadIdx.x == 0) {")
            out.append(_wait_code(plan, g))
            # the acquire (cache invalidate) is a per-CU operation: issue it once, from the waiting thread,
            # not from all 4 waves (measured on MI300X: 4x the L2 invalidates slowed the phases after
            # DEVICE-scope events); the __syncthreads after the switch orders every other wave's loads after it
            if g.body.coherence == "coherent":
                # the tile's activation loads are device-coherent (sc0 sc1): nothing cached needs invalidating, and
                # the weights (and the touch hook's lines) stay in this XCD's L2 across the phase boundary
                out.append("        // coherent tile: no acquire invalidate (activation loads bypass the XCD caches)")
            elif g.body.coherence == "coherent_l1":
                # payload loads are agent scope (sc1): they skip this CU's stale L1 lines and read the XCD's L2, which a
                # same-XCD producer's drained stores reach and a cross-XCD producer's DEVICE flush wrote back
                out.append("        // coherent-L1 tile: no acquire invalidate (payload loads are agent scope)")
            elif g.body.coherence == "coherent_loads":
                # loads past the XCD caches as above; its own stores are plain, so its release writes back (below)
                out.append("        // coherent-load tile: no acquire invalidate (payload loads bypass the XCD caches; producers write back)")
            else:
                acq = sorted({_scope_macro(plan, ev) for ev in g.in_edges}, key=lambda s: Scope[s].value)
                relayed = any(_relayed(plan, plan.events[ev]) for ev in g.in_edges)
                for sc in acq:
                    if sc == "DEVICE" and relayed:
                        out.append("        if (etx_lok) ETX_ACQUIRE_DOMAIN(); else ETX_ACQUIRE_DEVICE();   // relay did the domain-level half")
                    else:
                        out.append(f"        ETX_ACQUIRE_{sc}();")
            out.append("      }")
        out.extend(_setup_body(plan, g, symbols))
        out.append("      break; }")
    out.append("    default: break;")
    out.append("  }")
    out.append("  __syncthreads();")
    out.append("  if (tr && threadIdx.x == 0) tr[1] = (uint64_t)ETX_TIMER();")
    if bar2:
        # BAR2: read the next head HERE, before the body, and keep it in registers. Issuing these two global loads
        # after the drain (where they first sat) put an HBM round trip on the publication path -- every dependent
        # worker waited for it, which measured +0.117 ms on 2026-10-01. Behind the body they cost nothing.
        out.append("  int32_t nx_tid = -1; etx_task nx_task;")
        out.append("  if (threadIdx.x == 0 && cursor < cend) { nx_tid = p.static_queue[cursor]; nx_task = p.static_descs[cursor]; ++cursor; }")
    if max_pro:
        out.append(f"  for (int j = 0; j < {max_pro}; ++j) if (psym[j] >= 0) {{ etx_call_body(psym[j], &pctx[j]); __syncthreads(); }}")
    out.append("  etx_call_body(sym, &ctx);   // phase 2: the only call site of each body")
    out.append("#ifndef ETX_NO_ALLWAVE_DRAIN")
    out.append("  ETX_RELEASE_DOMAIN();   // every wave drains its own stores before the barrier: thread 0's release only covers wave 0")
    out.append("#endif")
    if bar2:
        out.append("  // BAR2: publish the head fetched before the body into the other parity slot (write-after-read safe:")
        out.append("  // the running task reads slot par, this writes par ^ 1); the barrier below broadcasts it for free.")
        out.append("  // Registers only -- no global load between the drain and the barrier, which is the publication path.")
        out.append("  if (threadIdx.x == 0) { s_tid2[par ^ 1] = nx_tid; if (nx_tid >= 0) s_task2[par ^ 1] = nx_task; }")
    out.append("  __syncthreads();")
    out.append("  if (tr && threadIdx.x == 0) { tr[2] = (uint64_t)ETX_TIMER(); tr[3] = worker; }")
    out.append("  if (threadIdx.x != 0) return;")
    out.append("  switch (t.type) {   // phase 3: releases, arrives, pushes")
    for g in grids:
        mode = plan.modes.get(g.name, "static")
        body: list[str] = []
        for ev_name, e in plan.graph.events.items():
            if e.runtime_init_by == g.name and e.runtime_count is not None:
                ep = plan.events[ev_name]
                expr = _runtime_count_c(plan, e.runtime_count)
                body.append(f"        for (int i = 0; i < p.ev_shape[{list(plan.events).index(ev_name)}][0]; ++i) "
                            f"p.events[ETX_EVW({ep.offset} + i)] = {expr};   // runtime init of {ev_name}")
        for ev in g.out_edges:
            ep = plan.events[ev]
            if ep.scope != Scope.DEVICE:                       # DEVICE-scope arrives carry their own (last-arriver) release
                if ep.coherent_consumer and g.body.coherence != "coherent":
                    body.append("        ETX_RELEASE_DEVICE();   // a coherent consumer bypasses this XCD's L2: write it back")
                else:
                    body.append(f"        ETX_RELEASE_{_scope_macro(plan, ev)}();")
        if g.out_edges:
            body.append(_arrive_code(plan, g, mode))
        if body:
            out.append(f"    case {plan.type_ids[g.name]}: {{ // {g.name}")
            out.extend(body)
            out.append("      break; }")
    out.append("    default: break;")
    out.append("  }")
    out.append("}")
    out.append("")
    out.append("#ifndef ETX_MIN_WG_PER_CU   // the plan's residency (P1). NOTE: launch_bounds' second argument is MIN")
    out.append("// WAVES PER EU, not workgroups per CU, and for this persistent kernel (one 16-wave workgroup per CU by")
    out.append("// construction) it cannot raise occupancy at all -- it only caps VGPRs and forces spills. Round 34: =8 gave")
    out.append("// VGPR 64 / 1678 spills / runtime occupancy still 4.000 / 9.39 wall against 4.88. Leave it at the default.")
    out.append(f"#define ETX_MIN_WG_PER_CU {plan.wg_per_cu}")
    out.append("#endif")
    out.append("#if ETX_MIN_WG_PER_CU > 0")
    out.append(f"extern \"C\" __global__ void __launch_bounds__(ETX_THREADS, ETX_MIN_WG_PER_CU) etx_megakernel_d{device}(etx_params p) {{")
    out.append("#else")
    out.append(f"extern \"C\" __global__ void __launch_bounds__(ETX_THREADS) etx_megakernel_d{device}(etx_params p) {{")
    out.append("#endif")
    out.append("  // Logical worker id = domain * workers_per_domain + slot, where the slot is claimed at start.")
    out.append("  // This makes the static queues domain-affine under ANY workgroup->domain mapping (measured (k+6) mod 8 on one VM).")
    out.append("  __shared__ uint32_t s_domain, s_worker; __shared__ int32_t s_tid, s_slot;")
    out.append("#ifdef ETX_HAS_IMPORTS")
    out.append("  etx_import_init();   // imported tiles' slot barriers (etx/import.h); ordered by the barrier below")
    out.append("#endif")
    out.append("  if (threadIdx.x == 0) {")
    out.append("    const uint32_t d = etx_discover_domain(p, blockIdx.x);")
    out.append("#ifdef ETX_SLOT_BY_CU")
    out.append("    // pair workgroups by CU: the k-th workgroup to arrive on the i-th CU of the domain takes slot k * cus + i, so")
    out.append("    // slot w and slot w + cus share a CU (two requests interleaved one per workgroup; docs/PLAN-v3.md 4.4)")
    out.append("    int32_t* cu_count = p.domain_slots + ETX_NUM_DOMAINS;   // layout: etx/host.h ETX_SLOT_WORDS")
    out.append("    int32_t* cu_k = cu_count + ETX_NUM_DOMAINS;")
    out.append("    int32_t* cu_map = cu_k + ETX_NUM_DOMAINS * 256;")
    out.append("    const uint32_t key = etx_amdgcn_cu_key();")
    out.append("    int32_t* mapped = cu_map + d * 256 + key;")
    out.append("    int32_t idx = -1;")
    out.append("    if (atomicCAS(mapped, -1, -2) == -1) { idx = atomicAdd(cu_count + d, 1); __hip_atomic_store(mapped, idx, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT); }")
    out.append("    else { while ((idx = __hip_atomic_load(mapped, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) < 0) {} }")
    out.append("    const int32_t k = atomicAdd(cu_k + d * 256 + key, 1);")
    out.append(f"    const int32_t slot = (idx < ETX_CUS_PER_DOMAIN && k < {plan.wg_per_cu}) ? k * ETX_CUS_PER_DOMAIN + idx : 0x7FFFFFFF;")
    out.append("#else")
    out.append("    const int32_t slot = atomicAdd(p.domain_slots + d, 1);")
    out.append("#endif")
    out.append("    s_domain = d; s_slot = slot; s_worker = (slot < p.workers_per_domain) ? d * p.workers_per_domain + slot : 0xFFFFFFFFu;")
    out.append("  }")
    out.append("  __syncthreads();")
    out.append("  const uint32_t domain = s_domain, worker = s_worker;")
    if plan.relay:
        out.append("  if (worker == 0xFFFFFFFFu && s_slot == p.workers_per_domain && p.ev_mirror) {   // first surplus workgroup of each domain: its relay (P5)")
        out.append("    etx_relay(p, domain);")
        out.append("    return;")
        out.append("  }")
    out.append("  int32_t cursor = 0, cend = 0;")
    out.append("  if (worker != 0xFFFFFFFFu) { cursor = p.static_begin[worker]; cend = p.static_end[worker]; }   // surplus workers only serve queues")
    if static_only:
        out.append("  // every task on this device is static: no readiness probe, no queue pops -- take the head and")
        out.append("  // wait on it (the probe was one more poller of the same word and one more round trip per task)")
        if bar2:
            out.append("  // BAR2: double-buffered head (parity slots); the fetch of slot par ^ 1 happens inside")
            out.append("  // etx_run_task before its post-drain barrier, so each task costs exactly two __syncthreads")
            out.append("  __shared__ etx_task s_task2[2]; __shared__ int32_t s_tid2[2];")
            out.append("  if (threadIdx.x == 0) {   // preamble: the first head into slot 0")
            out.append("    if (cursor < cend) { s_tid2[0] = p.static_queue[cursor]; s_task2[0] = p.static_descs[cursor]; ++cursor; } else s_tid2[0] = -1;")
            out.append("  }")
            out.append("  __syncthreads();")
            out.append("  int par = 0;")
            out.append("  for (;;) {")
            out.append("    const int32_t tid = s_tid2[par];")
            out.append("    if (tid < 0) break;")
            out.append("    const etx_task cur = s_task2[par];")
            out.append("    if (p.trace_exec && threadIdx.x == 0) atomicAdd(p.trace_exec + tid, 1);")
            out.append("    etx_run_task(p, cur, worker, domain, tid, cursor, cend, s_tid2, s_task2, par);")
            if argpf:
                out.append("#ifdef ETX_HAS_IMPORTS   // ETX_VL_ARGPF: wave 1 pulls the NEXT task's argument block while thread 0 arrives")
                out.append("    if (threadIdx.x - 64u < 64u && s_tid2[par ^ 1] >= 0) {")
                out.append("      const int32_t ntt = s_task2[par ^ 1].type;")
                out.append("      if (etx_argpf_bytes[ntt]) etx_imp_touch(etx_imp_argblock(p.type_args + ntt * p.max_args,")
                out.append("                                                    etx_argpf_off[ntt] + (int32_t)domain * etx_argpf_str[ntt]), etx_argpf_bytes[ntt]);")
                out.append("    }")
                out.append("#endif")
            out.append("    par ^= 1;")
            out.append("  }")
        else:
            out.append("  __shared__ etx_task s_task;")
            if argpf:
                out.append("  __shared__ int32_t s_ncur;   // ETX_VL_ARGPF: the queue position of the NEXT head")
            out.append("  for (;;) {")
            out.append("    if (threadIdx.x == 0) {")
            if argpf:
                out.append("      if (cursor < cend) { s_tid = p.static_queue[cursor]; s_task = p.static_descs[cursor]; ++cursor; } else s_tid = -1;")
                out.append("      s_ncur = cursor;")
            else:
                out.append("      if (cursor < cend) { s_tid = p.static_queue[cursor]; s_task = p.static_descs[cursor]; ++cursor; } else s_tid = -1;")
            out.append("    }")
            out.append("    __syncthreads();")
            out.append("    const int32_t tid = s_tid;")
            out.append("    if (tid < 0) break;")
            out.append("    const etx_task cur = s_task;")
            out.append("    if (p.trace_exec && threadIdx.x == 0) atomicAdd(p.trace_exec + tid, 1);")
            out.append("    etx_run_task(p, cur, worker, domain, tid);")
            if argpf:
                out.append("#ifdef ETX_HAS_IMPORTS   // ETX_VL_ARGPF: wave 1 pulls the NEXT task's argument block while thread 0 arrives")
                out.append("    if (threadIdx.x - 64u < 64u && s_ncur < cend) {")
                out.append("      const etx_task nt = p.static_descs[s_ncur];")
                out.append("      if (etx_argpf_bytes[nt.type]) etx_imp_touch(etx_imp_argblock(p.type_args + nt.type * p.max_args,")
                out.append("                                                    etx_argpf_off[nt.type] + (int32_t)domain * etx_argpf_str[nt.type]), etx_argpf_bytes[nt.type]);")
                out.append("    }")
                out.append("#endif")
            out.append("    __syncthreads();")
            out.append("  }")
        if dev_reset:
            out.extend(_reset_tail())
        out.append("}")
        out.append("")
        out.extend(_emit_unfused(plan, device, grids, symbols, max_pro))
        return _diagnostic_source("\n".join(out))
    out.append("  int32_t ticket_local = -1, ticket_global = -1;   // ticket-ring reservations, see etx_try_pop")
    out.append("  uint32_t spins = 0;")
    out.append("  for (;;) {")
    out.append("    if (threadIdx.x == 0) {")
    out.append("      int32_t tid = -1;")
    out.append("      if (cursor < cend) { int32_t h = p.static_queue[cursor]; if (etx_deps_ready(p, h)) { tid = h; ++cursor; } }")
    out.append("      if (tid < 0) tid = etx_try_pop(p.local_queue + domain, &ticket_local);")
    out.append("      if (tid < 0) tid = etx_try_pop(&p.global_queue, &ticket_global);")
    out.append("      if (tid < 0 && cursor < cend) { tid = p.static_queue[cursor]; ++cursor; }   // spin on the head inside the wait code")
    out.append("      s_tid = tid;")
    out.append("    }")
    out.append("    __syncthreads();")
    out.append("    const int32_t tid = s_tid;")
    out.append("    if (tid < 0) {")
    out.append("      if (etx_step_done(p)) break;                 // static queue drained and no dynamic work left")
    out.append("      if (ETX_POLL_DEVICE(p.ctrl_abort)) break;")
    out.append("      ETX_BACKOFF(); if (++spins > p.spin_limit) { if (threadIdx.x == 0) atomicOr(p.ctrl_abort, 2); break; }")
    out.append("      __syncthreads(); continue;")
    out.append("    }")
    out.append("    spins = 0;")
    out.append("    if (p.trace_exec && threadIdx.x == 0) atomicAdd(p.trace_exec + tid, 1);")
    out.append("    const etx_task cur = p.descs[tid];")
    out.append("    etx_run_task(p, cur, worker, domain, tid);")
    out.append("    if (threadIdx.x == 0 && etx_type_is_dynamic[cur.type]) atomicAdd(p.ctrl_done, 1);   // static tasks cost no atomic")
    out.append("    __syncthreads();")
    out.append("  }")
    out.append("}")
    out.append("")
    out.extend(_emit_unfused(plan, device, grids, symbols, max_pro))
    return _diagnostic_source("\n".join(out))


def _diagnostic_source(source):
    # An independent diagnostic TU. Ordinary emission remains byte-for-byte
    # unchanged; host and device must both compile with ETX_INBODY=1.
    import os
    if os.environ.get("ETX_INBODY_EMIT") == "1":
        from .inbody import instrument
        source = instrument(source)
    return source


def _reset_tail() -> list[str]:
    """dev_reset: the last workgroup out rewrites the per-step state so the host enqueues nothing.

    Emitted from plan.graph.codegen, never from a compile define: plan_data.h publishes the same decision as
    ETX_EMIT_DEV_RESET, so the host cannot believe the tail is present when it is not (see _feature_manifest).

    Ordering. A workgroup increments the exit counter only after its loop has ended, so when the count reaches
    gridDim.x every worker has run its last arrive. But the arrive and the counter are both RELAXED atomics on
    different addresses, so the counter reaching gridDim.x does not by itself mean the arrives have landed: a
    late arrive could decrement an event word the reset had already refreshed, leaving the next step one short.
    The release fence before the increment and the acquire fence after the full count close exactly that gap --
    once per workgroup per step, not per task."""
    return ["  if (p.n_reset_ops) {   // last workgroup out rewrites the per-step state; no grid sync, no cooperative",
            "    __syncthreads();",
            "    __shared__ int32_t s_last;",
            "    if (threadIdx.x == 0) {",
            "      __threadfence();   // release: this worker's arrives land before its increment is visible",
            "      s_last = (atomicAdd(p.ctrl_abort + 2, 1) + 1 == (int32_t)gridDim.x);",
            "      if (s_last) __threadfence();   // acquire: every other worker's arrives are visible before the reset",
            "    }",
            "    __syncthreads();",
            "    if (s_last) etx_device_reset(p);",
            "  }"]


def _emit_unfused(plan: Plan, device: int, grids, symbols: list[str], max_pro: int) -> list[str]:
    """Unfused baseline: one ordinary kernel launch per task grid (the host replays them in plan order,
    optionally as one HIP graph); one workgroup per task, the SAME tile bodies, no events -- the kernel
    boundary is the only synchronisation. Measures what the megakernel itself buys."""
    out = ["", "// ---- unfused baseline: launched once per grid, blockIdx = task within the grid (see etx_host::run_unfused)",
           "#if ETX_MIN_WG_PER_CU > 0",
           f"extern \"C\" __global__ void __launch_bounds__(ETX_THREADS, ETX_MIN_WG_PER_CU) etx_unfused_d{device}(etx_params p, int32_t first) {{",
           "#else",
           f"extern \"C\" __global__ void __launch_bounds__(ETX_THREADS) etx_unfused_d{device}(etx_params p, int32_t first) {{",
           "#endif",
           "  __shared__ __align__(16) unsigned char etx_lds[ETX_LDS_USED > 0 ? ETX_LDS_USED : 16];",
           "#ifdef ETX_HAS_IMPORTS",
           "  etx_import_init(); __syncthreads();",
           "#endif",
           "  const int32_t tid = first + (int32_t)blockIdx.x;",
           "  const etx_task t = p.descs[tid];",
           "  etx_ctx ctx; for (int i = 0; i < 4; ++i) ctx.coord[i] = t.coord[i];",
           "  ctx.shape = p.shape; ctx.args = p.type_args + t.type * p.max_args; ctx.events = p.events; ctx.ev_offset = p.ev_offset; ctx.ev_shape = (const int32_t*)p.ev_shape;",
           "  ctx.domain = etx_discover_domain(p, blockIdx.x); ctx.worker = blockIdx.x; ctx.lds = etx_lds; ctx.cst[0] = ctx.cst[1] = ctx.cst[2] = ctx.cst[3] = 0;",
           "  int sym = -1;"]
    if max_pro:
        out.append(f"  etx_ctx pctx[{max_pro}]; int psym[{max_pro}];")
        out.append(f"  for (int j = 0; j < {max_pro}; ++j) psym[j] = -1;")
    out.append("  switch (t.type) {")
    for g in grids:
        out.append(f"    case {plan.type_ids[g.name]}: {{ // {g.name}")
        out.extend(_setup_consts(g))
        out.extend(_setup_body(plan, g, symbols))
        out.append("      break; }")
    out.append("    default: break;")
    out.append("  }")
    if max_pro:
        out.append(f"  for (int j = 0; j < {max_pro}; ++j) if (psym[j] >= 0) {{ etx_call_body(psym[j], &pctx[j]); __syncthreads(); }}")
    out.append("  etx_call_body(sym, &ctx);")
    out.append("}")
    out.append("")
    return out


def emit_plan_json(plan: Plan) -> str:
    args = _arg_table(plan)
    data: dict[str, Any] = {
        "inlined": {n: {"id": plan.type_ids[n], "symbol": gg.body.symbol} for n, gg in plan.inlined.items()},
        "graph": plan.graph.name, "arch": plan.arch, "bindings": plan.bindings,
        "symbols": plan.graph.symbols, "shape": [plan.bindings.get(s) for s in plan.graph.symbols],
        "devices": plan.n_devices, "domains": plan.n_domains, "workers_per_domain": plan.workers_per_domain,
        "threads": max([g.resource.threads for g in plan.graph.grids] + [64]),
        "args": args,
        "types": {g.name: {"id": plan.type_ids[g.name], "symbol": g.body.symbol, "mode": plan.modes.get(g.name),
                           "grid": list(plan.inst.grid_shapes[g.name]), "prologue": [list(x) for x in g.prologue], "device": g.device,
                           "touch": g.body.touch, "coherence": g.body.coherence}
                  for g in plan.graph.grids},
        "events": [{"name": e.name, "id": i, "offset": e.offset, "shape": list(e.shape), "scope": e.scope.name,
                    "memory": e.memory, "counts": e.counts, "runtime_init": e.runtime_init}
                   for i, e in enumerate(plan.events.values())],
        "descs": [{"id": t.id, "type": t.type_id, "coord": list(t.coord), "device": t.device, "domain": t.domain,
                   "worker": t.worker, "mode": t.mode} for t in plan.tasks],
        "static_queues": {f"{d}:{w}": q for (d, w), q in plan.static_queues.items()},
        "local_queue_capacity": {f"{d}:{dom}": c for (d, dom), c in plan.local_queue_capacity.items()},
        "global_queue_capacity": plan.global_queue_capacity,
        # push lists: hybrid consumer -> (its domain << 24) | task id; dynamic consumer -> ~task id (global queue)
        "ev_consumers": {f"{ev}:{lin}": [((plan.tasks[i].domain << 24) | i) if plan.tasks[i].mode == "hybrid" else ~i for i in ids]
                         for (ev, lin), ids in plan.ev_consumers.items()},
        "tensor_placement": plan.tensor_placement,
        "prefetch": plan.prefetch,
        "eliminated_events": plan.eliminated_events,
        "kernel_instances": plan.kernel_instances,
        "log": plan.log,
    }
    # dependencies, for trace analysis (etx.tools.timeline: critical path): every event instance once with its
    # producer task ids, and per task the indices of the event instances it waits on
    ev_ids: dict[Any, int] = {}
    ev_inst: list[list[Any]] = []
    for ev, prods in plan.inst.producers.items():
        ev_ids[ev] = len(ev_inst)
        ev_inst.append([ev[0], list(ev[1]), [plan.task_index[p] for p in prods]])
    data["ev_inst"] = ev_inst
    data["task_in"] = [[ev_ids[e] for e in plan.inst.task_in[(t.grid, t.coord)] if e in ev_ids] for t in plan.tasks]
    return json.dumps(data, indent=1)
