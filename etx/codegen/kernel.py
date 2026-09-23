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

import json
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


_ARG_CACHE: dict[int, dict[str, int]] = {}


def _arg_table(plan: Plan) -> dict[str, int]:
    key = id(plan)
    if key not in _ARG_CACHE:
        names: list[str] = []
        for t in plan.graph.tensors:
            names.append(t)
        _ARG_CACHE[key] = {n: i for i, n in enumerate(names)}
    return _ARG_CACHE[key]


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
            lines.append(f"#define ETX_TARGET(idx) etx_wait_{sc}(p.events + {ep.offset} + (idx), p.ctrl_abort)")
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
        arrive = (f"etx_arrive_flush_DEVICE(p, {ep.offset} + (idx), domain)" if ep.scope == Scope.DEVICE
                  else f"etx_arrive_{sc}(p.events + {ep.offset} + (idx))")
        if not has_dynamic_consumers:       # the producer's own mode is irrelevant: a static producer must still push dynamic consumers
            lines.append(f"#define ETX_TARGET(idx) (void){arrive}")
        else:
            lines.append(f"#define ETX_TARGET(idx) do {{ if ({arrive} == 0) etx_push_consumers(p, {ev_id}, (idx), domain); }} while (0)")
        lines.append(m.to_c(coord_vars, f"p.ev_shape[{ev_id}]", "ETX_TARGET", _runtime_ptrs(plan, g), width, _symbols(plan)).rstrip())
        lines.append("#undef ETX_TARGET")
    return "\n".join(lines)


def emit_kernel(plan: Plan, device: int = 0) -> str:
    m = plan.machine
    is_hip = m.vendor == "amd"
    grids = [g for g in plan.graph.grids if g.device == device]
    threads = max([g.resource.threads for g in grids] + [64])
    lds = max([g.resource.lds_bytes + g.resource.prefetch_bytes for g in grids] + [0])
    out: list[str] = []
    out.append(f"// generated by etx: graph {plan.graph.name}, arch {m.name}, device {device}")
    out.append("// tile bodies are linked from the frontend (link mode); this file owns the persistent loop only")
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
    else:
        for g in grids:
            out.append(f"extern \"C\" __device__ void {g.body.symbol}(const etx_ctx*);")
            if g.body.prefetch:
                out.append(f"extern \"C\" __device__ void {g.body.prefetch}(const etx_ctx*);")
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
    dyn_flags = ["0"] * n_types
    for g in plan.graph.grids:
        dyn_flags[plan.type_ids[g.name]] = "1" if plan.modes.get(g.name, "static") != "static" else "0"
    out.append(f"static __device__ const unsigned char etx_type_is_dynamic[{max(1, n_types)}] = {{{', '.join(dyn_flags) or '0'}}};")
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
            out.append(f"#define ETX_TARGET(idx) if (ETX_POLL_{ep.scope.name}(p.events + {ep.offset} + (idx)) > 0) return false")
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
    out.append("static __device__ __forceinline__ void etx_run_task(const etx_params& p, const etx_task t, uint32_t worker, uint32_t domain, int32_t tid) {")
    out.append("  __shared__ __align__(16) unsigned char etx_lds[ETX_LDS_USED > 0 ? ETX_LDS_USED : 16];")
    out.append("  uint64_t* tr = p.trace_time ? p.trace_time + (size_t)tid * 4 : nullptr;")
    out.append("  if (tr && threadIdx.x == 0) tr[0] = (uint64_t)ETX_TIMER();")
    out.append("  etx_ctx ctx; for (int i = 0; i < 4; ++i) ctx.coord[i] = t.coord[i];")
    out.append("  ctx.shape = p.shape; ctx.args = p.type_args + t.type * p.max_args; ctx.events = p.events; ctx.ev_offset = p.ev_offset; ctx.ev_shape = (const int32_t*)p.ev_shape;")
    out.append("  ctx.domain = domain; ctx.worker = worker; ctx.lds = etx_lds; ctx.cst[0] = ctx.cst[1] = ctx.cst[2] = ctx.cst[3] = 0;")
    out.append("  int sym = -1;")
    if max_pro:
        out.append(f"  etx_ctx pctx[{max_pro}]; int psym[{max_pro}];")
        out.append(f"  for (int j = 0; j < {max_pro}; ++j) psym[j] = -1;")
    out.append("  switch (t.type) {   // phase 1: immediates, waits, acquire")
    for g in grids:
        mode = plan.modes.get(g.name, "static")
        out.append(f"    case {plan.type_ids[g.name]}: {{ // {g.name} [{mode}] grid={g.grid}")
        if g.consts:
            cs = list(g.consts)[:4] + [0] * (4 - min(4, len(g.consts)))
            out.append(f"      ctx.cst[0] = {cs[0]}; ctx.cst[1] = {cs[1]}; ctx.cst[2] = {cs[2]}; ctx.cst[3] = {cs[3]};")
        if g.body.prefetch and plan.options.prefetch and plan.machine.capabilities.get("async_copy_to_lds", "none") != "none":
            out.append(f"      {g.body.prefetch}(&ctx);   // lever 3: weights do not depend on events; warm them before waiting")
        if g.in_edges:
            out.append("      if (threadIdx.x == 0) {")
            out.append(_wait_code(plan, g))
            # the acquire (cache invalidate) is a per-CU operation: issue it once, from the waiting thread,
            # not from all 4 waves (measured on MI300X: 4x the L2 invalidates slowed the phases after
            # DEVICE-scope events); the __syncthreads after the switch orders every other wave's loads after it
            acq = sorted({_scope_macro(plan, ev) for ev in g.in_edges}, key=lambda s: Scope[s].value)
            relayed = any(_relayed(plan, plan.events[ev]) for ev in g.in_edges)
            for sc in acq:
                if sc == "DEVICE" and relayed:
                    out.append("        if (etx_lok) ETX_ACQUIRE_DOMAIN(); else ETX_ACQUIRE_DEVICE();   // relay did the domain-level half")
                else:
                    out.append(f"        ETX_ACQUIRE_{sc}();")
            out.append("      }")
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
        out.append("      break; }")
    out.append("    default: break;")
    out.append("  }")
    out.append("  __syncthreads();")
    out.append("  if (tr && threadIdx.x == 0) tr[1] = (uint64_t)ETX_TIMER();")
    if max_pro:
        out.append(f"  for (int j = 0; j < {max_pro}; ++j) if (psym[j] >= 0) {{ etx_call_body(psym[j], &pctx[j]); __syncthreads(); }}")
    out.append("  etx_call_body(sym, &ctx);   // phase 2: the only call site of each body")
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
                            f"p.events[{ep.offset} + i] = {expr};   // runtime init of {ev_name}")
        for ev in g.out_edges:
            if plan.events[ev].scope != Scope.DEVICE:          # DEVICE-scope arrives carry their own (last-arriver) release
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
    static_only = all(t.mode == "static" for t in plan.tasks if t.device == device)
    out.append(f"extern \"C\" __global__ void __launch_bounds__(ETX_THREADS) etx_megakernel_d{device}(etx_params p) {{")
    out.append("  // Logical worker id = domain * workers_per_domain + slot, where the slot is claimed at start.")
    out.append("  // This makes the static queues domain-affine under ANY workgroup->domain mapping (measured (k+6) mod 8 on one VM).")
    out.append("  __shared__ uint32_t s_domain, s_worker; __shared__ int32_t s_tid, s_slot;")
    out.append("  if (threadIdx.x == 0) {")
    out.append("    const uint32_t d = etx_discover_domain(p, blockIdx.x);")
    out.append("    const int32_t slot = atomicAdd(p.domain_slots + d, 1);")
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
        maxq = max([len(q) for (d, w), q in plan.static_queues.items() if d == device] + [0])
        lds_budget = int(plan.machine.resources["lds_kb"]) * 1024
        stage = maxq > 0 and lds + maxq * 24 + 1024 <= lds_budget
        out.append("  // every task on this device is static: no readiness probe, no queue pops -- take the head and")
        out.append("  // wait on it (the probe was one more poller of the same word and one more round trip per task)")
        if stage:
            out.append(f"  // descriptor residency: this worker's whole queue ({maxq} max) is staged in spare LDS once, so taking the")
            out.append("  // next task is an LDS read instead of an HBM miss on the critical path of the phase's last worker")
            out.append(f"  __shared__ etx_task s_q[{maxq}]; __shared__ int32_t s_qid[{maxq}];")
            out.append("  const int32_t qn = cend - cursor;")
            out.append("  for (int32_t i = threadIdx.x; i < qn; i += blockDim.x) { s_q[i] = p.static_descs[cursor + i]; s_qid[i] = p.static_queue[cursor + i]; }")
            out.append("  __syncthreads();")
            out.append("  for (int32_t k = 0; k < qn; ++k) {")
            out.append("    const int32_t tid = s_qid[k];")
            out.append("    const etx_task cur = s_q[k];")
        else:
            out.append("  __shared__ etx_task s_task;")
            out.append("  for (;;) {")
            out.append("    if (threadIdx.x == 0) {")
            out.append("      if (cursor < cend) { s_tid = p.static_queue[cursor]; s_task = p.static_descs[cursor]; ++cursor; } else s_tid = -1;")
            out.append("    }")
            out.append("    __syncthreads();")
            out.append("    const int32_t tid = s_tid;")
            out.append("    if (tid < 0) break;")
            out.append("    const etx_task cur = s_task;")
        out.append("    if (p.trace_exec && threadIdx.x == 0) atomicAdd(p.trace_exec + tid, 1);")
        out.append("    etx_run_task(p, cur, worker, domain, tid);")
        out.append("    __syncthreads();")
        out.append("  }")
        out.append("}")
        out.append("")
        return "\n".join(out)
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
    return "\n".join(out)


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
                           "grid": list(plan.inst.grid_shapes[g.name]), "prologue": [list(x) for x in g.prologue], "device": g.device}
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
    return json.dumps(data, indent=1)
