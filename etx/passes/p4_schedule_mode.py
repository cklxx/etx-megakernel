"""Pass 4: scheduling mode per task grid, with printed reasons.

    S_balance = straggler time dynamic scheduling is expected to recover
    C_cross   = cross-domain pushes x T_cross (cross-device: prohibitive)
    C_queue   = pop contention
    dynamic  if S_balance > C_cross + C_queue and no cross-domain edges
    hybrid   if S_balance > C_cross + C_queue and cross-domain edges exist
    static   otherwise; always static across devices
Data-dependent grids never go static (static would degrade to an E[0] barrier).
"""
from __future__ import annotations

import math
from dataclasses import replace

from ..ir.types import Scope
from .plan import Plan

MARGIN = 1.5     # bias toward static: the paper's regular workloads lose 6-17% under dynamic scheduling


COHERENCE_MODES = {"declared": None, "l1": "coherent_l1"}


def _stores_coherent(body) -> bool:
    """A producer's payload is past its L2 without a write-back: a "coherent" tile, or a "coherent_l1" tile whose family
    declares its stores write-through ("coherent_stores": the DSL ops; not fleet's tiles, whose stores are plain)."""
    return body.coherence == "coherent" or (body.coherence == "coherent_l1" and "coherent_stores" in body.coherence_caps)


def apply_coherence_mode(plan: Plan) -> None:
    """The build-wide coherence switch (PassOptions.coherence / ETX_COHERENCE): every grid whose tile family declares
    the target mode in `coherence_caps` takes it; nothing else changes. "declared" (the default) touches nothing."""
    mode = plan.options.coherence
    if mode not in COHERENCE_MODES:
        raise ValueError(f"coherence={mode!r}: one of {sorted(COHERENCE_MODES)} (ETX_COHERENCE)")
    target = COHERENCE_MODES[mode]
    if target is None:
        return
    switched = [g for g in plan.graph.grids if target in g.body.coherence_caps]
    for g in switched:
        g.body = replace(g.body, coherence=target)
    if switched:
        plan.graph.codegen["coh_l1"] = 1          # the DSL accessors' agent-scope variant (etx/ops/coh.py)
    plan.say(f"P4: coherence={mode}: {len(switched)} of {len(plan.graph.grids)} grids -> {target} "
             f"(tile families declaring it); the rest keep their declared coherence")


def run(plan: Plan) -> None:
    apply_coherence_mode(plan)
    m = plan.machine
    inst = plan.inst
    total_workers = max(1, plan.workers_per_device())
    for g in plan.graph.grids:
        n = len(inst.tasks[g.name])
        waves = max(1, math.ceil(n / total_workers))
        cross_dom = cross_dev = 0
        for c in inst.tasks[g.name]:
            t = plan.task((g.name, c))
            for ev in inst.task_in[(g.name, c)]:
                for p in inst.producers[ev]:
                    pt = plan.task(p)
                    if pt.device != t.device:
                        cross_dev += 1
                    elif pt.domain != t.domain:
                        cross_dom += 1
        s_balance = g.duration_cv * g.duration_us * waves * 2.0
        c_cross = cross_dom * m.t_push_us(cross_domain=True) / total_workers + (waves * m.t_sync_us(Scope.DEVICE) if cross_dom else 0.0)
        sharing = total_workers if cross_dom == 0 else plan.workers_per_domain
        c_queue = waves * m.t_pop_us() * (1.0 + 0.01 * sharing)
        if plan.options.force_mode:
            mode, why = plan.options.force_mode, "forced by options"
        elif cross_dev > 0:
            mode, why = "static", f"{cross_dev} cross-device edges: pushes over P2P are prohibitive (ETC TP=4 dynamic 0.83x)"
        elif g.has_runtime_edges and s_balance > MARGIN * (c_cross + c_queue):
            mode = "hybrid" if cross_dom else "dynamic"
            why = f"data-dependent edges and S_balance={s_balance:.2f}us > {MARGIN}x(C_cross={c_cross:.2f}+C_queue={c_queue:.2f})"
        elif g.has_runtime_edges:
            # measured on MI300X (small MoE, B=64): forced static with the runtime-map barrier
            # ran 0.445 ms vs 0.654 ms hybrid; the barrier costs less than the queue path for short tasks
            mode = "static"
            why = f"data-dependent edges but S_balance={s_balance:.2f}us <= {MARGIN}x(C_cross={c_cross:.2f}+C_queue={c_queue:.2f}); static with a barrier on the runtime-tensor writer"
        elif s_balance > MARGIN * (c_cross + c_queue):
            mode = "dynamic" if cross_dom == 0 else "hybrid"
            why = f"S_balance={s_balance:.2f}us > {MARGIN}x(C_cross={c_cross:.2f}+C_queue={c_queue:.2f}); cross-domain edges={cross_dom}"
        else:
            mode, why = "static", f"S_balance={s_balance:.2f}us <= {MARGIN}x(C_cross={c_cross:.2f}+C_queue={c_queue:.2f})"
        plan.modes[g.name] = mode
        plan.reasons[g.name] = why
        for c in inst.tasks[g.name]:
            plan.task((g.name, c)).mode = mode
        plan.say(f"P4: {g.name}: {mode} ({why})")
    # A statically scheduled grid with runtime edge maps evaluates those maps
    # when its task is taken, which can be before the runtime tensors are written.
    # Dynamic/hybrid tasks are only pushed after their producers, so they are
    # safe; static ones get an explicit "*" wait on the writer's out-events first
    # (the E[0]-barrier degradation the paper describes). Caught on MI300X:
    # forced-static MoE read tile_expert before grouping wrote it.
    from ..ir.edgemap import EdgeMap
    for g in plan.graph.grids:
        if plan.modes[g.name] != "static" or not g.has_runtime_edges:
            continue
        rts = {rt for mm in list(g.in_edges.values()) + list(g.out_edges.values()) for rt in mm.runtime}
        barrier: dict[str, EdgeMap] = {}
        for rt in rts:
            for w in plan.graph.grids:
                if rt in w.writes:
                    for ev in w.out_edges:
                        barrier[ev] = EdgeMap.parse("*")
        if barrier:
            g.in_edges = {**barrier, **{k: v for k, v in g.in_edges.items() if k not in barrier}}   # barrier waits come first
            plan.say(f"P4: {g.name}: static with runtime maps -> waits on all of {list(barrier)} before evaluating them")
            plan.inst = None  # type: ignore  # re-instantiated by the pipeline below
    if plan.inst is None:
        from ..ir.instantiate import instantiate
        plan.inst = instantiate(plan.graph, plan.bindings, plan.runtime)
    # A globally scheduled (dynamic) grid runs its tasks on any domain, so the
    # placement that justified a DOMAIN scope no longer holds: escalate every
    # event it produces or consumes to DEVICE. Caught on MI300X: with DOMAIN
    # scope the consumer read stale L2 and produced zeros.
    for g in plan.graph.grids:
        if plan.modes[g.name] != "dynamic":
            continue
        for ev in list(g.in_edges) + list(g.out_edges):
            ep = plan.events[ev]
            need = m.effective_scope(Scope.DEVICE)
            if ep.scope < need:
                ep.scope = need
                plan.graph.events[ev].scope = need
                plan.say(f"P4: event {ev}: scope raised to {need.name} because {g.name} is globally scheduled")
    # Coherence of each event's endpoints (docs/qwen/RESULTS-2026-09-30.md: the acquire's whole-L2 invalidate is why the
    # megakernel's phases start cold). A "coherent" tile accesses its activations past the XCD caches, so it needs no
    # acquire invalidate; but a coherent consumer also bypasses a plain producer's L2, so such a producer must write
    # back (wbl2) on release even inside one XCD. The emitter reads these two bits per event. A "coherent_loads" tile is
    # a coherent consumer and a plain producer.
    for ev_name, ep in plan.events.items():
        prods = [g for g in plan.graph.grids if ev_name in g.out_edges]
        cons = [g for g in plan.graph.grids if ev_name in g.in_edges]
        ep.coherent_consumer = any(c.body.coherence in ("coherent", "coherent_loads") for c in cons)
        ep.producers_coherent = bool(prods) and all(_stores_coherent(pg.body) for pg in prods)
        if ep.coherent_consumer and not ep.producers_coherent:
            plan.say(f"P4: event {ev_name}: plain producer with a coherent consumer -> write-back release")
