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

from ..ir.types import Scope
from .plan import Plan

MARGIN = 1.5     # bias toward static: the paper's regular workloads lose 6-17% under dynamic scheduling


def run(plan: Plan) -> None:
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
        elif g.has_runtime_edges:
            mode = "hybrid" if cross_dom else "dynamic"
            why = f"data-dependent edges (static would degrade to E[0]); cross-domain edges={cross_dom}"
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
