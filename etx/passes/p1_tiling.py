"""Pass 1: tile sizing, resource classes, worker count.

Tile *shapes* are the frontend's business (the TileOp already fixed them);
this pass checks they fit the target and derives how many workgroups per CU
can be co-resident, which fixes the persistent grid size.  The register-union
problem (one kernel's occupancy is set by its heaviest tile) is handled by
grouping tiles into resource classes; splitting classes into separate kernel
instances is optional and off by default (design §09 Pass 1).
"""
from __future__ import annotations

import math

from .plan import Plan


def _wg_per_cu(plan: Plan, vgpr: int, agpr: int, lds_bytes: int, threads: int) -> int:
    m = plan.machine
    res = m.resources
    regs = max(1, vgpr + agpr)
    granule = res.get("reg_granule", 1)
    regs = math.ceil(regs / granule) * granule
    waves_per_wg = max(1, math.ceil(threads / m.wave_size))
    # regs_per_lane is register-file capacity per SIMD lane slot, not the
    # architectural per-thread limit. Round each warp's allocation first.
    waves_per_simd_by_regs = min(res["max_waves_per_simd"], res["regs_per_lane"] // regs)
    waves_per_cu = waves_per_simd_by_regs * res["simds_per_cu"]
    by_regs = waves_per_cu // waves_per_wg
    reserved = res.get("lds_reserved_per_wg_bytes", 256)
    by_lds = int(res["lds_kb"] * 1024 // (lds_bytes + reserved)) if lds_bytes + reserved else res["max_wg_per_cu"]
    return min(res["max_wg_per_cu"], by_regs, by_lds)


def run(plan: Plan) -> None:
    m = plan.machine
    classes: dict[str, list[str]] = {}
    per_grid_wg: dict[str, int] = {}
    for g in plan.graph.grids:
        r = g.resource
        if r.lds_bytes + r.prefetch_bytes > m.resources["lds_kb"] * 1024:
            raise ValueError(f"{g.name}: LDS need {r.lds_bytes + r.prefetch_bytes} B exceeds {m.resources['lds_kb']} KB on {m.name}")
        reg_max = m.resources.get("regs_per_thread_max", m.resources["regs_per_lane"])
        if r.vgpr + r.agpr > reg_max:
            raise ValueError(f"{g.name}: {r.vgpr}+{r.agpr} registers exceed {reg_max} on {m.name}")
        wg = _wg_per_cu(plan, r.vgpr, r.agpr, r.lds_bytes + r.prefetch_bytes, r.threads)
        if wg == 0:
            raise ValueError(f"{g.name}: resource demand cannot fit one workgroup on {m.name}")
        per_grid_wg[g.name] = wg
        key = f"rc{wg}"
        classes.setdefault(key, []).append(g.name)
        plan.resource_classes[g.name] = key
    plan.wg_per_cu = min(per_grid_wg.values()) if per_grid_wg else 1
    plan.workers_per_domain = m.cus_per_domain() * plan.wg_per_cu
    plan.n_domains = m.num_domains
    devices = sorted({g.device for g in plan.graph.grids}) or [0]
    plan.n_devices = len(devices)
    if plan.options.split_resource_classes and len(classes) > 1:
        for d in devices:
            for key, grids in classes.items():
                plan.kernel_instances.append({"device": d, "resource_class": key, "grids": grids})
        plan.say(f"P1: {len(classes)} resource classes split into {len(plan.kernel_instances)} kernel instances: {classes}")
    else:
        for d in devices:
            plan.kernel_instances.append({"device": d, "resource_class": "union", "grids": [g.name for g in plan.graph.grids if g.device == d]})
        heaviest = min(per_grid_wg, key=per_grid_wg.get) if per_grid_wg else "-"
        plan.say(f"P1: register/LDS union -> {plan.wg_per_cu} wg/CU set by {heaviest}; {plan.workers_per_domain} workers/domain x {plan.n_domains} domains x {plan.n_devices} device(s); classes={classes}")
