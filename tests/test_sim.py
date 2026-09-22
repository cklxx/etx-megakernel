import pytest

from etx.passes import PassOptions, compile_graph
from etx.sim import simulate
from examples import moe_layer, splitk_sum


def _plan(mod, arch="gfx942", bindings=None, **opts):
    b = mod.bindings(**(bindings or {}))
    return compile_graph(mod.build(), arch, b, mod.runtime(b), PassOptions(**opts))


def test_no_deadlock_and_all_tasks_run():
    for mod in (splitk_sum, moe_layer):
        plan = _plan(mod)
        r = simulate(plan)
        assert not r.deadlock, r.blocked
        assert sum(1 for s in r.segments if s.kind == "task") == len(plan.tasks)
        assert r.makespan_us > 0


@pytest.mark.parametrize("mode", ["static", "dynamic", "hybrid"])
def test_forced_modes_complete(mode):
    plan = _plan(moe_layer, force_mode=mode)
    r = simulate(plan)
    assert not r.deadlock, r.blocked


def test_static_deadlock_is_detected():
    plan = _plan(splitk_sum, bindings={"n": 1024}, force_mode="static")
    # corrupt one worker queue: put a consumer in front of one of its producers on the same worker
    key, q = next((k, q) for k, q in plan.static_queues.items() if any(plan.tasks[t].grid == "partial_sum" for t in q))
    prod = next(t for t in q if plan.tasks[t].grid == "partial_sum")
    cons_row = plan.tasks[prod].coord[0]
    cons = next(t.id for t in plan.tasks if t.grid == "final_sum" and t.coord[0] == cons_row)
    for k, qq in plan.static_queues.items():
        if cons in qq:
            qq.remove(cons)
    q.insert(q.index(prod), cons)
    r = simulate(plan)
    assert r.deadlock and any("final_sum" in b for b in r.blocked)


def test_critical_path_budget_is_reported():
    plan = _plan(moe_layer)
    r = simulate(plan)
    assert r.busy_us > 0 and r.n_workers == plan.workers_per_device()
    assert set(r.per_grid_us) <= {g.name for g in plan.graph.grids}
