import pytest

from etx.ir import Scope
from etx.machine import list_archs, load_machine
from etx.passes import PassOptions, compile_graph, verify_plan
from examples import gemm_rs, moe_layer, splitk_sum


@pytest.mark.parametrize("arch", list_archs())
def test_pipeline_runs_on_every_arch(arch):
    for mod in (splitk_sum, moe_layer):
        b = mod.bindings()
        plan = compile_graph(mod.build(), load_machine(arch), b, mod.runtime(b))
        assert verify_plan(plan) == []
        assert plan.tasks and plan.events
        assert all(m in ("static", "dynamic", "hybrid") for m in plan.modes.values())


def test_event_elimination_inlines_pure_norm():
    b = moe_layer.bindings()
    plan = compile_graph(moe_layer.build(), "gfx942", b, moe_layer.runtime(b))
    assert "E_norm" in plan.eliminated_events
    assert "norm" not in [g.name for g in plan.graph.grids]
    assert [n for n, _ in plan.graph.grid("qkv_proj").prologue] == ["norm"]
    assert "norm" in plan.inlined and plan.type_ids["norm"] == len(plan.graph.grids)
    off = compile_graph(moe_layer.build(), "gfx942", b, moe_layer.runtime(b), PassOptions(event_elimination=False))
    assert "E_norm" in off.events


def test_chiplet_scopes_are_lowered_per_machine():
    b = moe_layer.bindings()
    mi300 = compile_graph(moe_layer.build(), "gfx942", b, moe_layer.runtime(b))
    scopes = {e.scope for e in mi300.events.values()}
    assert Scope.DOMAIN in scopes or Scope.DEVICE in scopes
    h100 = compile_graph(moe_layer.build(), "sm_90", b, moe_layer.runtime(b))
    assert all(e.scope == Scope.DEVICE for e in h100.events.values())


def test_data_dependent_grids_never_static():
    b = moe_layer.bindings()
    plan = compile_graph(moe_layer.build(), "gfx942", b, moe_layer.runtime(b))
    for g in ("gather", "group_gemm", "down", "combine"):
        assert plan.modes[g] != "static", plan.reasons[g]


def test_cross_device_forces_static_and_system_scope():
    b = gemm_rs.bindings()
    plan = compile_graph(gemm_rs.build(2), "gfx90a", b, {})
    assert plan.n_devices == 2
    assert plan.events["E"].scope == Scope.SYSTEM and plan.events["E"].memory == "fine_grained"
    for d in range(2):
        assert plan.modes[f"reduce_scatter_d{d}"] == "static"
        assert "cross-device" in plan.reasons[f"reduce_scatter_d{d}"]
    assert len(plan.kernel_instances) == 2


def test_reasons_are_printed():
    b = moe_layer.bindings()
    plan = compile_graph(moe_layer.build(), "gfx942", b, moe_layer.runtime(b))
    assert all(plan.reasons[g] for g in plan.modes)
    assert any(line.startswith("P3: eliminate E_norm") for line in plan.log)
