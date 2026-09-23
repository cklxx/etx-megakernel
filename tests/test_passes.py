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


def test_data_dependent_grids_static_only_with_barrier():
    """A grid with runtime edge maps may be static only if it first waits on the
    writer of the runtime tensors (otherwise it evaluates the map on stale data)."""
    b = moe_layer.bindings()
    plan = compile_graph(moe_layer.build(), "gfx942", b, moe_layer.runtime(b))
    for name in ("gather", "group_gemm", "down", "combine"):
        g = plan.graph.grid(name)
        if plan.modes[name] == "static":
            first = next(iter(g.in_edges.values()))
            assert first.kind == "all", f"{name}: static with runtime maps must start with a '*' barrier wait"
            assert "E_group" in g.in_edges or "E_route" in g.in_edges
    forced = compile_graph(moe_layer.build(), "gfx942", b, moe_layer.runtime(b), PassOptions(force_mode="static"))
    assert forced.graph.grid("group_gemm").in_edges["E_group"].kind == "all"


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


def test_relay_decision_dsv2lite():
    """P5 turns the per-domain relay on when hundreds of workers poll one DEVICE word."""
    from examples.dsv2lite import graph as G
    from etx.passes import compile_graph
    from etx.passes.plan import PassOptions
    plan = compile_graph(G.build(layers=2), "gfx942", {})
    assert plan.relay and plan.workers_per_domain == 37
    assert plan.relay_words[0], "mirrored words"
    off = compile_graph(G.build(layers=2), "gfx942", {}, options=PassOptions(relay="off"))
    assert not off.relay and off.workers_per_domain == 38


def test_relay_off_for_small_fanout():
    from etx.passes import compile_graph
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("sk", pathlib.Path(__file__).parent.parent / "examples" / "splitk_sum.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    plan = compile_graph(m.build(), "gfx942", m.bindings())
    assert not plan.relay
