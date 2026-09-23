"""The narrow starting point: fleet-mi300x's DeepSeek-V2-Lite decode graph in ETX.
Checks the graph verifies, places every Chiplet-task on its XCD, keeps fleet's
event-scope structure (2 global + 6 local event tensors per MoE layer), and that
the simulator's prediction with fleet's measured task durations lands near
fleet's measured 3.60 ms/token."""
from etx.ir import Scope
from etx.passes import compile_graph, verify_plan
from etx.sim import simulate
from examples.dsv2lite import graph as dsv


def test_graph_shape_matches_fleet():
    g = dsv.build()
    plan = compile_graph(g, "gfx942", {}, {})
    assert verify_plan(plan) == []
    # fleet: 1,791 logical tasks / token but one descriptor per worker for Chiplet-tasks;
    # ETX counts descriptors: 27 layers x ~2,000 + embed/lm_head/argmax
    assert 50_000 < len(plan.tasks) < 60_000
    # every (xcd, worker) task sits on its XCD
    for t in plan.tasks:
        if t.grid.startswith(("qkv_", "oproj_", "router_", "gate_up_", "down_")):
            assert t.domain == t.coord[0]
    # per MoE layer: o_proj and down are the only cross-XCD (DEVICE) events
    ev = plan.events
    assert ev["E_oproj_5"].scope == Scope.DEVICE and ev["E_down_5"].scope == Scope.DEVICE
    for name in ("E_qkv_5", "E_qabs_5", "E_attn_5", "E_merge_5", "E_norm_5", "E_gu_5"):
        assert ev[name].scope == Scope.DOMAIN, name


def test_simulator_prediction_is_in_fleets_range():
    plan = compile_graph(dsv.build(layers=3), "gfx942", {}, {})
    r = simulate(plan, seed=0)
    assert not r.deadlock, r.blocked
    per_layer = r.makespan_us / 3
    # fleet measures 133 us per MoE layer (3.6 ms / 27); the model should be the same order
    assert 80 < per_layer < 300, per_layer
