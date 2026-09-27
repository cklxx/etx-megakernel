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


def test_two_request_interleave_graph(monkeypatch):
    """Two requests: two workgroups per CU (76 workers per XCD without the relay), every task on its request's half of
    the slots (r * 38 + w), and each coupling mode simulates without deadlock."""
    from etx.codegen import emit_kernel
    from etx.passes import PassOptions
    for couple in ("router", "experts", "none"):
        monkeypatch.setenv("ETX_DSV2_COUPLE", couple)
        plan = compile_graph(dsv.build(layers=3, requests=2), "gfx942", {}, {}, PassOptions(relay="off"))
        assert plan.wg_per_cu == 2 and plan.workers_per_domain == 76
        for t in plan.tasks:
            r = 1 if t.grid.startswith("r1_") else 0
            assert r * 38 <= t.worker - t.domain * 76 < r * 38 + 38, (t.grid, t.worker)
        res = simulate(plan, seed=0)
        assert not res.deadlock, (couple, res.blocked[:3])
    src = emit_kernel(plan, 0)
    assert "#ifdef ETX_SLOT_BY_CU" in src and "etx_amdgcn_cu_key()" in src and "k < 2" in src
    # the cross-request edge of the default coupling: r1's layer 1 waits on r0's router of layer 1
    monkeypatch.setenv("ETX_DSV2_COUPLE", "router")
    g = dsv.build(layers=3, requests=2)
    assert "r0_E_norm_1" in g.grid("r1_qkv_1").in_edges and "r1_E_norm_0" in g.grid("r0_qkv_1").in_edges


def test_cli_compile_path(tmp_path, monkeypatch):
    """The build scripts compile through `python -m etx compile` (bindings() / runtime() of the example module), not
    through compile_graph: exercise that path, with an isolated block, so a broken example module fails here."""
    from etx.tools.cli import main
    monkeypatch.setenv("ETX_DSV2_ISOLATE", "0:qkv_5;0:gate_up_5+down_5")
    monkeypatch.setenv("ETX_DSV2_ISOLATE_REPS", "2")
    assert main(["compile", "examples/dsv2lite/graph.py", "--arch", "gfx942", "--out", str(tmp_path), "--inline-tiles"]) == 0
    assert (tmp_path / "megakernel_d0.hip").exists() and "iso1x1_r0_down_5" in (tmp_path / "plan.json").read_text()
