"""etx.tools.timeline on a simulated DeepSeek-V2-Lite schedule: the critical path must telescope to the span,
the per-XCD shares must add up, and the byte spec must reproduce the model's per-layer bytes."""
import io
import json

import pytest

from etx.passes import compile_graph
from etx.sim import simulate
from etx.tools import timeline as tl
from examples.dsv2lite import bytes as dsb
from examples.dsv2lite import graph as dsv


@pytest.fixture(scope="module")
def sim_trace():
    plan = compile_graph(dsv.build(layers=3), "gfx942", {}, {})
    res = simulate(plan, seed=0)
    assert not res.deadlock
    pj = tl.plan_json_from_obj(plan)
    rows = tl.trace_from_sim(plan, res)
    return plan, res, pj, rows


def test_plan_json_carries_dependencies(sim_trace):
    plan, _, pj, _ = sim_trace
    assert len(pj["task_in"]) == len(plan.tasks)
    deps = tl._deps_from_plan(pj)
    # every down_2 task waits on its XCD's gate_up_2 event: 37 producers on the same XCD
    down = next(t for t in plan.tasks if t.grid == "down_2")
    prods = [plan.tasks[p] for p in deps[down.id]]
    assert len(prods) == 37 and all(p.grid == "gate_up_2" and p.domain == down.domain for p in prods)


def test_critical_path_telescopes_to_span(sim_trace):
    _, res, pj, rows = sim_trace
    tr = tl.build_trace(pj, rows, spec=dsb.spec())
    steps = tl.critical_path(tr)
    total = sum(s["body"] + s["gap"] for s in steps)
    assert total == pytest.approx(tr.span, rel=1e-3)
    assert tr.span == pytest.approx(res.makespan_us, rel=1e-2)
    kinds = {s["kind"] for s in steps}
    assert "dep" in kinds
    # the path runs through every phase of a MoE layer
    names = {s["task"].name for s in steps}
    for g in ("qkv_2", "attn_2", "merge_2", "oproj_2", "router_2", "gate_up_2", "down_2"):
        assert g in names, g


def test_window_domain_shares_and_bytes(sim_trace):
    _, _, pj, rows = sim_trace
    tr = tl.build_trace(pj, rows, grids=r"_2$", spec=dsb.spec())
    for d in tl.per_domain(tr):
        assert d["body"] + d["wait"] + d["idle"] == pytest.approx(1.0, abs=1e-6)
    per = {a["type"]: a["bytes"] for a in tl.per_type(tr)}
    want = dsb.per_layer(1024)
    for k in ("qkv", "oproj", "gate_up", "down"):
        assert per[f"{k}_2"] == pytest.approx(want[k], rel=1e-6)
    layer = sum(a["bytes"] for a in tl.per_type(tr) if a["type"].endswith("_2"))
    assert layer == pytest.approx(sum(want[k] for k in ("qkv", "qabs", "attn", "merge", "oproj", "router", "gate_up", "down")), rel=1e-6)
    md = tl.memory_demand(tr)
    assert layer / 4.3e6 * 0.99 <= md["floor_us"] <= tr.span        # neighbours' bodies overlapping the window add bytes
    out = io.StringIO()
    res = tl.report(tr, out=out)
    assert "critical path" in out.getvalue() and res["critical_span"] > 0


def test_memory_demand_groups_replicated_grids():
    ts = [tl.Task(i, 0, f"topk_{i}_x", 0, 0, 2.0 * i, 2.0 * i, 2.0 * i + 1.0) for i in range(3)]
    tr = tl.Trace(ts, {t.id: t for t in ts}, ["topk"], 1, 1, None, 0.0, 6.0)
    assert dict(tl.memory_demand(tr)["low_by_kind"])["topk"] == pytest.approx(3.0)


def test_raw_roundtrip_and_html(sim_trace, tmp_path):
    _, _, pj, rows = sim_trace
    raw = tmp_path / "t.bin"
    from array import array
    a = array("q", [v for r in rows for v in r]); raw.write_bytes(a.tobytes())
    (tmp_path / "plan.json").write_text(json.dumps(pj))
    (tmp_path / "b.json").write_text(json.dumps(dsb.spec()))
    assert tl.main([str(tmp_path / "plan.json"), str(raw), "--bytes", str(tmp_path / "b.json"), "--grids", "_1$",
                    "--html", str(tmp_path / "t.html"), "--json", str(tmp_path / "t.json")]) == 0
    page = (tmp_path / "t.html").read_text()
    assert page.startswith("<title>") and "gate_up_1" in page
    assert json.loads((tmp_path / "t.json").read_text())["critical_span"] > 0
