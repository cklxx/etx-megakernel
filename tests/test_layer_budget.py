"""Check the raw trace against the host summary and test layer differences."""
from pathlib import Path

import pytest

from etx.tools import layer_budget as lb
from etx.tools import timeline as tl


def small_trace(offset=0):
    names = ["attn_0", "reduce_0", "attn_1", "reduce_1"]
    plan = dict(types={n: {"id": i} for i, n in enumerate(names)},
                descs=[dict(id=i, type=i, coord=[0], domain=0) for i in range(4)],
                domains=1, workers_per_domain=2, task_in=[[], [0], [1], [2]],
                ev_inst=[[0, 0, [0]], [0, 0, [1]], [0, 0, [2]]])
    # The negative gap tests overlap. The edge to layer 1 belongs to layer 0.
    rows = [(i, 0, 0, 0, 0, tk + offset, rd + offset, dn + offset, w)
            for i, (tk, rd, dn, w) in enumerate([(100, 110, 130, 0), (101, 125, 140, 1),
                                                 (141, 150, 160, 0), (161, 170, 180, 1)])]
    return plan, rows


def test_gap_wait_and_comparison():
    plan, rows = small_trace()
    a = lb.analyze(plan, rows)
    assert a["layers"][0]["gap"] == pytest.approx(0.05)
    assert [e["gap"] for e in a["layers"][0]["edges"]] == pytest.approx([-0.05, 0.10])
    assert a["layers"][0]["worker_wait"] == pytest.approx({"0": 0.10, "1": 0.24})
    assert a["layers"][0]["wait_per_worker"] == pytest.approx(0.17)
    b = lb.analyze(*small_trace(10**15))
    assert b["phases"] == a["phases"]
    delta = lb.compare(a, b, r"^(attn|reduce)_")
    assert all(d["excess"] == 0 and d["wait_per_worker"] == 0 for d in delta)
    b["layers"][0]["gap"] -= 1
    b["phases"][0]["body"] -= 2
    delta = lb.compare(a, b, r"^(attn|reduce)_")
    assert delta[0]["excess"] == pytest.approx(3)


def test_reject_wrong_plan():
    plan, rows = small_trace()
    with pytest.raises(ValueError, match="task count"):
        lb.analyze(plan, rows[:-1])
    plan["descs"][0]["coord"] = [1]
    with pytest.raises(ValueError, match="does not match"):
        lb.analyze(plan, rows)


def test_timeline_preserves_tick_differences():
    a = tl.build_trace(*small_trace())
    b = tl.build_trace(*small_trace(10**15))
    assert a.span == b.span
    assert [(t.taken, t.ready, t.done) for t in a.tasks] == [
        (t.taken, t.ready, t.done) for t in b.tasks]


@pytest.mark.parametrize("field", [5, 6, 7, "all", "missing"])
def test_reject_incomplete_trace(field):
    plan, rows = small_trace()
    if field == "missing":
        rows.pop()
    else:
        row = list(rows[-1])
        if field == "all":
            row[5:8] = [0, 0, 0]
        else:
            row[field] = 0
        rows[-1] = tuple(row)
    with pytest.raises(ValueError, match="trace is not complete"):
        lb.analyze(plan, rows)


# The layer_budget command on the round-11 pair, --compare and the data gate are in tests/e2e/test_api_layer_budget.py.


def test_metric_limits(capsys):
    result = lb.analyze(*small_trace())
    lb.report(result, "test")
    text = capsys.readouterr().out
    assert "Type-id order is not dependency order" in text
    assert "Negative overlaps can cancel positive gaps" in text
    assert "Do not use either sum alone for the 8.3 us rental gate" in text
    assert "dependency_gap_us" in text
    assert "metric_limits" in result
    doc = (Path(__file__).resolve().parents[1] / "docs/qwen/DESIGN-acquire-2026-10-05.md").read_text()
    for sentence in result["metric_limits"].split(". "):
        assert sentence in doc


def test_gap_arithmetic_with_parallel_phases():
    plan, _ = small_trace()
    # Type order: A, C, B, D. Dependencies: A -> B -> D and A -> C -> D.
    # A: [10, 20], B: [25, 40], C: [50, 60], D: [70, 80] us.
    # Type order: 30 - 35 + 30 = 25 us.
    # Dependency order uses type id for ties: A, C, B, D. It also gives 25 us.
    # Add B -> C. Dependency order is now A, B, C, D: 5 + 10 + 10 = 25 us.
    # Per-layer sums differ: type order gives -5 and 30, dependency order gives 15 and 10.
    plan["task_in"] = [[], [0, 1], [0], [1, 2]]
    plan["ev_inst"] = [[0, 0, [0]], [0, 0, [2]], [0, 0, [1]]]
    rows = [(i, 0, 0, 0, 0, ready, ready, done, i % 2)
            for i, (ready, done) in enumerate([(10, 20), (50, 60), (25, 40), (70, 80)])]
    # Without B -> C, B and C are parallel. The signed sum hides the 30 us gap.
    plan["task_in"][1] = [0]
    parallel = lb.analyze(plan, rows, tick_ns=1000)
    assert parallel["layers"][0]["gap"] == pytest.approx(-5)
    assert parallel["layers"][0]["dependency_gap"] == pytest.approx(-5)
    assert [e["gap"] for e in parallel["layers"][0]["edges"]] == pytest.approx([30, -35])
    plan["task_in"][1] = [0, 1]
    result = lb.analyze(plan, rows, tick_ns=1000)
    assert [result["layers"][i]["gap"] for i in (0, 1)] == pytest.approx([-5, 30])
    assert [result["layers"][i]["dependency_gap"] for i in (0, 1)] == pytest.approx([15, 10])
    assert result["dependency_order"] == ["attn_0", "attn_1", "reduce_0", "reduce_1"]


def test_dependency_order_absent(capsys):
    plan, rows = small_trace()
    del plan["task_in"]
    result = lb.analyze(plan, rows)
    assert result["dependency_order"] is None
    assert all(p["dependency_gap"] is None for p in result["layers"].values())
    lb.report(result, "test")
    assert "unavailable" in capsys.readouterr().out


def test_dependency_phase_cycle(capsys):
    plan, rows = small_trace()
    plan["task_in"][0] = [2]
    result = lb.analyze(plan, rows)
    assert result["dependency_order"] is None
    lb.report(result, "test")
    assert "phase dependencies have a cycle" in capsys.readouterr().out
