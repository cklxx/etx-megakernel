"""Check the raw trace against the host summary and test layer differences."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from etx.tools import layer_budget as lb
from etx.tools import timeline as tl


def require_round11(candidates):
    for base in candidates:
        directory = base / "results_2026-10-01/round11"
        if all((directory / name).is_file() for name in
               ("trace_r11_decode.bin", "trace_r11_decode.log")) and list(directory.glob("*.json")):
            return directory
    message = ("Round11 acceptance gate did not run: data set is missing. "
               "Set ETX_DATA_ROOT to its parent directory.")
    # Local acceptance and GPU runs set ETX_REQUIRE_DATA=1 so a lost archive cannot pass silently.
    if os.environ.get("ETX_REQUIRE_DATA") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def round11():
    root = Path(__file__).resolve().parents[1]
    candidates = ([Path(os.environ["ETX_DATA_ROOT"])] if "ETX_DATA_ROOT" in os.environ
                  else [root])
    directory = require_round11(candidates)
    rows = tl.read_raw(directory / "trace_r11_decode.bin")
    # Match task count and grid-instance count, then check every task in analyze.
    grid_count = len({r[0] for r in rows})
    matches = []
    for path in directory.glob("*.json"):
        plan = tl.load_plan(path)
        if (len(plan.get("descs", [])) == len(rows)
                and sum(len(k["grids"]) for k in plan.get("kernel_instances", [])) == grid_count):
            matches.append(path)
    assert len(matches) == 1
    assert len(rows) == 60362
    assert grid_count == 436  # The log header says 434, but its phase table has 436 rows.
    return directory, matches[0]


def test_round11_host_correspondence(round11, capsys, tmp_path):
    directory, plan = round11
    output = tmp_path / "budget.json"
    assert lb.main([str(plan), str(directory / "trace_r11_decode.bin"), "--json", str(output)]) == 0
    actual = capsys.readouterr().out
    expected = (directory / "trace_r11_decode.log").read_text()
    pattern = r"^  \S+\s+\d+\s+\d+\.\d{2}\s+\d+\.\d{2}\s+\d+\.\d\s+\d+\.\d\s+\d+\.\d$"
    expected_rows = re.findall(pattern, expected, re.M)
    actual_rows = re.findall(pattern, actual, re.M)
    assert len(expected_rows) == 436
    assert actual_rows == expected_rows
    for text in (expected, actual):
        assert re.search(r"phase trace \(us\): total span (\d+\.\d)", text)[1] == "5081.5"
    result = json.loads(output.read_text())["A"]
    assert result["critical_span"] == pytest.approx(result["span"], abs=0.001)
    assert result["layers"]["2"]["gap"] == pytest.approx(9.95)
    assert result["layers"]["2"]["dependency_gap"] == pytest.approx(9.95)


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


def test_compare_command(tmp_path, capsys):
    from array import array

    plan, rows = small_trace()
    plan_path = tmp_path / "plan.json"
    raw_path = tmp_path / "trace.bin"
    output = tmp_path / "comparison.json"
    plan_path.write_text(json.dumps(plan))
    raw_path.write_bytes(array("q", [v for row in rows for v in row]).tobytes())
    assert lb.main([str(plan_path), str(raw_path), "--compare", str(plan_path), str(raw_path),
                    "--json", str(output)]) == 0
    text = capsys.readouterr().out
    assert "A minus B" in text
    assert "0 0.00 0.00 0.00 0.00" in text
    assert all(p["excess"] == 0 for p in json.loads(output.read_text())["difference"])


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


def run_round11_gate(tmp_path, require):
    env = {k: v for k, v in os.environ.items() if k != "ETX_REQUIRE_DATA"}
    env["ETX_DATA_ROOT"] = str(tmp_path)
    if require:
        env["ETX_REQUIRE_DATA"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", f"{__file__}::test_round11_host_correspondence"],
        env=env, capture_output=True, text=True)


def test_missing_round11_is_failure_when_data_required(tmp_path):
    run = run_round11_gate(tmp_path, require=True)
    assert run.returncode != 0
    assert "acceptance gate did not run: data set is missing" in run.stdout
    assert "1 skipped" not in run.stdout


def test_missing_round11_is_skip_by_default(tmp_path):
    run = run_round11_gate(tmp_path, require=False)
    assert run.returncode == 0
    assert "acceptance gate did not run: data set is missing" in run.stdout
    assert "1 skipped" in run.stdout


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
