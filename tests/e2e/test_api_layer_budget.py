"""Command-level cases moved from tests/test_layer_budget.py, through external APIs only."""
from array import array
import json
import os
from pathlib import Path
import re
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def layer_budget(*argv):
    return run_command([sys.executable, "-m", "etx.tools.layer_budget", *map(str, argv)], cwd=ROOT)


@pytest.fixture(scope="module")
def round11():
    directory = Path(os.environ.get("ETX_DATA_ROOT", ROOT)) / "results_2026-10-01/round11"
    if not (all((directory / name).is_file() for name in ("trace_r11_decode.bin", "trace_r11_decode.log"))
            and list(directory.glob("*.json"))):
        message = ("Round11 acceptance gate did not run: data set is missing. "
                   "Set ETX_DATA_ROOT to its parent directory.")
        # Local acceptance and GPU runs set ETX_REQUIRE_DATA=1 so a lost archive cannot pass silently.
        if os.environ.get("ETX_REQUIRE_DATA") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return directory


def test_round11_host_correspondence(round11, tmp_path):
    raw = round11 / "trace_r11_decode.bin"
    runs = {plan: layer_budget(plan, raw, "--json", tmp_path / f"{plan.stem}.json")
            for plan in sorted(round11.glob("*.json"))}
    matches = [plan for plan, run in runs.items() if run.returncode == 0]
    assert len(matches) == 1, {str(p): r.stdout[-400:] for p, r in runs.items()}
    actual = runs[matches[0]].stdout
    expected = (round11 / "trace_r11_decode.log").read_text()
    pattern = r"^  \S+\s+\d+\s+\d+\.\d{2}\s+\d+\.\d{2}\s+\d+\.\d\s+\d+\.\d\s+\d+\.\d$"
    expected_rows = re.findall(pattern, expected, re.M)
    assert len(expected_rows) == 436  # The log header says 434, but its phase table has 436 rows.
    assert re.findall(pattern, actual, re.M) == expected_rows
    for text in (expected, actual):
        assert re.search(r"phase trace \(us\): total span (\d+\.\d)", text)[1] == "5081.5"
    result = json.loads((tmp_path / f"{matches[0].stem}.json").read_text())["A"]
    assert result["critical_span"] == pytest.approx(result["span"], abs=0.001)
    assert result["layers"]["2"]["gap"] == pytest.approx(9.95)
    assert result["layers"]["2"]["dependency_gap"] == pytest.approx(9.95)


def test_compare_command(tmp_path):
    names = ["attn_0", "reduce_0", "attn_1", "reduce_1"]
    plan = dict(types={n: {"id": i} for i, n in enumerate(names)},
                descs=[dict(id=i, type=i, coord=[0], domain=0) for i in range(4)],
                domains=1, workers_per_domain=2, task_in=[[], [0], [1], [2]],
                ev_inst=[[0, 0, [0]], [0, 0, [1]], [0, 0, [2]]])
    rows = [(i, 0, 0, 0, 0, tk, rd, dn, w) for i, (tk, rd, dn, w) in enumerate(
        [(100, 110, 130, 0), (101, 125, 140, 1), (141, 150, 160, 0), (161, 170, 180, 1)])]
    plan_path, raw_path, output = tmp_path / "plan.json", tmp_path / "trace.bin", tmp_path / "comparison.json"
    plan_path.write_text(json.dumps(plan))
    raw_path.write_bytes(array("q", [v for row in rows for v in row]).tobytes())
    run = layer_budget(plan_path, raw_path, "--compare", plan_path, raw_path, "--json", output)
    assert run.returncode == 0, run.stdout
    assert "A minus B" in run.stdout
    assert "0 0.00 0.00 0.00 0.00" in run.stdout
    assert all(p["excess"] == 0 for p in json.loads(output.read_text())["difference"])


def run_round11_gate(tmp_path, require):
    env = {k: v for k, v in os.environ.items() if k != "ETX_REQUIRE_DATA"}
    env["ETX_DATA_ROOT"] = str(tmp_path)
    if require:
        env["ETX_REQUIRE_DATA"] = "1"
    return run_command([sys.executable, "-m", "pytest", "-q", "-rs", "-o", "addopts=",
                        f"{__file__}::test_round11_host_correspondence"], cwd=ROOT, env=env)


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
