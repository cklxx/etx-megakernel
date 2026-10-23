"""The tools/ds_batch_attribution.py command, moved from tests/test_ds_batch_attribution_tool.py."""
import json
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def family(name, body, dep, queue, tasks):
    return {
        "family": name,
        "cp_body_us": body,
        "cp_dep_gap_us": dep,
        "cp_queue_gap_us": queue,
        "tasks": tasks,
        "body_work_us": 0.0,
        "wait_work_us": 0.0,
    }


def row(batch, round_, step_ms, families):
    return {
        "batch": batch,
        "round": round_,
        "gate": {"step_ms": str(step_ms)},
        "trace": {"families": families},
    }


def synthetic():
    """B1, B2 and B4; the round-2 B1 row is first to prove round selection."""
    return {
        "results": [
            row(1, 2, 9.999, [family("decoy", 9999, 999, 999, 0)]),
            row(1, 1, 2.0, [
                family("a", 1000, 100, 0, 10),
                family("b", 500, 50, 0, 5),
                family("tiny", 0, 0, 0, 0),
            ]),
            row(2, 1, 4.0, [
                family("a", 2100, 150, 20, 20),
                family("b", 900, 50, 30, 10),
                family("tiny", 10, 5, 5, 0),
            ]),
            row(4, 1, 9.0, [
                family("a", 4300, 250, 40, 40),
                family("b", 1900, 50, 60, 20),
                family("tiny", 20, 5, 5, 0),
            ]),
        ]
    }


def test_main_prints_steps_deltas_and_filters(tmp_path):
    path = tmp_path / "trace_report.json"
    path.write_text(json.dumps(synthetic()))
    run = run_command([sys.executable, "-m", "tools.ds_batch_attribution", str(path)], cwd=ROOT)
    assert run.returncode == 0, run.stdout
    out = run.stdout
    assert "batch  step_us   tok/s     body     dep   queue   tasks" in out
    assert "    1     2000     500     1500   150.0     0.0      15" in out
    assert "    4     9000     444     6220   305.0   105.0      60" in out
    assert "      delta step +2000 us: body +1510, dep +55.0, queue +55.0" in out
    assert "      delta step +5000 us: body +3210, dep +100.0, queue +50.0" in out
    assert "body +1100" in out and "body +2200" in out
    assert "tiny" not in out  # every tiny move is under the reporting threshold
