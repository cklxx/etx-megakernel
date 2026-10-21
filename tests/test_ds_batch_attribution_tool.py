"""Pin the arithmetic of tools/ds_batch_attribution.py on a synthetic trace."""
import json
from pathlib import Path

from tools import ds_batch_attribution as attr


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


def test_first_rounds_selects_the_lowest_round_not_file_order():
    rows = attr.first_rounds(synthetic())
    assert [r["batch"] for r in rows] == [1, 2, 4]
    assert [r["round"] for r in rows] == [1, 1, 1]
    assert attr.step_us(rows[0]) == 2000.0
    assert attr.totals(rows[0]) == {"body": 1500.0, "dep": 150.0, "queue": 0.0, "tasks": 15}


def test_summary_totals_throughput_and_deltas():
    summary = attr.summarize(synthetic())
    assert [s["step_us"] for s in summary] == [2000.0, 4000.0, 9000.0]
    assert [s["tokens_per_s"] for s in summary] == [500.0, 500.0, 4000.0 / 9.0]
    assert [(s["body"], s["dep"], s["queue"], s["tasks"]) for s in summary] == [
        (1500.0, 150.0, 0.0, 15),
        (3010.0, 205.0, 55.0, 30),
        (6220.0, 305.0, 105.0, 60),
    ]
    assert summary[0]["delta"] is None and summary[0]["families"] == []
    assert summary[1]["delta"] == {
        "step_us": 2000.0, "body": 1510.0, "dep": 55.0, "queue": 55.0}
    assert summary[2]["delta"] == {
        "step_us": 5000.0, "body": 3210.0, "dep": 100.0, "queue": 50.0}


def test_family_deltas_keep_material_moves_and_skip_small_ones():
    summary = attr.summarize(synthetic())
    assert summary[1]["families"] == [
        {"family": "a", "body": 1100.0, "dep": 50.0, "queue": 20.0},
        {"family": "b", "body": 400.0, "dep": 0.0, "queue": 30.0},
    ]
    assert summary[2]["families"] == [
        {"family": "a", "body": 2200.0, "dep": 100.0, "queue": 20.0},
        {"family": "b", "body": 1000.0, "dep": 0.0, "queue": 30.0},
    ]


def test_family_absent_from_previous_batch_is_not_compared():
    assert attr.family_deltas([], [family("new", 500, 50, 50, 1)]) == []


def test_main_prints_steps_deltas_and_filters(tmp_path, capsys):
    path = tmp_path / "trace_report.json"
    path.write_text(json.dumps(synthetic()))
    assert attr.main(str(path)) == 0
    out = capsys.readouterr().out
    assert "batch  step_us   tok/s     body     dep   queue   tasks" in out
    assert "    1     2000     500     1500   150.0     0.0      15" in out
    assert "    4     9000     444     6220   305.0   105.0      60" in out
    assert "      delta step +2000 us: body +1510, dep +55.0, queue +55.0" in out
    assert "      delta step +5000 us: body +3210, dep +100.0, queue +50.0" in out
    assert "body +1100" in out and "body +2200" in out
    assert "tiny" not in out  # every tiny move is under the reporting threshold


def test_real_receipt_matches_the_published_b1_b2_b4_numbers():
    path = (Path(__file__).resolve().parents[1] /
            "examples/gpu-live/results/2026-10-16-multibatch/trace_report.json")
    summary = attr.summarize(attr.load_report(path))
    assert [s["batch"] for s in summary] == [1, 2, 4]
    assert [round(s["step_us"]) for s in summary] == [3254, 4576, 9027]
    assert [round(s["tokens_per_s"]) for s in summary] == [307, 437, 443]
    assert [(round(s["body"]), round(s["dep"], 1), round(s["queue"], 1), s["tasks"])
            for s in summary] == [
        (2716, 378.7, 0.0, 53022),
        (3735, 542.2, 69.7, 109892),
        (7360, 955.1, 381.6, 219784),
    ]
    delta = summary[2]["delta"]
    assert (round(delta["step_us"]), round(delta["body"]),
            round(delta["dep"], 1), round(delta["queue"], 1)) == (4451, 3625, 412.9, 312.0)
