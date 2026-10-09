"""Byte floors and measured phase intervals for a decode step."""
import io
import json
import os
from pathlib import Path

import pytest

from etx.tools import timeline as tl
from examples.vllm_llm import bw_spec

ROOT = Path(__file__).resolve().parents[1]
ROUND11 = Path(os.environ.get("ETX_DATA_ROOT", ROOT)) / "results_2026-10-01/round11"
PLAN = ROUND11 / "plan_r8base.json"
RAW = ROUND11 / "trace_r11_decode.bin"
CONFIG = ROOT / "examples/llm/configs/qwen3-30b-a3b.json"


def test_phase_arithmetic_by_hand():
    tasks = [
        tl.Task(0, 0, "qkv_0", 0, 0, 0, 2, 4, 4_300_000),
        tl.Task(1, 1, "prep_0", 0, 0, 4, 10, 12, 8_600_000),
        tl.Task(2, 2, "reduce_0", 0, 0, 12, 18, 24),
    ]
    tr = tl.Trace(tasks, {t.id: t for t in tasks}, [t.name for t in tasks], 1, 1, None, 0, 24)
    spec = {"^qkv_0$": 4_300_000, "^prep_0$": 8_600_000}
    r = tl.bandwidth_phases(tr, spec)
    assert [p["phase"] for p in r["ranked"]] == ["qkv", "prep"]
    qkv, prep = r["ranked"]
    assert (qkv["bytes"], qkv["us"], qkv["floor_us"], qkv["wasted_us"]) == (4_300_000, 10, 1, 9)
    assert qkv["gb_s"] == pytest.approx(430)
    assert qkv["percent"] == pytest.approx(10)
    assert (prep["bytes"], prep["us"], prep["floor_us"], prep["wasted_us"]) == (8_600_000, 8, 2, 6)
    assert prep["gb_s"] == pytest.approx(1075)
    assert prep["percent"] == pytest.approx(25)
    assert r["unknown"][0]["phase"] == "reduce" and r["unknown"][0]["us"] == 6
    assert r["unknown"][0]["bytes"] is None
    assert (r["span_us"], r["known_bytes"], r["floor_us"], r["percent"]) == (24, 12_900_000, 3, 12.5)
    out = io.StringIO()
    tl.bandwidth_report(tr, spec, out=out)
    assert "reduce" in out.getvalue() and "unknown" in out.getvalue()
    assert "floor 3.0 us" in out.getvalue()


def test_rank_uses_wasted_time_not_byte_size():
    tasks = [
        tl.Task(0, 0, "a_0", 0, 0, 0, 0, 1, 4_300_000),
        tl.Task(1, 1, "b_0", 0, 0, 1, 6, 7, 8_600_000),
        tl.Task(2, 2, "a_1", 0, 0, 7, 9, 10, 4_300_000),
    ]
    tr = tl.Trace(tasks, {t.id: t for t in tasks}, [t.name for t in tasks], 1, 1, None, 0, 10)
    r = tl.bandwidth_phases(tr, {f"^{t.name}$": t.nbytes for t in tasks})
    assert [(p["phase"], p["us"], p["wasted_us"]) for p in r["ranked"]] == [
        ("a", 7, 5), ("b", 3, 1),
    ]


def test_round11_spec_and_ranked_total(tmp_path, capsys):
    if not (PLAN.is_file() and RAW.is_file()):
        message = "Round11 acceptance needs ETX_DATA_ROOT with the archived pair"
        # Local acceptance and GPU runs set ETX_REQUIRE_DATA=1 so a lost archive cannot pass silently.
        if os.environ.get("ETX_REQUIRE_DATA") == "1":
            pytest.fail(message)
        pytest.skip(message)
    plan = tl.load_plan(PLAN)
    rules = bw_spec.spec(plan, CONFIG, 1055, write_next=True)
    assert len(rules) == 436
    assert rules["^qkv_1$"] == (20_971_520 + 4096) / 256
    assert rules["^attn_0$"] == 2_160_640 / 20
    assert rules["^topk_0_x$"] == 352
    assert rules["^reduce_0$"] == 12608
    assert rules["^argmax$"] == 303888
    path = tmp_path / "bytes.json"
    assert bw_spec.main([str(PLAN), "--config", str(CONFIG), "--context", "1055",
                         "--argmax-write-next", "1", "--output", str(path)]) == 0
    assert json.loads(path.read_text()) == rules
    rows = tl.read_raw(RAW)
    assert len(rows) == len(plan["descs"]) == 60362
    assert all(row[0] == d["type"] and list(row[1:5]) == (d["coord"] + [0] * 4)[:4]
               and 0 < row[5] <= row[6] <= row[7] for row, d in zip(rows, plan["descs"]))
    tr = tl.build_trace(plan, rows, spec=rules)
    r = tl.bandwidth_phases(tr, rules)
    assert r["ranked"][0]["phase"] == "qkv"
    assert r["ranked"][0]["wasted_us"] == pytest.approx(535.7, abs=0.1)
    assert r["known_bytes"] == 6_190_408_720
    assert r["span_us"] == pytest.approx(5081.49, abs=0.01)
    assert r["unknown"] == []
    # Pin every displayed row, including all eight original excess values.
    expected = [
        ("qkv", 1006825472, 769.8, 1307.9, 30.4, 535.7),
        ("moe_up", 2415919104, 1051.1, 2298.4, 53.5, 489.3),
        ("moe_down", 1207959552, 748.9, 1612.9, 37.5, 468.0),
        ("router", 25362432, 438.9, 57.8, 1.3, 433.0),
        ("topk", 135168, 422.0, 0.3, 0.0, 422.0),
        ("reduce", 2420736, 397.4, 6.1, 0.1, 396.8),
        ("attn", 103710720, 353.4, 293.4, 6.8, 329.3),
        ("prep", 123136, 319.2, 0.4, 0.0, 319.1),
        ("o_proj", 805306368, 382.8, 2103.6, 48.9, 195.5),
        ("lm_head", 622333952, 171.0, 3639.8, 84.6, 26.3),
        ("prologue", 4096, 11.3, 0.4, 0.0, 11.3),
        ("argmax", 303888, 10.1, 30.1, 0.7, 10.0),
        ("in_norm", 4096, 5.6, 0.7, 0.0, 5.6),
    ]
    actual = [(p["phase"], round(p["bytes"]), *(round(p[k], 1) for k in
               ("us", "gb_s", "percent", "wasted_us"))) for p in r["ranked"]]
    assert actual == expected
    old_bytes = sum(p["bytes"] for p in r["ranked"] if p["phase"] not in {"topk", "reduce", "argmax"})
    assert old_bytes == 6_187_548_928
    latency = tl.latency_phases(tr)
    assert sum(p["cp_body_us"] + p["cp_dep_gap_us"] + p["cp_queue_gap_us"] for p in latency) == pytest.approx(tr.span)
    assert [(p["phase"], p["tasks"], round(p["active_xcds"], 2)) for p in latency
            if p["phase"] in {"router", "prep", "topk", "reduce", "attn", "o_proj"}] == [
        ("prep", 192, 3.68), ("attn", 960, 3.60), ("reduce", 192, 3.45),
        ("o_proj", 12288, 7.58), ("router", 12288, 4.30), ("topk", 384, 7.59)]
    assert tl.main([str(PLAN), str(RAW), "--latency-report"]) == 0
    assert tl.main([str(PLAN), str(RAW), "--bytes", str(path), "--bandwidth-report"]) == 0
    output = capsys.readouterr().out
    assert "qkv" in output and "TOTAL known 6.1904 GB; device span 5081.5 us" in output


def payload_fixture():
    types = {
        "topk_0_x": {"id": 0, "symbol": "etx_tile_rep_dsl_topk"},
        "attn_0": {"id": 1, "symbol": "etx_tile_attn_mfma16_g2"},
        "reduce_0": {"id": 2, "symbol": "etx_tile_attn_reduce"},
        "argmax": {"id": 3, "symbol": "vl_argmax"},
    }
    return {"types": types, "descs": [{"type": i} for i, n in enumerate((2, 4, 2, 1)) for _ in range(n)]}


def test_payload_arithmetic_by_hand():
    plan = payload_fixture()
    cfg = dict(E=4, TOPK=2, NH=4, NKV=2, HD=8, V=10)
    # Two replicas: 2 * (8 + 24). Two partitions: 4 * (2 * (16 + 8) + 16).
    assert bw_spec.payload_bytes(plan, cfg, 257, True) == {
        "topk_0_x": 64, "reduce_0": 256, "argmax": 36}
    assert bw_spec.payload_bytes(plan, cfg, 256, False) == {
        "topk_0_x": 64, "reduce_0": 160, "argmax": 32}
    with pytest.raises(bw_spec.bl.Refusal, match="write_next runtime flag"):
        bw_spec.payload_bytes(plan, cfg, 257, None)
    with pytest.raises(bw_spec.bl.Refusal, match="cannot cover context"):
        bw_spec.payload_bytes(plan, cfg, 513, True)
    plan["types"]["topk_0_x"]["symbol"] = "unknown"
    with pytest.raises(bw_spec.bl.Refusal, match="payload layout"):
        bw_spec.payload_bytes(plan, cfg, 257, True)


def test_latency_arithmetic_by_hand():
    tasks = [
        tl.Task(0, 0, "a_0", 0, 0, 0, 0, 4),
        tl.Task(1, 0, "a_0", 0, 1, 0, 2, 6),
        tl.Task(2, 0, "a_0", 1, 2, 0, 1, 3),
        tl.Task(3, 1, "b_0", 1, 2, 3, 8, 10),
    ]
    tr = tl.Trace(tasks, {t.id: t for t in tasks}, ["a_0", "b_0"], 2, 2,
                  [[], [], [], [0, 1, 2]], 0, 10)
    a, b = tl.latency_phases(tr)
    assert a["body_us"] == pytest.approx(10 / 3)
    assert a["wait_us"] == 1
    assert a["active_tasks"] == pytest.approx(10 / 6)
    # Domain zero has overlapping bodies: its union is six, not eight.
    assert a["active_xcds"] == pytest.approx(8 / 6)
    assert a["start_spread_us"] == 2
    assert a["finish_skew_us"] == 3
    assert a["outgoing_gap_us"] == 2
    assert b["outgoing_gap_us"] == 0
    assert b["wait_us"] == 5  # Includes predecessor work; this is not exposed overhead.
    assert b["dep_gap_mean_us"] == b["dep_gap_p95_us"] == b["dep_gap_max_us"] == 2
    assert b["cp_body_us"] == b["cp_dep_gap_us"] == 2
    assert sum(r["cp_body_us"] + r["cp_dep_gap_us"] + r["cp_queue_gap_us"] for r in (a, b)) == 10


def test_latency_refuses_partial_trace():
    task = tl.Task(0, 0, "a_0", 0, 0, 0, 0, 1)
    tr = tl.Trace([task], {0: task}, ["a_0"], 1, 1, [[], [0]], 0, 1)
    with pytest.raises(ValueError, match="full, complete trace"):
        tl.latency_phases(tr)
    tr.deps = None
    with pytest.raises(ValueError, match="dependency lists"):
        tl.latency_phases(tr)
