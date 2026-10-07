"""Byte floors and measured phase intervals for a decode step."""
import io
import json
from pathlib import Path

import pytest

from etx.tools import timeline as tl
from examples.vllm_llm import bw_spec

ROOT = Path(__file__).resolve().parents[1]
ROUND11 = Path("/Users/ckl/code/etx/results_2026-10-01/round11")
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


@pytest.mark.skipif(not (PLAN.is_file() and RAW.is_file()), reason="round11 archive is not present")
def test_round11_spec_and_ranked_total(tmp_path, capsys):
    plan = tl.load_plan(PLAN)
    rules = bw_spec.spec(plan, CONFIG, 1055)
    assert len(rules) == 339
    assert rules["^qkv_1$"] == (20_971_520 + 4096) / 256
    assert rules["^attn_0$"] == 2_160_640 / 20
    assert "^topk_0_x$" not in rules
    path = tmp_path / "bytes.json"
    assert bw_spec.main([str(PLAN), "--config", str(CONFIG), "--context", "1055",
                         "--output", str(path)]) == 0
    assert json.loads(path.read_text()) == rules
    tr = tl.build_trace(plan, tl.read_raw(RAW), spec=rules)
    r = tl.bandwidth_phases(tr, rules)
    assert r["ranked"][0]["phase"] == "qkv"
    assert r["ranked"][0]["wasted_us"] == pytest.approx(535.7, abs=0.1)
    assert r["known_bytes"] == 6_187_548_928
    assert r["span_us"] == pytest.approx(5081.49, abs=0.01)
    assert {p["phase"] for p in r["unknown"]} == {"topk", "reduce", "argmax"}
    assert tl.main([str(PLAN), str(RAW), "--bytes", str(path), "--bandwidth-report"]) == 0
    output = capsys.readouterr().out
    assert "qkv" in output and "TOTAL known 6.1875 GB; device span 5081.5 us" in output
