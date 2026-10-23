"""bw_spec and the timeline latency/bandwidth reports on the archived round-11 pair, through the external API."""
import json
import os
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
ROUND11 = Path(os.environ.get("ETX_DATA_ROOT", ROOT)) / "results_2026-10-01/round11"
PLAN = ROUND11 / "plan_r8base.json"
RAW = ROUND11 / "trace_r11_decode.bin"
CONFIG = ROOT / "examples/llm/configs/qwen3-30b-a3b.json"


def test_round11_spec_and_reports_from_the_commands(tmp_path):
    if not (PLAN.is_file() and RAW.is_file()):
        message = "Round11 acceptance needs ETX_DATA_ROOT with the archived pair"
        # Local acceptance and GPU runs set ETX_REQUIRE_DATA=1 so a lost archive cannot pass silently.
        if os.environ.get("ETX_REQUIRE_DATA") == "1":
            pytest.fail(message)
        pytest.skip(message)
    path = tmp_path / "bytes.json"
    spec = run_command([sys.executable, "-m", "examples.vllm_llm.bw_spec", PLAN, "--config", CONFIG, "--context", "1055",
                        "--argmax-write-next", "1", "--output", path], cwd=ROOT)
    assert spec.returncode == 0, spec.stdout
    rules = json.loads(path.read_text())
    assert len(rules) == 436
    assert rules["^qkv_1$"] == (20_971_520 + 4096) / 256 and rules["^argmax$"] == 303888
    latency = run_command([sys.executable, "-m", "etx.tools.timeline", PLAN, RAW, "--latency-report"], cwd=ROOT)
    assert latency.returncode == 0, latency.stdout
    bandwidth = run_command([sys.executable, "-m", "etx.tools.timeline", PLAN, RAW, "--bytes", path,
                             "--bandwidth-report"], cwd=ROOT)
    assert bandwidth.returncode == 0
    assert "qkv" in bandwidth.stdout and "TOTAL known 6.1904 GB; device span 5081.5 us" in bandwidth.stdout
