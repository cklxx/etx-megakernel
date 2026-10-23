"""Timeline and XCD-account commands on a simulated DeepSeek-V2-Lite schedule, through the external API."""
import json
from pathlib import Path
import sys

import pytest
from etx.testing import run_command, simulated_trace_files

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trace(tmp_path_factory):
    return simulated_trace_files(tmp_path_factory.mktemp("trace"))


def test_timeline_raw_roundtrip_and_html(trace, tmp_path):
    paths, _ = trace
    run = run_command([sys.executable, "-m", "etx.tools.timeline", paths["plan"], paths["raw"], "--bytes", paths["bytes"],
                       "--grids", "_1$", "--html", tmp_path / "t.html", "--json", tmp_path / "t.json"], cwd=ROOT)
    assert run.returncode == 0, run.stdout
    page = (tmp_path / "t.html").read_text()
    assert page.startswith("<title>") and "gate_up_1" in page
    assert json.loads((tmp_path / "t.json").read_text())["critical_span"] > 0


def test_xcd_account_writes_outputs(trace, tmp_path):
    paths, domains = trace
    out = tmp_path / "acct"
    run = run_command([sys.executable, "-m", "examples.vllm_llm.xcd_account", paths["plan"], paths["raw"],
                       "--bytes", paths["bytes"], "--out", out, "--layer", "1"], cwd=ROOT)
    assert run.returncode == 0, run.stdout
    assert "critical path" in (out / "account.txt").read_text()
    assert json.loads((out / "account.json").read_text())["workers"] > 0
    assert (out / "xcd_layer1.svg").read_text().startswith("<svg")
    assert set(json.loads((out / "xcd_series_step.json").read_text())["xcd"]) == {str(d) for d in range(domains)}
