"""The experiment result check: key extraction and derived-file reruns."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_results  # noqa: E402
import extract_key  # noqa: E402

REGISTRY = ROOT / "examples/gpu-live/results/registry.json"


def test_committed_results_reproduce():
    assert check_results.main(["--registry", str(REGISTRY)]) == 0


def test_pattern_registry_kinds():
    cases = {
        "remark:     VGPRs Spill: 5 [-Rpass-analysis=kernel-resource-usage]": "resource",
        "[16:22:55 + 13m09s] fail    captures rc=1 16s": "exit",
        "rsync(54023): error: unexpected end of file": "error",
        "    BGATE arm=x2 b=2 PASS": "gate",
        "    TRACE COMPLETE round=1 b=1 tasks=53022 span_us=3094.430": "timing",
        "commit 554133d744b4591c9cf87c68b6817df781137c9b": "meta",
        "ABREPORT label=x verdict=INCOMPLETE reason=round 0 arm B rc=3": "ab",
        "ABMACHINE start host | error: no FAIL here": "ab",
        " systemctl restart fail2ban.service": None,
    }
    for text, kind in cases.items():
        assert extract_key.classify(text)[0] == kind, text


@pytest.mark.parametrize("text,verdict", [
    ("BGATE arm=x4 b=4 PASS", "PASS"), ("B4-RESULT INCOMPLETE", "INCOMPLETE"),
    ("3 passed, 1 failed", "FAIL"), ("12 passed, 0 failed", "PASS"),
    ("SEALED /x complete=True", "PASS")])
def test_gate_verdict(text, verdict):
    assert extract_key.gate_verdict(text) == verdict


def _copy(tmp_path):
    base = tmp_path / "results"
    src = REGISTRY.parent / "2026-10-16-multibatch"
    shutil.copytree(src, base / "2026-10-16-multibatch")
    entry = next(e for e in json.loads(REGISTRY.read_text())["key_extraction"]
                 if e["dir"] == "2026-10-16-multibatch")
    registry = base / "registry.json"
    registry.write_text(json.dumps(dict(key_extraction=[entry], derived=[dict(
        output="2026-10-16-multibatch/summary.json", kind="module_cli",
        module="examples.dsv2lite.multibatch_trace_summary",
        args=["{in:2026-10-16-multibatch/trace_report.json}",
              "{in:2026-10-16-multibatch/baseline/batch_report.json}", "{out}"])])))
    return base, registry


def test_edited_summary_or_log_fails_the_check(tmp_path):
    base, registry = _copy(tmp_path)
    assert check_results.main(["--registry", str(registry)]) == 0
    summary = base / "2026-10-16-multibatch/summary.json"
    summary.write_text(summary.read_text().replace("53022", "53023", 1))
    assert check_results.main(["--registry", str(registry)]) == 1
    shutil.copyfile(REGISTRY.parent / "2026-10-16-multibatch/summary.json", summary)
    log = base / "2026-10-16-multibatch/runner.log"
    log.write_text(log.read_text() + "late line\n")  # the source SHA256 changes
    assert extract_key.main(["--registry", str(registry), "--check"]) == 1


def test_key_lines_carry_file_line_and_sha256():
    key = json.loads((REGISTRY.parent / "2026-10-15/b4/key.json").read_text())
    sha = {s["file"]: s["sha256"] for s in key["sources"]}
    for line in key["lines"]:
        text = (REGISTRY.parent / "2026-10-15/b4" / line["file"]).read_text().splitlines()[line["line"] - 1]
        assert text.rstrip()[:extract_key.MAX_TEXT] == line["text"]
        assert line["sha256"] == sha[line["file"]]
