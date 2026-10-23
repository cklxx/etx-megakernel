"""The vlbest report command over recorded sessions, moved from tests/test_vlbest_report.py, through external APIs only."""
import json
from pathlib import Path
import sys

from etx.testing import certified_import_session, run_command

ROOT = Path(__file__).resolve().parents[2]
# Session files beside the certified import session; regenerated from the harness by
# tests/test_vlbest_report.py::test_frozen_report_sessions_match_the_harness.
SESSIONS = json.loads((Path(__file__).parent / 'data/vlbest_report_sessions.json').read_text())


def report(tmp_path, kind):
    certified_import_session(tmp_path)
    for name, text in SESSIONS[kind].items():
        (tmp_path / name).write_text(text)
    return run_command([sys.executable, '-m', 'examples.vllm_llm.vlbest_report', str(tmp_path)], cwd=ROOT)


def test_report_spread_diagnostic_roundtrip(tmp_path):
    run = report(tmp_path, 'spread')
    assert run.returncode == 0, run.stdout[-2000:]
    output = run.stdout
    assert 'SPREAD-BOUND fused−R: False' in output
    assert '→ LEAD' in output
    assert 'FULL-CLAIM ELIGIBLE' in output
    assert 'ETX: step mean; vLLM: sample median' in output
    assert 'FUSION step-mean basis' in output
    assert 'FUSION RESULT wall_ms basis (pairs.py)' in output
    assert 'Synthetic GPU | ROCm test-version' in output


def test_changed_final_configuration_refused(tmp_path):
    run = report(tmp_path, 'changed_final')
    assert run.returncode == 2 and 'finalist configuration changed' in run.stdout
