"""DeepSeek analysis commands behind the deepseek-tools harness, on retained receipts, through external APIs only."""
import hashlib
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / 'examples/gpu-live/results/2026-10-16-multibatch'
# sha256 of the summary the pre-move `python -m examples.dsv2lite.multibatch_trace_summary` wrote from these receipts.
SUMMARY_SHA256 = '4656e7609cc34c45cdd5ea93ad02e4777605d2d9c5a2cc5c3177343aa4b5c709'


def test_multibatch_summary_reproduces_the_retained_derivation(tmp_path):
    out = tmp_path / 'summary.json'
    run = run_command([sys.executable, '-m', 'examples.dsv2lite.multibatch_trace_summary', RECEIPTS / 'trace_report.json',
                       RECEIPTS / 'baseline/batch_report.json', out], cwd=ROOT)
    assert run.returncode == 0, run.stdout
    assert hashlib.sha256(out.read_bytes()).hexdigest() == SUMMARY_SHA256


def test_compare_fleet_requires_its_traces():
    run = run_command([sys.executable, '-m', 'examples.dsv2lite.compare_fleet'], cwd=ROOT)
    assert run.returncode == 2 and '--fleet-trace' in run.stdout
