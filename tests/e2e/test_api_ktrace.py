"""ktrace / vendor / kcmp report commands through the external API; recorded fixtures, no GPU."""
import csv
import json
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def boundary_trace(tmp_path):
    nested = tmp_path / 'nested'
    nested.mkdir()
    write_csv(nested / 'x_kernel_trace.csv', [
        dict(Start_Timestamp=1100, End_Timestamp=4250, Kernel_Name='first_crossing'),
        dict(Start_Timestamp=1200, End_Timestamp=2000, Kernel_Name='contained'),
        dict(Start_Timestamp=2200, End_Timestamp=5670, Kernel_Name='worst_kernel'),
        dict(Start_Timestamp=7100, End_Timestamp=11160, Kernel_Name='later_worst')])
    write_csv(nested / 'x_marker_api_trace.csv', [
        dict(Start_Timestamp=1000, End_Timestamp=4000, Function='kcmp:o_proj:2:21'),
        dict(Start_Timestamp=7000, End_Timestamp=8000, Function='kcmp:o_proj:2:22')])
    return tmp_path


def test_vendor_cpu_dry_run():
    result = run_command([sys.executable, 'examples/vllm_llm/vendor_gemv.py', '--dry-run'], cwd=ROOT, check=True)
    assert 'stopped before torch import' in result.stdout
    assert json.loads(result.stdout.splitlines()[-1])['measurements'] is None


def test_thresholds_are_reported_verbatim(tmp_path):
    text = (ROOT / 'docs/qwen/KCMP-THRESHOLDS.md').read_text()
    thresholds = text.split('## 4. Thresholds, written before the run\n\n')[1].split('\n## 5.')[0]
    output = run_command([sys.executable, '-m', 'examples.vllm_llm.kcmp_report', tmp_path,
                          '--historical-root', tmp_path], cwd=ROOT).stdout
    assert thresholds in output
    assert 'PAIR ' not in output
    assert 'NOT MEASURED: incomplete pairs' in output


def test_boundary_report_cli(boundary_trace):
    result = run_command([sys.executable, '-m', 'etx.tools.ktrace', '--trace-dir', boundary_trace, '--boundaries'], cwd=ROOT)
    assert result.returncode == 0, result.stdout
    payload, summary = result.stdout.split('BOUNDARY-SUMMARY ')
    assert json.loads(payload) == [
        dict(label='kcmp:o_proj:2:22', in_range=1, overhang_us=3.2,
             kernel='later_worst', kernel_end_us=11.2, marker_end_us=8.0),
        dict(label='kcmp:o_proj:2:21', in_range=3, overhang_us=1.7,
             kernel='worst_kernel', kernel_end_us=5.7, marker_end_us=4.0)]
    # stdout and stderr share one log here: the exact payload + summary also proves nothing went to stderr.
    assert summary == 'lib= ranges=2 crossings=2 max_overhang_us=3.2\n'
    assert 'duration' not in result.stdout
    assert 'median_us' not in result.stdout
