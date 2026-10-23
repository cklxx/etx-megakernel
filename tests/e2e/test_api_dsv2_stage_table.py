"""The DeepSeek stage-table command regenerating the committed table, moved from tests/test_dsv2_stage_table.py."""
import json
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
# examples.dsv2lite.stage_table.OUT, pinned in tests/test_e2e_pins.py.
COMMITTED = ROOT / 'examples/dsv2lite/results/stages'


def test_every_stage_has_a_lever_and_the_rows_partition_the_token(tmp_path):
    run = run_command([sys.executable, '-m', 'examples.dsv2lite.stage_table', '--out', str(tmp_path)], cwd=ROOT, timeout=600)
    if run.returncode and 'release' in run.stdout and 'download' in run.stdout:
        pytest.skip('archived data sets unavailable (gh release download failed)')
    assert run.returncode == 0, run.stdout[-2000:]
    for name in ('stage_table.json', 'stage_table.md'):     # the committed table regenerates byte for byte
        assert (tmp_path / name).read_bytes() == (COMMITTED / name).read_bytes(), name
    out = json.loads((tmp_path / 'stage_table.json').read_text())
    rows = {r['stage']: r for r in out['rows']}
    assert len(rows) == 14
    assert abs(sum(r['current_us'] for r in rows.values()) - out['summary']['traced_token_us']) < 1e-6
    for r in rows.values():
        assert r['lever'] and r['formula'], r['stage']
        assert (r['delta_us'] < 0 and r['basis'] != '-') or 'REJECTED' in r['lever'], r['stage']
    saving = -sum(r['delta_us'] for r in rows.values())
    assert abs(out['summary']['arithmetic']['saving_us'] - saving) < 1e-6
    assert out['summary']['simulated']['saving_us'] <= saving + 1e-6
