"""Command-level cases moved from tests/test_nvchk.py, through external APIs only."""
import json
from pathlib import Path
import sys

import pytest
from etx.testing import certificate_envelope as envelope, certified_import_session as fixture_session, run_command, timing_log as etx

ROOT = Path(__file__).resolve().parents[2]


def nvchk(*argv):
    return run_command([sys.executable, '-m', 'examples.sessions.nvchk', *map(str, argv)], cwd=ROOT)


@pytest.mark.parametrize('failure', [None, 'missing', 'status', 'identity', 'wrong_arm', 'gpu'])
def test_triplet_report_requires_complete_same_build(tmp_path, failure):
    fixture_session(tmp_path)
    identity = json.loads((tmp_path / 'build_identity.json').read_text())
    statuses = []
    for rnd in range(1, 4):
        for arm in ('poll', 'block', 'graph'):
            name = f'{arm}_r{rnd}'
            mode = 'graph' if arm == 'graph' else 'fused'
            wall = {'poll': 4.5, 'block': 4.4, 'graph': 4.7}[arm]
            text = f'# job {name} gpu=0\n' + envelope(identity, etx(wall, arm=mode))
            (tmp_path / f'{name}.log').write_text(text)
            statuses.append(dict(name=name, status='ok'))
    p = tmp_path / 'block_r2.log'
    if failure == 'missing':
        p.unlink()
    elif failure == 'status':
        statuses[4]['status'] = 'timeout'
    elif failure == 'identity':
        p.write_text(p.read_text().replace('phase=after', 'phase=wrong'))
    elif failure == 'wrong_arm':
        p.write_text(p.read_text().replace('mode: ETX megakernel', 'mode: unfused'))
    elif failure == 'gpu':
        p.write_text(p.read_text().replace('gpu=0', 'gpu=1'))
    (tmp_path / 'status.json').write_text(json.dumps(statuses))
    result = nvchk('--report-sync', tmp_path)
    assert result.returncode == (2 if failure else 0)
    assert ('SYNC NOT-COMPLETE' if failure else 'SYNC supported=True') in result.stdout


def test_dry_run_lists_the_session():
    result = nvchk('--dry-run')
    assert result.returncode == 0, result.stdout
    assert 'check_timed' in result.stdout and 'vllm_last' in result.stdout
