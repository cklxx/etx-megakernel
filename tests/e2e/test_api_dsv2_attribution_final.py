"""The DeepSeek batch-attribution final lane, moved from tests/test_dsv2_batch_attribution.py, through external APIs only."""
import json
import os
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
# examples.dsv2lite.batch_attribution.BATCHES, pinned in tests/test_e2e_pins.py.
BATCHES = (1, 2, 4)


def final(tmp_path):
    env = dict(os.environ, ETX_SESS_LOGDIR=str(tmp_path), ETX_DSV2_FLEET=str(tmp_path), ETX_DSV2_DATA=str(tmp_path))
    run = run_command([sys.executable, '-m', 'examples.dsv2lite.batch_attribution', 'final'], cwd=ROOT, env=env)
    assert run.returncode == 0, run.stdout[-1500:]
    return json.loads((tmp_path/'b4_result.json').read_text())


def test_final_is_structured_when_all_dependencies_failed(tmp_path):
    result = final(tmp_path)
    assert result['status'] == 'INCOMPLETE'
    assert all('error' in result[k] for k in ('performance', 'attribution', 'routing'))


def test_full_claim_stays_incomplete_when_attribution_and_vllm_complete(tmp_path):
    (tmp_path/'attribution.json').write_text('{"status":"COMPLETE"}')
    (tmp_path/'routing.json').write_text('{"status":"COMPLETE","bound_prefix":true}')
    (tmp_path/'batch_report.json').write_text(json.dumps(dict(batches={str(b): dict(status='COMPLETE') for b in BATCHES})))
    (tmp_path/'brounds.json').write_text(json.dumps(dict(same_kernel_graph={str(b): dict(status='UNAVAILABLE') for b in BATCHES})))
    value = final(tmp_path)
    assert value['status'] == 'INCOMPLETE' and value['performance_claim_status'].startswith('INCOMPLETE')
    assert value['corrected_vllm_status'] == value['attribution_status'] == 'COMPLETE'
