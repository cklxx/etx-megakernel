"""The DeepSeek batch report command not projecting from legacy subset means, moved from tests/test_dsv2_batch.py."""
import json
import os
from pathlib import Path
import shutil
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
# Five recorded rounds (ETX B1/B4, vLLM B1/B4); regenerated from the log parser by
# tests/test_dsv2_batch.py::test_frozen_report_rounds_match_the_parser.
ROUNDS = Path(__file__).parent / 'data/dsv2_report_rounds.json'
# examples.dsv2lite.arms.BATCHES, pinned in tests/test_e2e_pins.py.
BATCHES = (1, 2, 4, 8)


def test_report_does_not_project_from_legacy_subset_means(tmp_path):
    selection = {str(b): dict(config='aiter') for b in BATCHES}
    (tmp_path / 'vsel.json').write_text(json.dumps(dict(selection=selection)))
    shutil.copyfile(ROUNDS, tmp_path / 'brounds.json')
    (tmp_path / 'routing.json').write_text(json.dumps(dict(overall={'4': 15})))
    env = dict(os.environ, ETX_SESS_LOGDIR=str(tmp_path), ETX_DSV2_FLEET=str(tmp_path), ETX_DSV2_DATA=str(tmp_path))
    run = run_command([sys.executable, '-m', 'examples.dsv2lite.batch_session', 'report'], cwd=ROOT, env=env)
    assert run.returncode == 0, run.stdout[-1500:]
    report = json.loads((tmp_path / 'batch_report.json').read_text())
    assert 'PENDING' in report['batches']['4']['next']['next']
