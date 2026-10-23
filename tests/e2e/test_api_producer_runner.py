"""The job runner refusing measure after a failed qualify, moved from tests/test_producer.py, through external APIs only."""
import json
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_runner_failed_qualify_skips_measure_and_preserves_unique_records(tmp_path):
    """Execute the actual runner with CPU stand-ins. No GPU code is executed."""
    lease=tmp_path/'lease'
    jobfile=tmp_path/'session.py'
    jobfile.write_text(f'''MAX_SESSION_SECONDS=20
SESSION_LEASE={str(lease)!r}
JOBS=[dict(name='qualify',lane='gpu',gpu=0,cmd="exit 2",writes=['correctness.json'],timeout=2,cwd={str(ROOT)!r}),
      dict(name='measure',lane='gpu',gpu=0,cmd="exit 99",reads=['correctness.json'],timeout=2,cwd={str(ROOT)!r})]
''')
    args=[sys.executable,'-m','etx.tools.gpu_jobs',str(jobfile),'--gpus','0','--logdir',str(tmp_path/'records'),'--progress-s','1']
    assert run_command(args,cwd=ROOT).returncode!=0
    first=json.loads(lease.read_text())
    assert run_command(args,cwd=ROOT).returncode!=0
    assert json.loads(lease.read_text())==first
    records=list((tmp_path/'records').iterdir());assert len(records)==2
    for d in records:
        statuses=json.loads((d/'status.json').read_text())
        assert [j['status'] for j in statuses]==['fail','skipped']
        assert not (d/'measure.log').exists()
    only=run_command([*args,'--only','measure'],cwd=ROOT)
    assert only.returncode!=0 and 'complete graph' in only.stdout
