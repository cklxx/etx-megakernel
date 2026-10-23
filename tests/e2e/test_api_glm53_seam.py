"""GLM seam commands (summary, bounded runner) through the external API; CPU fixtures, never a GPU."""
import json
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
LABEL = 'production-shaped synthetic handoff; not full layer cost'


def receipts(batch=2, trace=False, iterations=2, rounds=2, warmup=1):
    config = dict(schema=1, label=LABEL, batch=batch, trace=trace,
                  iterations=iterations, rounds=rounds, warmup=warmup, row_bytes=32768)
    records = [('SEAM-CONFIG', config)]
    for rnd in range(rounds):
        for arm in ('payload', 'control'):
            for phase, n in (('warmup', warmup), ('sample', iterations)):
                for step in range(n):
                    x = 100 if arm == 'payload' else 80
                    rec = dict(round=rnd, arm=arm, phase=phase, iteration=step,
                               correct=True, rows_checked=batch, elements_checked=batch*16384,
                               wall_us=x+rnd+step, rank0_span_us=x-10, rank1_span_us=x-20)
                    if trace:
                        rec['trace_rows'] = [dict(row=r, send=[9000000, 9000100, 9000300],
                                                  consume=[1, 101, 301],
                                                  feedback=[9000200, 9000900, 9001000]) for r in range(batch)]
                    records.append(('SEAM-STEP', rec))
    records.append(('SEAM-RESULT', dict(status='PASS', steps_checked=rounds*2*(iterations+warmup))))
    return records


def write_log(tmp_path, records, name='run.log'):
    path = tmp_path/name
    path.write_text('full unfiltered chatter\n' + '\n'.join(f'{k} {json.dumps(v)}' for k, v in records)+'\n')
    return path


def seam(*argv, timeout=60):
    return run_command([sys.executable, '-m', 'examples.glm53.seam', *map(str, argv)], cwd=ROOT, timeout=timeout)


def test_offline_cli_without_gpu(tmp_path):
    path = write_log(tmp_path, receipts(trace=True))
    proc = seam('summary', path)
    assert proc.returncode == 0, proc.stdout
    assert json.loads(proc.stdout)['status'] == 'COMPLETE'
    path.write_text('truncated run\n')
    proc = seam('summary', path)
    assert proc.returncode == 2 and 'INCOMPLETE' in proc.stdout


@pytest.mark.parametrize('args', [
    ['--iterations', '0'], ['--rounds', '-1'], ['--warmup', '0'],
    ['--timeout', 'nan'], ['--batches', '1', '1'],
])
def test_run_arguments_fail_before_gpu_call(tmp_path, args):
    logdir = tmp_path/'logs'
    assert seam('run', '--build-root', tmp_path/'build', '--logdir', logdir, *args).returncode == 2
    assert not logdir.exists()


def test_runner_missing_binary_keeps_full_log_and_fails(tmp_path):
    logs = tmp_path/'logs'
    assert seam('run', '--build-root', tmp_path/'missing', '--logdir', logs, '--batches', '1').returncode == 2
    assert (logs/'b1-trace0.log').exists()


@pytest.mark.parametrize('wrong', [
    dict(batch=2), dict(trace=True), dict(iterations=1), dict(rounds=1), dict(warmup=1),
])
def test_runner_rejects_internally_complete_wrong_contract(tmp_path, wrong):
    requested = dict(batch=1, trace=False, iterations=2, rounds=2, warmup=2)
    actual = dict(requested, **wrong)
    data = '\n'.join(f'{k} {json.dumps(v)}' for k, v in receipts(**actual))
    exe = tmp_path/'build'/'b1'/'run'
    exe.parent.mkdir(parents=True)
    (exe.parent/'plan.json').write_text('{}')
    (exe.parent/'plan_data.h').write_text('// fixture')
    exe.write_text(f'#!{sys.executable}\nprint({data!r})\n')
    exe.chmod(0o755)
    logs = tmp_path/'logs'
    assert seam('run', '--build-root', tmp_path/'build', '--logdir', logs,
                '--batches', '1', '--iterations', '2', '--rounds', '2', '--warmup', '2').returncode == 2
    # The receipt itself was complete (the offline parser accepts a copy outside the run's manifest);
    # only the request contract catches this mismatch.
    alone = tmp_path/'alone'; alone.mkdir()
    (alone/'b1-trace0.log').write_bytes((logs/'b1-trace0.log').read_bytes())
    assert seam('summary', alone/'b1-trace0.log').returncode == 0
    manifest = json.loads((logs/'manifest.json').read_text())
    assert manifest['status'] == 'FAILED'
    assert 'contract' in manifest['runs'][0]['error']
    assert not (logs/'b1-trace1.log').exists()


def test_runner_enforces_process_deadline_and_preserves_partial_log(tmp_path):
    exe = tmp_path/'build'/'b1'/'run'
    exe.parent.mkdir(parents=True)
    (exe.parent/'plan.json').write_text('{"fixture": "CPU dry run, not a real plan"}\n')
    (exe.parent/'plan_data.h').write_text('// CPU dry-run fixture\n')
    exe.write_text('#!/bin/sh\nprintf "%s\\n" "partial native log"\nexec sleep 30\n')
    exe.chmod(0o755)
    logs = tmp_path/'timeout-logs'
    proc = seam('run', '--build-root', tmp_path/'build', '--logdir', logs, '--batches', '1', '--timeout', '0.2')
    assert proc.returncode == 2
    assert (logs/'b1-trace0.log').exists()  # Deadline may fire before the OS starts the child.
    assert 'timed out' in proc.stdout
    summary = seam('summary', logs/'b1-trace0.log')
    assert summary.returncode == 2 and 'INCOMPLETE' in summary.stdout


def test_fake_binary_complete_dry_run(tmp_path):
    import hashlib
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    # Execute the real orchestration against a CPU-only protocol fixture.
    exe = tmp_path/'build'/'b1'/'run'
    exe.parent.mkdir(parents=True)
    (exe.parent/'plan.json').write_text('{"fixture": "CPU dry run, not a real plan"}\n')
    (exe.parent/'plan_data.h').write_text('// CPU dry-run fixture\n')
    cases = {str(int(trace)): '\n'.join(f'{k} {json.dumps(v)}' for k, v in
                                      receipts(batch=1, trace=trace, iterations=1, rounds=1))
             for trace in (False, True)}
    exe.write_text(f'#!{sys.executable}\nimport json, sys\ncases = {cases!r}\n'
                   'print("unfiltered fake native diagnostics")\nprint(cases[sys.argv[4]])\n')
    exe.chmod(0o755)
    logs = tmp_path/'dry-logs'
    proc = seam('run', '--build-root', tmp_path/'build', '--logdir', logs,
                '--batches', '1', '--iterations', '1', '--rounds', '1', '--warmup', '1')
    assert proc.returncode == 0, proc.stdout
    for trace in (False, True):
        assert 'unfiltered fake native diagnostics' in (logs/f'b1-trace{int(trace)}.log').read_text()
    manifest = json.loads((logs/'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETE' and not manifest['model_certification']
    assert len(manifest['runs']) == 2
    for entry in manifest['runs']:
        assert entry['status'] == 'PASS'
        assert entry['log_sha256'] == sha(entry['log'])
        for artifact in entry['artifacts'].values():
            assert artifact['sha256'] == sha(artifact['path'])
    assert 'COMPLETE' in proc.stdout
    all_logs = sorted(logs.glob('*.log'))
    summary = seam('summary', *all_logs)
    assert summary.returncode == 0 and all(r['runner_manifest_verified'] for r in json.loads(summary.stdout)['runs'])
    all_logs[0].write_text(all_logs[0].read_text()+'extra chatter after collection\n')
    tampered = seam('summary', *all_logs)
    assert tampered.returncode == 2 and 'hash' in tampered.stdout
