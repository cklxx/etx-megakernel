"""Producer archive, attach and documented start commands, moved from tests/test_producer_readiness.py."""
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import venv

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def inputs(tmp_path):
    model = tmp_path/'model'
    hf = model/'hf'
    hf.mkdir(parents=True)
    for name in ('vl.bin', 'vl.manifest'):
        (model/name).write_bytes(b'synthetic packed input')
    (model/'golden_c1024.txt').write_text('prompt 1 2\nexpected 3\n')
    for name in ('config.json', 'REVISION', 'model.safetensors'):
        (hf/name).write_text('synthetic checkpoint')
    return dict(model=str(model), hf=str(hf))


def test_fresh_archive_interpreter_preflight_and_failure_report(tmp_path):
    """Real git archive, real Python/deps, actual job graph and report. No GPU.

    Only the hardware/source preflight and compute are CPU substitutes. The
    missing interpreter failure is exercised before attachment, not mocked.
    """
    archive = tmp_path/'archive'
    archive.mkdir()
    bundle = tmp_path/'source.tar'
    archived = run_command(['git', 'archive', '--output', str(bundle), 'HEAD'], cwd=ROOT, timeout=30)
    assert archived.returncode == 0, archived.stdout
    with tarfile.open(bundle) as tar:
        tar.extractall(archive, filter='data')
    bundle.unlink()
    assert not (archive/'.venv').exists()
    # This is the real pre-fix command failure in a new archive.
    missing = run_command(['bash', '-c', '.venv/bin/python -m examples.vllm_llm.producer_session preflight'],
                          cwd=archive, timeout=10)
    assert missing.returncode == 127
    print('ARCHIVE before attach exit=127:', missing.stdout.strip())
    # A new environment independent of both the archive and the previous checkout.
    envdir = tmp_path/'independent-env'
    venv.EnvBuilder(with_pip=False).create(envdir)
    py = envdir/'bin/python'
    found = run_command([str(py), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], timeout=30)
    assert found.returncode == 0, found.stdout
    purelib = Path(found.stdout.strip())
    # The checkout's numpy and yaml packages, located by the running interpreter.
    located = run_command([sys.executable, '-c', 'import numpy, yaml, pathlib; '
                           'print("\\n".join(str(pathlib.Path(m.__file__).parent) for m in (numpy, yaml)))'], timeout=30)
    assert located.returncode == 0, located.stdout
    for line in located.stdout.split():
        source = Path(line)
        shutil.copytree(source, purelib/source.name)
    attached = run_command([sys.executable, '-m', 'examples.vllm_llm.producer_stage', str(envdir)], cwd=archive, timeout=60)
    assert attached.returncode == 0, attached.stdout
    # Actual VM interpreter probe at the resolved archive root.
    code = 'from examples.vllm_llm.vm_preflight import check; raise SystemExit(check("/unused", "/unused", only=("interpreter",)))'
    probe = run_command([str(archive/'.venv/bin/python'), '-c', code], cwd=archive, timeout=30)
    print(probe.stdout)
    assert probe.returncode == 0, probe.stdout
    # Actual producer preflight and report must resolve even without GPU inputs.
    c = inputs(tmp_path)
    config = tmp_path/'config.json'
    config.write_text(json.dumps(c))
    records = tmp_path/'records'
    fixture = tmp_path/'cpu_session.py'
    fixture.write_text(f'''from examples.sessions.qwen_producer import JOBS
MAX_SESSION_SECONDS=30
SESSION_LEASE={str(tmp_path/'lease')!r}
for job in JOBS:
    job['cwd']={str(archive)!r}
    job['timeout']=10
    if job['name'] not in ('preflight','report'):
        job['cmd']='echo CPU-COMPUTE-SUBSTITUTE-MUST-NOT-RUN; exit 99'
''')
    env = dict(os.environ, ETX_QWEN_G6_CONFIG=str(config), ETX_SESS_MAX_SECONDS='30')
    run = run_command([str(py), '-u', '-m', 'etx.tools.gpu_jobs', str(fixture), '--gpus', '0',
                       '--logdir', str(records), '--progress-s', '1'], cwd=archive, env=env, timeout=40)
    print('ARCHIVE runner exit=', run.returncode, '\n', run.stdout)
    assert run.returncode != 0
    record, = records.iterdir()
    statuses = {j['name']: j['status'] for j in json.loads((record/'status.json').read_text())}
    assert statuses['preflight'] == 'fail'
    assert all(statuses[n] == 'skipped' for n in ('build_baseline','build_producer','qualify','measure','diagnose'))
    preflight = (record/'preflight.log').read_text()
    print(preflight)
    assert 'FAIL phase=preflight' in preflight
    report = json.loads((record/'report.json').read_text())
    print('ARCHIVE actual report:', json.dumps(report))
    assert report['status'] == 'NOT-COMPLETE' and report['performance_claim'] is False
    # Keep the small record; the archive and independent venv are reproducible.
    shutil.rmtree(archive)
    shutil.rmtree(envdir)


def test_documented_push_stage_attach_start_with_local_transport(tmp_path):
    """Execute the real shell archive/push/start path. Transport/probes only are local substitutes.

    SSH never contacts a host; rsync, product probes and key management are labelled
    no-ops. The real preflight fails closed on synthetic inputs; compute cannot run.
    """
    home = tmp_path/'home'
    (home/'code').mkdir(parents=True)
    binpath = tmp_path/'bin'
    binpath.mkdir()
    c = inputs(home)
    config = home/'config.json'
    config.write_text(json.dumps(c))
    def script(name, body):
        p = binpath/name
        p.write_text(body)
        p.chmod(0o755)
    script('ssh', f'''#!{sys.executable}
import subprocess, sys
args=sys.argv[1:]
while args and args[0]=='-o': args=args[2:]
assert args.pop(0)=='hotaisle@local-fixture'
raise SystemExit(subprocess.call(['bash','-c',' '.join(args)]))
''')
    for name in ('rsync','ssh-keygen'):
        script(name, '#!/bin/sh\necho LOCAL-TRANSPORT-SUBSTITUTE '+name+'\n')
    script('rocm-smi', '#!/bin/sh\necho "Card Series: CPU-SUBSTITUTE"\n')
    script('nproc', '#!/bin/sh\necho 1\n')
    script('setsid', '#!/bin/sh\nexec "$@"\n')
    # The independent-venv archive test above verifies dependency staging itself.
    (binpath/'python3').symlink_to(sys.executable)
    env = dict(os.environ, HOME=str(home), PATH=str(binpath)+os.pathsep+os.environ['PATH'])
    command = [sys.executable, '-u', '-m', 'tools.producer_start', 'local-fixture',
               '--venv', sys.prefix, '--config', str(config), '--allocation-deadline', str(time.time()+600)]
    started = run_command(command, cwd=ROOT, env=env, timeout=120)
    print('LOCAL PUSH/STAGE/ATTACH/START\n'+started.stdout)
    assert started.returncode == 0
    assert (home/'etx/SOURCE_IDENTITY').is_file()
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        reports = list((home/'sess').glob('*/report.json'))
        if reports:
            break
        time.sleep(.1)
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())['status'] == 'NOT-COMPLETE'
    record = reports[0].parent
    print((record/'preflight.log').read_text())
    assert 'FAIL phase=preflight' in (record/'preflight.log').read_text()
    assert not (record/'build_baseline.log').exists()
    assert not (record/'measure.log').exists()
    assert 'VERDICT ' in (record/'report.log').read_text()
    # The session may need one more tick to seal. Do not leave an active runner.
    while time.monotonic() < deadline and not (record/'SEAL').exists():
        time.sleep(.1)
    assert (record/'SEAL').exists()
    shutil.rmtree(home/'src')  # Runner is sealed; retain its logs under home/sess.
