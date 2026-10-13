from pathlib import Path
import subprocess
import json
import os

import pytest


def test_session_sh():
    script = Path(__file__).resolve().with_suffix('.sh')
    subprocess.run(['bash', str(script)], cwd=script.parent.parent, check=True)


ROOT = Path(__file__).resolve().parent.parent


def stub(tmp_path, name, body):
    path = tmp_path / 'bin' / name
    path.parent.mkdir(exist_ok=True)
    path.write_text('#!/bin/bash\n' + body + '\n')
    path.chmod(0o755)


@pytest.fixture
def remote(tmp_path):
    # ssh executes only the supplied stdin block in an isolated local tree.
    stub(tmp_path, 'ssh', 'shift 5; exec "$@"')
    stub(tmp_path, 'cat', 'exit 1')
    stub(tmp_path, 'hipconfig', '''if [ "${VERSION_MODE:-ok}" = retry ] && [ ! -f "$HOME/slept" ]; then exit 1; fi
[ "${VERSION_MODE:-ok}" != missing ] || exit 1
echo 7.14.0''')
    stub(tmp_path, 'sleep', 'echo "SLEEP $1"; touch "$HOME/slept"')
    stub(tmp_path, 'sudo', '[ "$*" = "docker info" ] && [ "${DOCKER_FAIL:-0}" = 0 ]')
    stub(tmp_path, 'python3', 'test -f etx/tools/gpu_jobs.py && echo "69 jobs; budget stub"')
    env = dict(os.environ, HOME=str(tmp_path), PATH=str(tmp_path / 'bin') + ':' + os.environ['PATH'])
    return tmp_path, env


def pushed(tree):
    for name in ('rocm-headers/inc/hip/hip_runtime.h', 'etx/etx/tools/gpu_jobs.py'):
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tree / 'rocm-headers/pytorch').mkdir(parents=True)


def run_session(env, *args):
    return subprocess.run(['bash', str(ROOT / 'tools/session.sh'), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True)


def test_preflight_empty_tree(remote):
    tree, env = remote
    result = run_session(env, 'preflight', 'localhost')
    assert result.returncode == 1
    assert 'PREFLIGHT FAIL pushed-tree: test -f ~/rocm-headers/inc/hip/hip_runtime.h' in result.stdout
    assert 'PREFLIGHT FAIL dry-run: cd ~/etx' in result.stdout
    assert 'PREFLIGHT OK' not in result.stdout


@pytest.mark.parametrize('mode', ['ok', 'retry', 'missing'])
def test_preflight_version(remote, mode):
    tree, env = remote
    pushed(tree)
    env['VERSION_MODE'] = mode
    result = run_session(env, 'preflight', 'localhost')
    assert result.returncode == (1 if mode == 'missing' else 0)
    assert result.stdout.count('SLEEP 300') == (0 if mode == 'ok' else 1)
    if mode == 'missing':
        assert 'PREFLIGHT FAIL rocm-version:' in result.stdout
    else:
        assert 'PREFLIGHT OK\nROCM-VERSION 7.14.0' in result.stdout


@pytest.mark.parametrize('failure', ['headers', 'docker'])
def test_preflight_failed_checks(remote, failure):
    tree, env = remote
    pushed(tree)
    if failure == 'headers':
        (tree / 'rocm-headers/inc/hip/hip_runtime.h').unlink()
    else:
        env['DOCKER_FAIL'] = '1'
    result = run_session(env, 'preflight', 'localhost')
    assert result.returncode == 1
    assert f'PREFLIGHT FAIL {"pushed-tree" if failure == "headers" else "container"}:' in result.stdout


def capture(dest):
    dest.mkdir(parents=True)
    (dest / 'status.json').write_text(json.dumps([
        {'name': 'build', 'status': 'fail', 'rc': 1, 'note': 'exit'},
        {'name': 'gate', 'status': 'skip', 'rc': None, 'note': 'build failed'},
        {'name': 'good', 'status': 'ok', 'rc': 0}]))
    (dest / 'build.log').write_text('# job build\n# exact build command\n0\n1\n2\n3\n4\n')
    (dest / 'runner.log').write_text('[23:59:00 +  0m00s] start\n[00:01:31 +  2m31s] session done\n')


def report(dest):
    return subprocess.run(['python3', str(ROOT / 'tools/session_postmortem.py'), str(dest)], text=True,
                          capture_output=True)


def test_postmortem_report_names_failed_skipped_and_missing(tmp_path):
    capture(tmp_path / 'one')
    result = report(tmp_path)
    assert result.returncode == 1
    for line in ('JOB build status=fail', '# exact build command', '0\n1\n2\n3\n4', 'JOB gate status=skip',
                 f'UNCAPTURED {tmp_path}/one/gate.log', ': 2.52 (captured runner span'):
        assert line in result.stdout
    assert 'JOB good' not in result.stdout
    (tmp_path / 'one/gate.log').write_text('# job gate\n# skipped command\n')
    assert report(tmp_path).returncode == 0


@pytest.mark.parametrize('missing', ['status.json', 'runner.log', 'build.log'])
def test_postmortem_report_partial(tmp_path, missing):
    capture(tmp_path / 'c')
    (tmp_path / 'c' / missing).unlink()
    (tmp_path / 'c/gate.log').write_text('# job gate\n# skipped command\n')
    result = report(tmp_path / 'c')
    assert result.returncode == 1 and f'UNCAPTURED {tmp_path}/c/{missing}' in result.stdout


@pytest.fixture
def box(tmp_path):
    """The box is this machine under HOME=tmp_path: ssh runs its command locally, rsync drops the host prefix."""
    stub(tmp_path, 'ssh', 'shift 5; exec bash -c "$*"')
    stub(tmp_path, 'rsync', 'a=(); for x in "$@"; do a+=("${x/#*@localhost:/$HOME/}"); done; exec /usr/bin/rsync "${a[@]}"')
    env = dict(os.environ, HOME=str(tmp_path), PATH=str(tmp_path / 'bin') + ':' + os.environ['PATH'])
    return tmp_path, env


def sealed_session(tmp_path, cmd):
    import sys
    session = tmp_path / 'session.py'
    session.write_text(f'JOBS = [dict(name="work", lane="cpu", cmd={cmd!r})]\n')
    subprocess.run([sys.executable, '-m', 'etx.tools.gpu_jobs', str(session), '--gpus', '', '--logdir',
                    str(tmp_path / 'sess')], cwd=ROOT, check=False, capture_output=True, timeout=30)
    return max((tmp_path / 'sess').iterdir())


def test_fetch_copies_one_sealed_record_into_its_own_directory(box):
    tmp_path, env = box
    rec = sealed_session(tmp_path, 'mkdir -p "$ETX_SESS_LOGDIR/b_x" && echo plan > "$ETX_SESS_LOGDIR/b_x/plan.json" '
                                   '&& echo bin > "$ETX_SESS_LOGDIR/b_x/run" && echo r > "$ETX_SESS_LOGDIR/pairs_x.json"')
    dest = tmp_path / 'capture'
    result = run_session(env, 'fetch', 'localhost', str(dest))
    assert result.returncode == 0, result.stdout + result.stderr
    got = dest / rec.name
    assert (got / 'b_x/plan.json').exists() and not (got / 'b_x/run').exists()        # R03: binaries stay
    assert 'VERIFIED' in result.stdout and '1 left on the box by design' in result.stdout
    assert run_session(env, 'fetch', 'localhost', str(dest)).returncode == 2          # R12: never onto a copy
    (got / 'pairs_x.json').write_text('changed\n')                                    # R04: the copy is checked
    (got / 'stale.json').write_text('{}')                                              # R12: nothing unsealed counts
    out = subprocess.run(['python3', str(ROOT / 'etx/session/record.py'), 'verify', str(got)], text=True,
                         capture_output=True).stdout
    assert 'MISMATCH pairs_x.json' in out and 'NOT IN THE SEALED MANIFEST stale.json' in out


def test_incomplete_unsealed_and_absent_records_are_never_results(box):
    tmp_path, env = box
    rec = sealed_session(tmp_path, 'exit 3')                                       # R13: a job not ok
    assert run_session(env, 'fetch', 'localhost', str(tmp_path / 'a')).returncode == 3
    (tmp_path / 'sess' / 'unsealed').mkdir()                                       # the runner died: no SEAL
    (tmp_path / 'sess' / 'unsealed' / 'pairs_x.json').write_text('{}')
    result = run_session(env, 'fetch', 'localhost', str(tmp_path / 'a'), 'unsealed')
    assert result.returncode == 3 and 'UNSEALED' in result.stdout
    for d in ('sess/unsealed', f'sess/{rec.name}'):
        subprocess.run(['rm', '-rf', str(tmp_path / d)], check=True)
    (tmp_path / 'sess/startup.abc').write_text("ModuleNotFoundError: No module named 'etx'\n")
    result = run_session(env, 'postmortem', 'localhost', str(tmp_path / 'pm'))       # R12: startup-failed box
    assert result.returncode == 1 and 'NO RECORD' in result.stdout and 'UNCAPTURED' in result.stdout
    assert "No module named 'etx'" in result.stdout


def test_stage_extracts_the_commit_into_a_new_directory(box, tmp_path):
    tmp, env = box
    repo = tmp / 'repo'
    (repo / 'tools').mkdir(parents=True)
    (repo / 'tools/session.sh').write_bytes((ROOT / 'tools/session.sh').read_bytes())
    (repo / '.gitignore').write_text('*.pyc\n')
    git = lambda *a: subprocess.run(['git', '-C', str(repo), *a], check=True, capture_output=True)
    git('init', '-q'); git('add', '.'); git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'x')
    (repo / 'yaml.pyc').write_text('ignored, untracked')                           # R08: never staged
    (tmp / 'etx').mkdir()
    (tmp / 'etx/yaml.pyc').symlink_to(tmp / 'elsewhere')                            # R08': a used box's tree
    result = subprocess.run(['bash', str(repo / 'tools/session.sh'), 'stage', 'localhost'], env=env, text=True,
                            capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    staged = (tmp / 'etx').resolve()
    assert sorted(p.name for p in staged.iterdir()) == ['.gitignore', 'SOURCE_IDENTITY', 'tools']
    from examples.dsv2lite.inplace_session import Session
    s = object.__new__(Session)
    s.root = staged
    assert 'archive ' + staged.name[:64] in s.source_identity()
    (repo / '.gitignore').write_text('changed\n')                                  # an uncommitted change: refused
    assert subprocess.run(['bash', str(repo / 'tools/session.sh'), 'stage', 'localhost'], env=env,
                          capture_output=True).returncode == 2
