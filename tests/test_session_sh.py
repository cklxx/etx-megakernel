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


def postmortem_env(tmp_path, fail=False):
    stub(tmp_path, 'rsync', 'exit ' + ('1' if fail else '0'))
    return dict(os.environ, HOME=str(tmp_path), PATH=str(tmp_path / 'bin') + ':' + os.environ['PATH'])


def capture(dest):
    dest.mkdir(parents=True)
    (dest / 'status.json').write_text(json.dumps([
        {'name': 'build', 'status': 'fail', 'rc': 1, 'note': 'exit'},
        {'name': 'gate', 'status': 'skip', 'rc': None, 'note': 'build failed'},
        {'name': 'good', 'status': 'ok', 'rc': 0}]))
    (dest / 'build.log').write_text('# job build\n# exact build command\n0\n1\n2\n3\n4\n')
    (dest / 'runner.log').write_text('[23:59:00 +  0m00s] start\n[00:01:31 +  2m31s] session done\n')


def test_postmortem_skipped_missing_log(tmp_path):
    dest = tmp_path / 'capture'
    capture(dest / 'session-one')
    result = run_session(postmortem_env(tmp_path), 'postmortem', 'localhost', str(dest))
    assert result.returncode == 1
    assert 'JOB build status=fail' in result.stdout
    assert '# exact build command' in result.stdout
    assert '0\n1\n2\n3\n4' in result.stdout
    assert 'JOB gate status=skip' in result.stdout
    assert f'UNCAPTURED {dest}/session-one/gate.log' in result.stdout
    assert ': 2.52 (captured runner span' in result.stdout
    assert 'JOB good' not in result.stdout


def test_postmortem_complete(tmp_path):
    dest = tmp_path / 'capture'
    capture(dest)
    (dest / 'gate.log').write_text('# job gate\n# skipped command\n')
    result = run_session(postmortem_env(tmp_path), 'postmortem', 'localhost', str(dest))
    assert result.returncode == 0
    assert 'UNCAPTURED' not in result.stdout


def test_postmortem_missing_tree_and_failed_fetch(tmp_path):
    dest = tmp_path / 'empty'
    result = run_session(postmortem_env(tmp_path, fail=True), 'postmortem', 'localhost', str(dest))
    assert result.returncode == 1
    for suffix in (' (fetch failed)', '/status.json', '/runner.log'):
        assert f'UNCAPTURED {dest}{suffix}' in result.stdout


@pytest.mark.parametrize('missing', ['status.json', 'runner.log', 'build.log'])
def test_postmortem_partial(tmp_path, missing):
    dest = tmp_path / 'capture'
    capture(dest)
    (dest / missing).unlink()
    result = run_session(postmortem_env(tmp_path), 'postmortem', 'localhost', str(dest))
    assert result.returncode == 1
    assert f'UNCAPTURED {dest}/{missing}' in result.stdout


def test_fetch_keeps_stage_b_logs(tmp_path):
    stub(tmp_path, 'rsync', 'printf "%s\\n" "$*" >> "$HOME/rsync-args"')
    env = dict(os.environ, HOME=str(tmp_path), PATH=str(tmp_path / 'bin') + ':' + os.environ['PATH'])
    result = run_session(env, 'fetch', 'localhost', str(tmp_path / 'capture'))
    assert result.returncode == 0
    first = (tmp_path / 'rsync-args').read_text().splitlines()[0]
    # review 07, R03: build records are part of the result -- only binaries, patched sources and objects stay on the
    # box; stage-B logs named b_* (b_r1_C2.log) are files and still retained
    assert "--exclude b_*/run " in first and "--exclude b_*/fleet_src/ " in first and "--exclude *.o " in first
    assert "--exclude b_*/ " not in first


def test_reused_fetch_drops_old_provenance_and_mirrors(tmp_path):
    # review 09, R12: a postmortem re-fetch over an earlier capture must not leave that capture's box manifest behind,
    # neither when the transfer fails (no manifest -> NOT VERIFIED) nor when it works (an exact mirror of the new box)
    stub(tmp_path, 'rsync', 'printf "%s\\n" "$*" >> "$HOME/rsync-args"; exit 23')
    env = dict(os.environ, HOME=str(tmp_path), PATH=str(tmp_path / 'bin') + ':' + os.environ['PATH'],
               ETX_FETCH_REUSE='1')
    dest = tmp_path / 'capture'
    dest.mkdir()
    (dest / 'session-old').mkdir()
    for name in ('SHA256SUMS', 'SHA256SUMS.local', 'pairs_l1.json'):
        (dest / name).write_text('old\n')
        (dest / 'session-old' / name).write_text('old\n')       # review 10: a nested earlier session too
    result = run_session(env, 'fetch', 'localhost', str(dest))
    assert result.returncode != 0
    assert not list(dest.rglob('SHA256SUMS*'))
    first = (tmp_path / 'rsync-args').read_text().splitlines()[0]
    assert first.startswith('-az --delete --delete-excluded ')


def test_reused_fetch_through_a_symlinked_destination_drops_old_provenance(tmp_path):
    # review 11, R12: DEST given as a symlink (no trailing slash) to an earlier capture
    stub(tmp_path, 'rsync', 'exit 23')
    env = dict(os.environ, HOME=str(tmp_path), PATH=str(tmp_path / 'bin') + ':' + os.environ['PATH'],
               ETX_FETCH_REUSE='1')
    real = tmp_path / 'real'
    (real / 'session-old').mkdir(parents=True)
    for d in (real, real / 'session-old'):
        (d / 'SHA256SUMS').write_text('old\n')
        (d / 'SHA256SUMS.local').write_text('old\n')
    (tmp_path / 'capture').symlink_to(real)
    result = run_session(env, 'fetch', 'localhost', str(tmp_path / 'capture'))
    assert result.returncode != 0
    assert (tmp_path / 'capture').is_symlink() and not list(real.rglob('SHA256SUMS*'))
