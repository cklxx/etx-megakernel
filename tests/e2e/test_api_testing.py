"""Public executor capabilities; none of these results qualify GPU numerics."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import sys

import pytest
from etx.external import ProcessRequest, execute, read_process_result
from etx.testing import scripted_executor, recorded_executor


def request(tmp_path, code='print("hello", flush=True)', **options):
    return ProcessRequest((sys.executable, '-I', '-u', '-c', code), str(tmp_path),
                          options.pop('deadline_s', 5.0), **options)


def test_host_process_has_full_logs_and_bound_evidence(tmp_path):
    spec = request(tmp_path, 'import os; os.write(1, b"out\\xff\\n"); os.write(2, b"err\\n")')
    directory = tmp_path / 'run'
    result = execute(spec, output=directory)
    assert result.backend == 'host-process'
    assert result.status == 'complete'
    assert result.returncode == 0
    assert (directory / 'output.log').read_bytes() == b'out\xff\nerr\n'
    assert read_process_result(directory, request=spec) == result
    assert json.loads((directory / 'request.json').read_text()) == spec.to_dict()
    assert 'END status=complete' in (directory / 'supervisor.log').read_text()
    with pytest.raises(FileExistsError):
        execute(spec, output=directory)
    with pytest.raises(ValueError, match='device qualification'):
        replace(result, backend='gpu')


@pytest.mark.parametrize('code,status,rc', [
    ('raise SystemExit(7)', 'failed', 7),
    ('import time; print("before", flush=True); time.sleep(10)', 'deadline', 124),
])
def test_failure_deadline_and_silence_are_not_success(tmp_path, code, status, rc):
    spec = request(tmp_path, code, deadline_s=0.3, silence_s=0.05)
    directory = tmp_path / 'run'
    result = execute(spec, output=directory)
    assert (result.status, result.returncode) == (status, rc)
    assert read_process_result(directory, request=spec) == result
    if status == 'deadline':
        assert (directory / 'output.log').read_text() == 'before\n'
        assert any('SILENCE' in event for event in result.events)


def test_explicit_environment_and_cwd(tmp_path):
    spec = request(tmp_path, 'import os; print(os.environ["ETX_TEST_VALUE"]); print(os.getcwd())',
                   environment={'ETX_TEST_VALUE': 'declared'})
    execute(spec, output=tmp_path / 'run')
    assert (tmp_path / 'run' / 'output.log').read_text().splitlines() == ['declared', str(tmp_path)]


def test_concurrent_calls_do_not_sweep_each_others_children(tmp_path):
    slow = request(tmp_path, 'import time; time.sleep(0.4); print("slow survived")')
    fast = request(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(execute, slow, output=tmp_path / 'slow')
        second = pool.submit(execute, fast, output=tmp_path / 'fast')
        assert first.result().status == second.result().status == 'complete'
    assert (tmp_path / 'slow' / 'output.log').read_text() == 'slow survived\n'


@pytest.mark.skipif(sys.platform != 'linux', reason='escaped descendant ownership requires Linux subreaper')
def test_deadline_reaps_escaped_descendants(tmp_path):
    code = '''import subprocess, sys, time
child = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(30)'], start_new_session=True)
print(child.pid, flush=True)
time.sleep(30)
'''
    result = execute(request(tmp_path, code, deadline_s=0.5), output=tmp_path / 'tree')
    assert result.status == 'deadline'
    pid = int((tmp_path / 'tree' / 'output.log').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_scripted_and_recorded_executors_never_launch_the_command(tmp_path):
    spec = ProcessRequest(('/not/a/real/executable',), str(tmp_path), 1)
    backend = scripted_executor(spec, output=b'recorded bytes\n')
    first = execute(spec, output=tmp_path / 'scripted', executor=backend)
    assert first.backend == 'scripted'
    second = execute(spec, output=tmp_path / 'replay', executor=recorded_executor(tmp_path / 'scripted'))
    assert second.backend == 'recorded'
    assert second.output_sha256 == first.output_sha256
    with pytest.raises(ValueError, match='exhausted'):
        execute(spec, output=tmp_path / 'exhausted', executor=backend)
    assert not (tmp_path / 'exhausted' / 'result.json').exists()


def test_scripted_request_mismatch_is_not_consumed(tmp_path):
    spec = request(tmp_path)
    backend = scripted_executor(spec)
    with pytest.raises(ValueError, match='request mismatch'):
        execute(replace(spec, deadline_s=1), output=tmp_path / 'wrong', executor=backend)
    assert execute(spec, output=tmp_path / 'right', executor=backend).status == 'complete'


@pytest.mark.parametrize('field', ['request', 'output', 'missing-result'])
def test_replay_revalidates_saved_evidence(tmp_path, field):
    spec = request(tmp_path)
    directory = tmp_path / 'source'
    execute(spec, output=directory, executor=scripted_executor(spec, output=b'original'))
    if field == 'request':
        data = spec.to_dict()
        data['deadline_s'] += 1
        (directory / 'request.json').write_text(json.dumps(data))
    elif field == 'output':
        (directory / 'output.log').write_bytes(b'changed')
    else:
        (directory / 'result.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        execute(spec, output=tmp_path / 'replay', executor=recorded_executor(directory))
    assert not (tmp_path / 'replay' / 'result.json').exists()


def test_spawn_error_is_durable_not_complete(tmp_path):
    spec = ProcessRequest(('/not/a/real/executable',), str(tmp_path), 1)
    result = execute(spec, output=tmp_path / 'error')
    assert result.status == 'error'
    assert result.returncode is None
    assert any('FileNotFoundError' in event for event in result.events)
    assert read_process_result(tmp_path / 'error', request=spec) == result


@pytest.mark.parametrize('change', [
    {'argv': 'not-an-argv'}, {'argv': ('relative',)}, {'cwd': 'relative'},
    {'deadline_s': 0}, {'deadline_s': True}, {'deadline_s': float('inf')},
    {'silence_s': -1}, {'environment': {'A': 1}}, {'environment': {'A=B': 'x'}},
    {'schema_version': True},
])
def test_process_request_refuses_ambiguous_inputs(tmp_path, change):
    with pytest.raises(ValueError):
        replace(request(tmp_path), **change)
