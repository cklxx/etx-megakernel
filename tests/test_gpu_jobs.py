"""Private scheduler conformance via bounded testing probes.

Operational coverage lives in tests/e2e/test_api_jobs.py. Logical GPU lanes here
execute CPU shell commands only; these checks are not GPU qualification.
"""
import pytest

from etx.testing import job_fault_probe, job_runner_probe


def test_status_atomic_replace(tmp_path):
    result = job_runner_probe('atomic-status', tmp_path)
    assert result['old_statuses'] == ['pending']
    assert result['new_statuses'] == ['fail']
    assert result['replacements'] == 1 and result['temporary_files_removed']
    assert result['final_status'] == 'fail'


@pytest.mark.parametrize('lane', ['cpu', 'gpu'])
def test_used_directory_is_refused(tmp_path, lane):
    result = job_runner_probe('occupied-directory', tmp_path / 'logs', lane=lane)
    assert result['returncode'] == 0
    assert 'not empty' in result['error']
    assert result['unchanged']


@pytest.mark.parametrize('kind,error', [
    ('launch-failure', 'OSError'), ('status-failure', 'OSError'),
    ('status-interrupt', 'KeyboardInterrupt'), ('status-exit', 'SystemExit'),
])
def test_scheduler_fault_stops_owned_processes_before_sealing(tmp_path, kind, error):
    result = job_fault_probe(kind, tmp_path)
    assert result['error'] == {'type': error, 'message': 'injected scheduler fault'}
    assert result['launches'] == 1
    assert result['stopped']
    assert result['stopped_before_seal'] == [True]
    assert result['statuses'] == ['fail', 'skipped']
    assert result['sealed_jobs'] == {'first': 'fail', 'second': 'skipped'}
    assert result['verification'] == 3  # Sealed, but explicitly incomplete.
    assert not result['complete'] and result['failure'] == error


def test_failed_final_status_publication_cannot_seal_complete(tmp_path):
    result = job_fault_probe('final-status-failure', tmp_path)
    assert result['error'] == {'type': 'OSError', 'message': 'injected scheduler fault'}
    assert result['stopped'] and result['stopped_before_seal'] == [True]
    assert result['statuses'] == ['ok'] and result['sealed_jobs'] == {'first': 'ok'}
    assert not result['complete'] and result['failure'] == 'OSError'
    assert result['verification'] == 3
    assert any('INCOMPLETE: runner failure: OSError' in line for line in result['verification_messages'])


def test_folded_forward_dependency_is_refused_before_work(tmp_path):
    result = job_fault_probe('folded-forward', tmp_path)
    assert 'later job on its GPU lane' in result['error']
    assert result['pins'] == [0, 0]
    assert result['launches'] == 0 and not result['record_created']


def test_independent_jobs_can_still_fold_to_one_gpu(tmp_path):
    result = job_fault_probe('folded-valid', tmp_path)
    assert result['error'] is None
    assert result['pins'] == [0, 0]
    assert result['launches'] == 0 and not result['record_created']
