"""Internal sealed-session conformance; operational coverage lives in E2E.

All subprocess probes are explicitly synthetic CPU work, never GPU evidence.
"""
import hashlib
import time

import pytest
from etx.testing import (sealed_child_probe, sealed_resume_probe,
                         sealed_dependency_probe, sealed_provenance_probe,
                         sealed_cleanup_interrupt_probe)


def test_cpu_subprocess_keeps_unfiltered_stdout_stderr_and_refuses_overwrite(tmp_path):
    log = tmp_path / 'synthetic-child.log'
    assert sealed_child_probe('stdout-stderr', log=log) == 0
    assert set(log.read_text().splitlines()) == {'SYNTHETIC stdout', 'SYNTHETIC stderr'}
    with pytest.raises(FileExistsError):
        sealed_child_probe('stdout-stderr', log=log)


def test_deadline_kills_own_cpu_process_group(tmp_path):
    marker = tmp_path / 'grandchild-must-not-survive'
    with pytest.raises(TimeoutError, match='no performance verdict'):
        sealed_child_probe('deadline', log=tmp_path / 'deadline.log', marker=marker)
    time.sleep(.35)
    assert not marker.exists()
    assert 'SYNTHETIC CPU' in (tmp_path / 'deadline.log').read_text()


def test_first_interrupt_during_cleanup_still_sweeps_cpu_descendants(tmp_path):
    marker = tmp_path / 'grandchild-must-not-survive-cleanup-interrupt'
    result = sealed_cleanup_interrupt_probe(log=tmp_path / 'cleanup.log', marker=marker)
    assert result['response']['error']['type'] == 'KeyboardInterrupt'
    assert 'interrupted by signal' in result['response']['error']['message']
    assert result['descendant_survived'] is False
    assert not marker.exists()


def test_resume_rejects_changed_sealed_bytes_and_incomplete_state(tmp_path):
    log = tmp_path / 'raw.log'
    log.write_text('SYNTHETIC initial')
    state = dict(config_sha256='c', source_revision='r', status='COMPLETE',
                 sealed={str(log): hashlib.sha256(log.read_bytes()).hexdigest()})
    sealed_resume_probe(state, 'c', 'r')
    with pytest.raises(ValueError, match='identity'):
        sealed_resume_probe(state, 'changed', 'r')
    log.write_text('SYNTHETIC changed')
    with pytest.raises(ValueError, match='sealed'):
        sealed_resume_probe(state, 'c', 'r')
    for status in ('RUNNING', 'FAILED', 'NOT_COMPLETE'):
        with pytest.raises(ValueError, match='ambiguous'):
            sealed_resume_probe({**state, 'sealed': {}, 'status': status}, 'c', 'r')


def test_dependency_and_optional_decision_gates_are_independent():
    assert not sealed_dependency_probe('G05', {}, {})
    assert sealed_dependency_probe('G05', {'G04': {'complete': True}}, {})
    assert not sealed_dependency_probe('G03', {'G02': {'admitted': False}}, {})
    assert sealed_dependency_probe('G03', {'G02': {'admitted': True}}, {})
    assert not sealed_dependency_probe('G13', {'G02': {'admitted': True}}, {})
    assert sealed_dependency_probe('G13', {'G02': {'admitted': True}}, {'coalescing_decision_active': True})
    assert not sealed_dependency_probe('G11', {'G12': {'complete': True}}, {})
    assert sealed_dependency_probe('G11', {'G12': {'complete': True}}, {'fold_decision_active': True})
    assert not sealed_dependency_probe('G12', {'G09': {'admitted': False}}, {})
    assert sealed_dependency_probe('G12', {'G09': {'admitted': True}}, {})


def test_non_git_provenance_is_unknown_not_clean_and_seals_content(tmp_path):
    sources = {'synthetic-source.txt': 'a' * 64}
    first = sealed_provenance_probe(sources, root=tmp_path)
    assert first['commit'] is None and first['dirty'] is None
    assert first['revision_origin'] == 'unknown'
    assert first['inventory_sha256'] == sealed_provenance_probe(sources, root=tmp_path)['inventory_sha256']
    changed = sealed_provenance_probe({'synthetic-source.txt': 'b' * 64}, root=tmp_path)
    assert changed['inventory_sha256'] != first['inventory_sha256']
    declared = sealed_provenance_probe(sources, root=tmp_path, revision='opaque-release-id')
    assert declared['commit'] == 'opaque-release-id'
    assert declared['dirty'] is None and declared['revision_origin'] == 'declared'
    assert declared['inventory_sha256'] == first['inventory_sha256']
