"""CPU acceptance lifecycle via the external API and bounded CPU-only probes."""
import json
import os
import sys

from etx.external import cpu_acceptance_fixtures, run_cpu_acceptance
from etx.testing import acceptance_dependency_failure, acceptance_probe


def test_fixture_artifacts(tmp_path):
    assert cpu_acceptance_fixtures(tmp_path) == 0
    assert json.loads((tmp_path / 'numerical.json').read_text())['passed']
    specs = json.loads((tmp_path / 'operator-specs.json').read_text())
    assert specs['mixtral-8x7b']['up']['grid'][0] == 3584


def test_subprocess_failures_keep_full_logs(tmp_path):
    result = acceptance_probe('command-failure', output=tmp_path).to_dict()
    assert result['returncode'] == 7
    assert 'diagnostic' in (tmp_path / result['log']).read_text()


def test_formal_missing_dependency_fails(tmp_path):
    # This case tests the missing JAR gate independently of the host Java install.
    # The sentinel must never execute: the missing asset is refused first.
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    java = binaries / 'java'
    java.write_text('#!/bin/sh\nprintf "unexpected-java-execution\\n" >&2\nexit 99\n')
    java.chmod(0o755)
    env = {'TLC_JAR': str(tmp_path / 'absent.jar'), 'LEAN': sys.executable,
           'PATH': str(binaries) + os.pathsep + os.environ.get('PATH', '')}
    result = acceptance_probe('formal', output=tmp_path, env=env).to_dict()
    log = (tmp_path / result['log']).read_text()
    assert result['returncode'] != 0 and 'Missing TLC_JAR' in log
    assert 'unexpected-java-execution' not in log


def test_matrix_child_failures_propagate(tmp_path):
    result = acceptance_probe('matrix', output=tmp_path,
                              env={'LOCAL_MATRIX': 'unsupported:base'}).to_dict()
    assert result['returncode'] != 0
    assert 'unsupported model' in (tmp_path / result['log']).read_text()


def test_acceptance_ignores_ambient_test_selection(tmp_path, monkeypatch):
    suite = tmp_path / 'test_example.py'
    suite.write_text('def test_one(): pass\ndef test_two(): pass\n')
    monkeypatch.setenv('PYTEST_ADDOPTS', '-k test_one')
    monkeypatch.setenv('PYTEST_PLUGINS', 'missing_ambient_plugin')
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '0')
    result = acceptance_probe('pytest', output=tmp_path, suite=suite).to_dict()
    report = json.loads((tmp_path / 'inventory.json').read_text())
    assert result['returncode'] == 0, (tmp_path / result['log']).read_text()
    assert report['complete'] and len(report['executed']) == 2
    # Even explicit selection must not silently weaken the inventory gate.
    result = acceptance_probe('pytest', output=tmp_path, suite=suite,
                              selection=('-k', 'test_one')).to_dict()
    report = json.loads((tmp_path / 'inventory.json').read_text())
    assert result['returncode'] != 0 and not report['complete']
    assert len(report['deselected']) == 1


def test_acceptance_rejects_early_exit(tmp_path):
    suite = tmp_path / 'test_example.py'
    suite.write_text('import pytest\ndef test_one(): pytest.exit("stop", returncode=0)\ndef test_two(): pass\n')
    result = acceptance_probe('pytest', output=tmp_path, suite=suite).to_dict()
    assert result['returncode'] != 0
    assert not json.loads((tmp_path / 'inventory.json').read_text())['complete']


def test_dependency_failure_preserves_failed_manifest(tmp_path):
    out = tmp_path / 'acceptance'
    with acceptance_dependency_failure():
        assert run_cpu_acceptance(out) == 1
    report = json.loads((out / 'acceptance.json').read_text())
    assert report['status'] == 'failed' and 'checksum' in report['dependency_error']
    assert not report['gpu_executed'] and not report['cpu_core_passed']
    assert not report['all_external_checks_passed'] and not report['steps']
    assert 'checksum' in (out / 'README.md').read_text()


def test_matrix_clears_ambient_model_flags(tmp_path):
    shim = tmp_path / 'python-shim'
    shim.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
if sys.argv[1:3] == ['-m', 'etx']:
    # Record only the tested toolchain/model variables, never inherited secrets.
    keys = ('ETX_VL_CHAIN', 'ETX_VL_REQUESTS', 'DSL_ABLATE_REVIEW', 'ETX_DEV_RESET',
            'ETX_LLM_CONFIG', 'ETX_TEST_ACCEPTANCE_INHERITED')
    with open(os.environ['REVIEW_ENV_RECORD'], 'w') as stream:
        json.dump({key: os.environ[key] for key in keys if key in os.environ}, stream)
    sys.exit(23)
if sys.argv[1:2] == ['-m']:
    sys.exit(0)
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
''')
    shim.chmod(0o755)
    for variant in ('base', 'chain'):
        out = tmp_path / variant
        record = tmp_path / f'{variant}-env.json'
        env = {'LOCAL_MATRIX': f'qwen3-8b:{variant}', 'ETX_PYTHON': str(shim),
               'REVIEW_ENV_RECORD': str(record), 'ETX_VL_CHAIN': '1', 'ETX_VL_REQUESTS': '4',
               'ETX_LLM_CONFIG': 'wrong-model.json', 'DSL_ABLATE_REVIEW': '1', 'ETX_DEV_RESET': '1',
               'ETX_TEST_ACCEPTANCE_INHERITED': 'preserved'}
        result = acceptance_probe('matrix', output=out, env=env).to_dict()
        assert result['returncode'] != 0  # Deliberately stop before any compiler.
        got = json.loads(record.read_text())
        assert got.get('ETX_VL_CHAIN') == ('1' if variant == 'chain' else None)
        assert 'ETX_VL_REQUESTS' not in got and 'DSL_ABLATE_REVIEW' not in got
        assert 'ETX_DEV_RESET' not in got  # no ETX_VL_ prefix, but read at plan time by model.py
        assert got['ETX_LLM_CONFIG'].endswith('qwen3-8b.json')
        assert got['ETX_TEST_ACCEPTANCE_INHERITED'] == 'preserved'
        config = json.loads((out / f'matrix/qwen3-8b_{variant}/configuration.json').read_text())
        assert config['variant'] == variant
        assert config['overrides'] == ({'ETX_VL_IMPORTED': '1', 'ETX_VL_CHAIN': '1'}
                                       if variant == 'chain' else {'ETX_VL_IMPORTED': '1'})
