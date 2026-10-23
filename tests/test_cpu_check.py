import json
from pathlib import Path
import subprocess
import sys

from etx.tools.cpu_check import fixtures, test_results as parse_results, command_step


def test_external_and_unexpected_skips_are_distinguished(tmp_path):
    xml = tmp_path / 'results.xml'
    xml.write_text('''<testsuites><testsuite>
      <testcase classname="tests.test_ref_numpy" name="test_ref_numpy_matches_hf"><skipped message="checkpoint"/></testcase>
      <testcase classname="tests.test_cpu_numerical" name="test_complete_acceptance_suite"><skipped message="unexpected"/></testcase>
      <testcase classname="tests.test_bench_sim" name="ok"/>
      <testcase classname="tests.test_bench_sim" name="bad"><failure/></testcase>
    </testsuite></testsuites>''')
    result = parse_results(xml)
    assert result['passed'] == 1 and len(result['skipped']) == 2
    assert len(result['unexpected_skips']) == 1 and len(result['failed']) == 1


def test_fixture_artifacts(tmp_path):
    assert fixtures(tmp_path) == 0
    assert json.loads((tmp_path / 'numerical.json').read_text())['passed']
    specs = json.loads((tmp_path / 'operator-specs.json').read_text())
    assert specs['mixtral-8x7b']['up']['grid'][0] == 3584


def test_vllm_source_skip_is_external_but_other_skips_and_failures_are_not(tmp_path):
    xml = tmp_path / 'results.xml'
    xml.write_text('''<testsuites><testsuite>
      <testcase classname="tests.test_vlbest_session" name="test_g0_local_source_baseline"><skipped message="set ETX_VLLM_SOURCE_ROOT to inject a vLLM source checkout"/></testcase>
      <testcase classname="tests.test_vlbest_session" name="test_g0_other"><skipped message="unexpected"/></testcase>
    </testsuite></testsuites>''')
    result = parse_results(xml)
    assert len(result['skipped']) == 2  # External checks remain pending, never counted as passes.
    assert result['passed'] == 0 and result['failed'] == []
    assert result['unexpected_skips'] == [
        {'test': 'tests.test_vlbest_session.test_g0_other', 'reason': 'unexpected'}]
    xml.write_text('''<testsuites><testsuite>
      <testcase classname="tests.test_vlbest_session" name="test_g0_local_source_baseline"><failure message="invalid source baseline"/></testcase>
    </testsuite></testsuites>''')
    result = parse_results(xml)
    assert result['failed'] == ['tests.test_vlbest_session.test_g0_local_source_baseline']
    assert result['skipped'] == [] and result['passed'] == 0


def test_subprocess_failures_keep_full_logs(tmp_path):
    result = command_step('bad', [sys.executable, '-c', 'print("diagnostic"); raise SystemExit(7)'], tmp_path)
    assert result['returncode'] == 7
    assert 'diagnostic' in (tmp_path / 'bad.log').read_text()


def test_formal_missing_dependency_fails(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'TLC_JAR': str(tmp_path / 'absent.jar'), 'LEAN': sys.executable}
    result = subprocess.run(['bash', str(root / 'formal/check.sh'), 'old', str(tmp_path / 'logs')],
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0 and 'Missing TLC_JAR' in result.stderr


def test_matrix_child_failures_propagate(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'LOCAL_MATRIX': 'unsupported:base'}
    result = subprocess.run(['bash', str(root / 'examples/vllm_llm/local_matrix.sh'), str(tmp_path / 'matrix')],
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0 and 'unsupported model' in result.stderr


def test_acceptance_ignores_ambient_test_selection(tmp_path, monkeypatch):
    from etx.tools.cpu_check import pytest_environment
    suite = tmp_path / 'test_example.py'
    suite.write_text('def test_one(): pass\ndef test_two(): pass\n')
    monkeypatch.setenv('PYTEST_ADDOPTS', '-k test_one')
    inventory = tmp_path / 'inventory.json'
    command = [sys.executable, '-m', 'pytest', '-q', '-o', 'addopts=', '-p',
               'etx.tools.pytest_acceptance', f'--acceptance-inventory={inventory}', str(suite)]
    result = subprocess.run(command, env=pytest_environment(), capture_output=True, text=True)
    report = json.loads(inventory.read_text())
    assert result.returncode == 0, result.stdout + result.stderr
    assert report['complete'] and len(report['executed']) == 2
    # Even explicit selection must not silently weaken the inventory gate.
    result = subprocess.run(command + ['-k', 'test_one'], env=pytest_environment(), capture_output=True, text=True)
    report = json.loads(inventory.read_text())
    assert result.returncode != 0 and not report['complete']
    assert len(report['deselected']) == 1


def test_acceptance_rejects_early_exit(tmp_path):
    from etx.tools.cpu_check import pytest_environment
    suite = tmp_path / 'test_example.py'
    suite.write_text('import pytest\ndef test_one(): pytest.exit("stop", returncode=0)\ndef test_two(): pass\n')
    inventory = tmp_path / 'inventory.json'
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p',
        'etx.tools.pytest_acceptance', f'--acceptance-inventory={inventory}', str(suite)],
        env=pytest_environment(), capture_output=True, text=True)
    assert result.returncode != 0
    assert not json.loads(inventory.read_text())['complete']


def test_archive_members_must_match_pinned_inventory(tmp_path):
    import hashlib
    import pytest
    from etx.tools.cpu_check import verify_archive
    member = tmp_path / 'plan.json'
    member.write_text('{"tasks": 1}\n')
    index = tmp_path / 'SHA256SUMS'
    index.write_text(hashlib.sha256(member.read_bytes()).hexdigest() + '  ./plan.json\n')
    pin = hashlib.sha256(index.read_bytes()).hexdigest()
    assert str(member) in verify_archive(tmp_path, pin)
    member.write_text('{"tasks": 999}\n')
    with pytest.raises(ValueError, match='member checksum'):
        verify_archive(tmp_path, pin)
    member.unlink()
    with pytest.raises(ValueError, match='missing file'):
        verify_archive(tmp_path, pin)
    index.write_text('rewritten inventory\n')
    with pytest.raises(ValueError, match='inventory checksum'):
        verify_archive(tmp_path, pin)


def test_archive_rejects_symlink_alias(tmp_path):
    import hashlib
    import pytest
    from etx.tools.cpu_check import verify_archive
    member = tmp_path / 'plan.json'
    member.write_text('{"tasks": 1}\n')
    index = tmp_path / 'SHA256SUMS'
    index.write_text(hashlib.sha256(member.read_bytes()).hexdigest() + '  ./plan.json\n')
    pin = hashlib.sha256(index.read_bytes()).hexdigest()
    assert str(member) in verify_archive(tmp_path, pin)
    alias = tmp_path / 'alias.json'
    alias.symlink_to(member)    # resolves onto a listed member, so a resolved-only inventory misses it
    with pytest.raises(ValueError, match='symlink'):
        verify_archive(tmp_path, pin)
    alias.unlink()
    (tmp_path / 'outside').symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match='symlink'):
        verify_archive(tmp_path, pin)


def test_dependency_failure_preserves_failed_manifest(tmp_path, monkeypatch):
    from etx.tools.validation import cpu_check
    def fail():
        raise ValueError('archive member checksum mismatch')
    monkeypatch.setattr(cpu_check, 'dependencies', fail)
    out = tmp_path / 'acceptance'
    assert cpu_check.run(out) == 1
    report = json.loads((out / 'acceptance.json').read_text())
    assert report['status'] == 'failed' and 'checksum' in report['dependency_error']


def test_matrix_clears_ambient_model_flags(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    shim = tmp_path / 'python-shim'
    shim.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
if sys.argv[1:3] == ['-m', 'etx']:
    with open(os.environ['REVIEW_ENV_RECORD'], 'w') as stream:
        json.dump(dict(os.environ), stream)
    sys.exit(23)
if sys.argv[1:2] == ['-m']:
    sys.exit(0)
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
''')
    shim.chmod(0o755)
    for variant in ('base', 'chain'):
        out = tmp_path / variant
        record = tmp_path / f'{variant}-env.json'
        env = {**os.environ, 'LOCAL_MATRIX': f'qwen3-8b:{variant}', 'ETX_PYTHON': str(shim),
               'REVIEW_ENV_RECORD': str(record), 'ETX_VL_CHAIN': '1', 'ETX_VL_REQUESTS': '4',
               'ETX_LLM_CONFIG': 'wrong-model.json', 'DSL_ABLATE_REVIEW': '1', 'ETX_DEV_RESET': '1'}
        result = subprocess.run(['bash', str(root / 'examples/vllm_llm/local_matrix.sh'), str(out)],
                                env=env, capture_output=True, text=True)
        assert result.returncode != 0  # Deliberately stop before any compiler.
        got = json.loads(record.read_text())
        assert got.get('ETX_VL_CHAIN') == ('1' if variant == 'chain' else None)
        assert 'ETX_VL_REQUESTS' not in got and 'DSL_ABLATE_REVIEW' not in got
        assert 'ETX_DEV_RESET' not in got  # no ETX_VL_ prefix, but read at plan time by model.py
        assert got['ETX_LLM_CONFIG'].endswith('qwen3-8b.json')
        config = json.loads((out / f'qwen3-8b_{variant}/configuration.json').read_text())
        assert config['variant'] == variant
        assert config['overrides'] == ({'ETX_VL_IMPORTED': '1', 'ETX_VL_CHAIN': '1'}
                                       if variant == 'chain' else {'ETX_VL_IMPORTED': '1'})
