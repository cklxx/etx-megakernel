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
