"""Real protocol experiment persistence/replay only through the external API."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
from etx.external import run_protocol_experiment, report_protocol_experiment, ProcessRequest, execute
from etx.testing import protocol_fault_executor

CASE = {'name': 'small', 'workload': 'splitk', 'size': 2, 'duration_cv': 0.1}
CONFIG = {'schema_version': 1, 'arch': 'gfx942', 'rounds': 2, 'seed': 7, 'cases': [CASE]}
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def experiment(tmp_path):
    output = tmp_path / 'run'
    assert run_protocol_experiment(CONFIG, output=output) == 0
    return output


def rewrite_samples(output, samples):
    path = output / 'samples.jsonl'
    path.write_text(''.join(json.dumps(sample) + '\n' for sample in samples))
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['artifacts'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest))


def test_recompute_is_exact_and_rounds_are_paired(experiment):
    before = (experiment / 'summary.json').read_bytes()
    samples = [json.loads(line) for line in (experiment / 'samples.jsonl').read_text().splitlines()]
    assert [sample['mode'] for sample in samples] == ['static', 'dynamic', 'hybrid', 'dynamic', 'hybrid', 'static']
    assert [sample['seed'] for sample in samples] == [7, 7, 7, 8, 8, 8]
    assert report_protocol_experiment(experiment) == 0
    assert (experiment / 'summary.json').read_bytes() == before
    manifest = json.loads((experiment / 'manifest.json').read_text())
    assert manifest['tensor_correctness'] == 'not_executed'
    assert not manifest['gpu_measurement']
    assert manifest['provenance']['source_sha256']['etx/sim/protocol.py']
    if manifest['provenance'].get('kind') == 'scoped-filesystem-snapshot':
        assert manifest['provenance']['commit'] is None
        assert manifest['provenance']['dirty'] is None


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'seed', 'trace', 'failure', 'inflated'])
def test_offline_report_rejects_invalid_pairs(experiment, fault):
    samples = [json.loads(line) for line in (experiment / 'samples.jsonl').read_text().splitlines()]
    if fault == 'missing':
        samples.pop()
    elif fault == 'duplicate':
        samples.append(samples[0])
    elif fault == 'seed':
        samples[0]['seed'] += 1
    elif fault == 'trace':
        samples[0]['result']['segments'] = []
    elif fault == 'failure':
        samples[0]['errors'] = ['injected failure']
    else:
        samples[0]['result']['makespan_us'] *= 100
    rewrite_samples(experiment, samples)
    assert report_protocol_experiment(experiment) == 1
    summary = json.loads((experiment / 'summary.json').read_text())
    assert not summary['valid'] and not summary['comparisons']


def test_corrupt_artifact_invalidates_previous_success_through_cli(experiment, tmp_path):
    with (experiment / 'samples.jsonl').open('a') as stream:
        stream.write('\n')
    result = execute(ProcessRequest((sys.executable, '-m', 'etx.tools.bench_sim', 'report', str(experiment)),
                                    str(ROOT), 20, environment={'PATH': '/usr/bin:/bin'}),
                     output=tmp_path / 'command')
    assert result.returncode == 1
    assert 'BENCH FAILED:' in (tmp_path / 'command/output.log').read_text()
    assert not json.loads((experiment / 'summary.json').read_text())['valid']


def test_refuse_overwrite(experiment):
    before = (experiment / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        run_protocol_experiment(CONFIG, output=experiment)
    assert (experiment / 'manifest.json').read_bytes() == before


def test_bad_configuration_rejected_before_output(tmp_path):
    config = copy.deepcopy(CONFIG)
    config['cases'][0]['size'] = 0
    with pytest.raises(ValueError, match='size'):
        run_protocol_experiment(config, output=tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('fault', ['exception', 'nonfinite'])
def test_executor_faults_are_retained_and_fail_closed(tmp_path, fault):
    output = tmp_path / 'failed'
    assert run_protocol_experiment(CONFIG, output=output, executor=protocol_fault_executor(fault)) == 1
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary['failures']) == 6 and not summary['comparisons']
    raw = (output / 'samples.jsonl').read_text()
    if fault == 'exception':
        assert 'injected simulator failure' in raw
    else:
        assert 'Infinity' not in raw and 'NaN' not in raw
        samples = [json.loads(line) for line in raw.splitlines()]
        assert all(sample['result']['makespan_us'] is None and sample['errors'] for sample in samples)


def test_protocol_cli_and_api_have_identical_samples(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(CONFIG))
    api, cli = tmp_path / 'api', tmp_path / 'cli'
    assert run_protocol_experiment(CONFIG, output=api) == 0
    result = execute(ProcessRequest((sys.executable, '-m', 'etx.tools.bench_sim', 'run', '--config', str(config),
                                     '--out', str(cli)), str(ROOT), 30, environment={'PATH': '/usr/bin:/bin'}),
                     output=tmp_path / 'command')
    assert result.returncode == 0
    for name in ('plans.json', 'samples.jsonl', 'summary.json'):
        assert (api / name).read_bytes() == (cli / name).read_bytes()
