"""Benchmark certificate conditions come from the timed log, never from constants."""
from pathlib import Path

import pytest
from examples.glm53 import bench
from examples.glm53.options import Options
from examples.glm53.prepare import metadata


def device_log(devices=8, parallel=1, attention_tp=1, up_tail_balance=0, up_pairs=36,
               names=None, ranks=None, arch='gfx942:sramecc+:xnack-'):
    names = names or ['AMD Instinct MI325X'] * devices
    ranks = list(range(devices)) if ranks is None else ranks
    lines = [f'GLM-VARIANT devices={devices} parallel={parallel} attention_tp={attention_tp} '
             f'up_tail_balance={up_tail_balance} up_pairs={up_pairs}']
    lines += [f'GLM-DEVICE rank={r} name={names[i]} arch={arch} free=1 payload_and_state=1'
              for i, r in enumerate(ranks)]
    return '\n'.join(lines) + '\n'


MODES = [(0, 0, 'pp'), (1, 0, 'ffn-tp'), (2, 0, 'ep'), (1, 1, 'full-tp'), (2, 1, 'head-tp-ep')]


@pytest.mark.parametrize('devices', [2, 4, 8])
@pytest.mark.parametrize('parallel,attention_tp,mode', MODES)
def test_evidence_reports_actual_world_gpu_and_mode(devices, parallel, attention_tp, mode):
    options = Options(devices=devices, parallel=parallel, attention_tp=attention_tp).to_dict()
    got = bench.device_evidence(device_log(devices, parallel, attention_tp), options)
    assert got['gpus'] == devices and got['gpu'] == 'AMD Instinct MI325X' and got['mode'] == mode
    assert got['gpu_arch'].startswith('gfx942') and got['effective_up_pairs'] == 36


def test_up_tail_balance_must_print_effective_pairs():
    options = Options(devices=8, parallel=2, up_tail_balance=1).to_dict()
    assert bench.device_evidence(device_log(8, 2, 0, 1, 32), options)['effective_up_pairs'] == 32
    with pytest.raises(bench.Refusal, match='differs from the build options'):
        bench.device_evidence(device_log(8, 2, 0, 1, 36), options)


@pytest.mark.parametrize('text,match', [
    (device_log(devices=2, parallel=1), 'differs from the build options'),
    (device_log(ranks=[0, 1]), 'GLM-DEVICE ranks'),
    (device_log(ranks=[0, 1, 2, 3, 4, 5, 6, 6]), 'GLM-DEVICE ranks'),
    (device_log(names=['AMD Instinct MI325X'] * 7 + ['AMD Instinct MI300X']), 'not uniform'),
    (device_log(names=[''] * 8), 'not uniform or not reported'),
    (device_log() * 2, 'expected exactly one'),
    ('GLM-MODEL-STEP pos=0 batch=1 ms=1\n', 'expected exactly one'),
])
def test_missing_or_contradictory_evidence_refuses(text, match):
    with pytest.raises(bench.Refusal, match=match):
        bench.device_evidence(text, Options(devices=8, parallel=1, attention_tp=1).to_dict())


def certificate(evidence, batch=8):
    staged = dict(config=metadata()[0]['text_config'], dimensions=dict(MAXLEN=1072))
    receipt = dict(status='TOKENS-EQUAL', repetitions=2, matched_each=[32] * batch)
    return bench.certificate(staged, Path('/tmp/build'), dict(status='GPU-LINKED'), batch,
                             [1.0] * 31, receipt, dict(schema=1), evidence=evidence)


@pytest.mark.parametrize('batch', [1, 2, 4, 8])
def test_certificate_carries_eight_gpu_tp_conditions(batch):
    options = Options(devices=8, parallel=1, attention_tp=1).to_dict()
    cert = certificate(bench.device_evidence(device_log(), options), batch)
    c = cert['conditions']
    assert c['gpus'] == 8 and c['gpu'] == 'AMD Instinct MI325X' and c['mode'] == 'full-tp'
    assert 'pipeline' not in c and 'MI300X' not in str(c) and cert['batch'] == batch


def test_certificate_without_evidence_assumes_nothing():
    c = certificate(None, 1)['conditions']
    assert c['gpus'] is None and c['gpu'] is None and c['mode'] is None
    assert 'pipeline' not in c and 'MI300X' not in str(c)
