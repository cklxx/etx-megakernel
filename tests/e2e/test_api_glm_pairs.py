"""T10 certificate CLI through the external process API; synthetic inputs, no GPU."""
import json
from pathlib import Path
import sys

import pytest
from etx.external import ProcessRequest, execute

ROOT = Path(__file__).resolve().parents[2]


def cert(tmp_path, name, build, ms, **change):
    value = dict(status='PASS', batch=1, steps=31, median_decode_wall_ms=ms, tokens_matched=32, tokens_expected=32,
                 qualification=dict(status='TOKENS-EQUAL'),
                 build_identity=dict(options_sha256=build, binary_sha256=build * 2),
                 conditions=dict(gpus=2, gpu='AMD Instinct MI300X', gpu_arch='gfx942', mode='pp' if build == 'r' else 'ffn-tp',
                                 context=1024, generated=32, layers=45))
    for key, v in change.items():
        if key in value['conditions']:
            value['conditions'][key] = v
        else:
            value[key] = v
    path = tmp_path / f'{name}.json'
    path.write_text(json.dumps(value))
    return path


def cli(tmp_path, paths, *args):
    output = tmp_path / f'command-{len(list(tmp_path.glob("command-*")))}'
    request = ProcessRequest((sys.executable, '-m', 'examples.glm53.pairs',
                              *(str(p) for p in paths), *args), str(ROOT), 20)
    result = execute(request, output=output)
    assert result.backend == 'host-process'
    assert result.status in ('complete', 'failed'), (output / 'output.log').read_text()
    return result.returncode


def run(tmp_path, values):
    paths = []
    for i, (r, c) in enumerate(values):
        a = cert(tmp_path, f'r{i}', 'r', r); b = cert(tmp_path, f'c{i}', 'c', c)
        paths += [a, b] if i % 2 == 0 else [b, a]      # counterbalanced order
    code = cli(tmp_path, paths, '--out', str(tmp_path / 'out.json'))
    return code, json.loads((tmp_path / 'out.json').read_text())


def test_keep_needs_all_positive_median_above_2pct_and_twice_spread(tmp_path):
    code, out = run(tmp_path, [(30.0, 29.0)] * 4 + [(30.0, 28.9)])
    assert code == 0 and out['verdict'] == 'KEEP' and out['median_gain_pct'] == pytest.approx(100 / 30)
    assert out['reference_mode'] == 'pp' and out['candidate_mode'] == 'ffn-tp'


@pytest.mark.parametrize('values', [
    [(30.0, 29.0)] * 4 + [(30.0, 30.1)],                     # one nonpositive pair
    [(30.0, 29.7)] * 5,                                       # median 1% <= 2%
    [(30.0, 29.0), (30.0, 27.0), (30.0, 29.0), (30.0, 25.0), (30.0, 29.0)],  # median below twice the spread
])
def test_no_established_gain(tmp_path, values):
    code, out = run(tmp_path, values)
    assert code == 1 and out['verdict'] == 'NO-ESTABLISHED-GAIN'


@pytest.mark.parametrize('change', [dict(status='FAIL'), dict(tokens_matched=31), dict(gpu=None),
                                    dict(qualification=dict(status='NOT-COMPLETE'))])
def test_incomplete_or_uncorrected_runs_refuse(tmp_path, change):
    paths = [cert(tmp_path, 'r0', 'r', 30.0), cert(tmp_path, 'c0', 'c', 29.0, **change)]
    assert cli(tmp_path, paths, '--pairs', '1') == 2


@pytest.mark.parametrize('change', [dict(gpu='AMD Instinct MI325X'), dict(gpus=8), dict(batch=2), dict(context=512)])
def test_mixed_conditions_refuse(tmp_path, change):
    paths = [cert(tmp_path, 'r0', 'r', 30.0), cert(tmp_path, 'c0', 'c', 29.0, **change)]
    assert cli(tmp_path, paths, '--pairs', '1') == 2


def test_wrong_pair_count_or_same_build_pair_refuses(tmp_path):
    r = [cert(tmp_path, f'r{i}', 'r', 30.0) for i in range(2)]
    c = [cert(tmp_path, f'c{i}', 'c', 29.0) for i in range(2)]
    assert cli(tmp_path, (r[0], c[0])) == 2                       # 1 pair, 5 expected
    assert cli(tmp_path, (r[0], r[1], c[0], c[1]), '--pairs', '2') == 2
