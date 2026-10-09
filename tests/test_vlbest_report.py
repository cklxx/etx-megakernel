"""Synthetic search evidence; no vLLM, GPU, image or network required."""
import json
from pathlib import Path
import hashlib
import sys

import pytest

from examples.vllm_llm import vlbest_report as r
from examples.vllm_llm.pairs import Refusal


def log(arm='S0', ms=5, cfg=None):
    cfg = cfg or r.config(arm)
    env = {k: v.replace('$' + 'HOME', '/synthetic') for k, v in cfg['env'].items()}
    echo = dict(async_scheduling=True, cudagraph_mode='FULL_AND_PIECEWISE', pass_config={},
                kernel_config={}, attention_backend=cfg['kw'].get('attention_backend'), env=env)
    lines = [r.CONFIG_PREFIX + json.dumps(echo), 'VLLM-SAMPLING detok=0']
    if 'attention_backend' in cfg['kw']:
        lines += [f"Using {cfg['kw']['attention_backend']} backend (selected via --attention-backend)"]
    lines += ['Enabled custom fusions: qk_norm_rope_kvcache',
              'QK-Norm+RoPE+KVCache fusion: replaced 48 pattern(s)',
              'Capturing CUDA graphs (decode, FULL)', 'Using TritonExperts MoE backend', "rms_norm=['native']"]
    if 'VLLM_TUNED_CONFIG_FOLDER' in env:
        lines += [f"Using configuration from {env['VLLM_TUNED_CONFIG_FOLDER']}/E=128,N=768,device_name=AMD_Instinct_MI300X.json for MoE layer"]
    lines += [f'VLLM-SAMPLE sample={i} first_divergence=32/32' for i in range(5)]
    lines += [f"VLLM label={arm} aiter={env.get('VLLM_ROCM_USE_AITER', '0')} batch=1 version=0.27.1 context=1024 first_divergence=32/32 median_ms={ms}"]
    return '\n'.join(lines)


@pytest.mark.parametrize('arm', [f'S{i}' for i in range(10)])
def test_proven(arm):
    assert r.observation(log(arm), r.config(arm)) == 5


@pytest.mark.parametrize('arm,remove', [
    ('S0', '"async_scheduling": true, '),
    ('S4', 'Using ROCM_AITER_FA backend (selected via --attention-backend)'),
    ('S4', 'Enabled custom fusions: qk_norm_rope_kvcache'),
    ('S4', 'QK-Norm+RoPE+KVCache fusion: replaced 48 pattern(s)'),
    ('S4', 'Capturing CUDA graphs (decode, FULL)'),
    ('S5', 'Using TritonExperts MoE backend'),
    ('S5', "rms_norm=['native']"),
    ('S5', '"VLLM_ROCM_USE_AITER_TRITON_GEMM": "0", '),
    ('S1', 'Using configuration from'),
    ('S9', '/moecfg/tuned/'),
    ('S8', 'VLLM-SAMPLING detok=0'),
])
def test_missing_proof(arm, remove):
    text = log(arm)
    assert remove in text
    with pytest.raises(Refusal, match='NOT-APPLIED'):
        r.observation(text.replace(remove, ''), r.config(arm))


@pytest.mark.parametrize('arm,extra', [('S7', '(EngineCore pid=123)'), ('S1', 'Using default MoE config')])
def test_contradictory_proof(arm, extra):
    with pytest.raises(Refusal, match='NOT-APPLIED'):
        r.observation(log(arm) + '\n' + extra, r.config(arm))


@pytest.mark.parametrize('bad', ['31/32', '31/31', '32/33', '0/32'])
def test_every_sample(bad):
    text = log().replace('sample=0 first_divergence=32/32', 'sample=0 first_divergence=' + bad)
    with pytest.raises(Refusal, match='failed 32/32'):
        r.observation(text, r.config('S0'))


def test_missing_sample():
    with pytest.raises(Refusal, match='failed 32/32'):
        r.observation(log().replace('VLLM-SAMPLE sample=0 first_divergence=32/32', ''), r.config('S0'))


@pytest.mark.parametrize('values,brackets,want', [
    ([4.95, 4.95], [[5, 5], [5, 5]], 'WINNER'),
    ([4.94, 5], [[5, 5], [5, 5]], 'NO-WINNER'),
    ([4.951, 4.951], [[5, 5], [5, 5]], 'NO-WINNER'),
    ([4.8, 4.8], [[5, 5.051], [5, 5]], 'INSUFFICIENT'),
    ([4.8, 4.8], [[5, 5.05], [5, 5]], 'WINNER'),
    ([4.8], [[5, 5]], 'INSUFFICIENT'),
])
def test_screening(values, brackets, want):
    assert r.screening(values, brackets)[0] == want


def test_brackets_and_exclusion(tmp_path):
    for rnd in (1, 2):
        for end in ('open', 'close'):
            name = f'a_r{rnd}_S0_{end}'
            (tmp_path / f'{name}.log').write_text(f'# job {name} gpu=0\n' + log())
        name = f'a_r{rnd}_S4'
        (tmp_path / f'{name}.log').write_text(f'# job {name} gpu=0\n' + log('S4', 1).replace('replaced 48', 'replaced 47'))
    result = r.screen(tmp_path, candidates=[r.config('S4')])
    assert not result['winners']
    assert result['tested'] == ['S0']
    assert 'NOT-APPLIED' in result['excluded']['S4']


def winner(arm, gain):
    return dict(config=r.config(arm), gain=gain, deltas=[gain, gain])


def test_combination():
    assert r.combination({'S1': winner('S1', -.1)}) is None
    cfg = r.combination({a: winner(a, g) for a, g in [('S1', -.1), ('S9', -.2), ('S4', -.3), ('S8', -.06)]})
    assert cfg['components'] == ['S4', 'S8', 'S9']
    assert cfg['env']['VLLM_ROCM_USE_AITER_MOE'] == '0'
    assert cfg['env']['VLLM_TUNED_CONFIG_FOLDER'].endswith('/tuned')
    assert '--no-detok' in cfg['flags']
    assert r.observation(log('C', cfg=cfg), cfg) == 5


def test_finalist():
    assert r.finalist({})['id'] == 'S0'
    assert r.finalist({a: winner(a, g) for a, g in [('S1', -.1), ('C', -.2)]})['id'] == 'C'


def test_reference():
    assert r.reference([5]*5, [4.999]*5, 'S5') == 'S5'
    assert r.reference([5]*5, [5.001]*5, 'S5') == 'S0'
    assert r.reference([5]*5, [5]*5, 'S5') == 'S0'
    with pytest.raises(Refusal, match='five confirmation'):
        r.reference([5]*4, [4]*5, 'S5')


def test_paired_resolution():
    assert r.paired([-.1]*5) == 'LEAD'
    assert r.paired([.1]*5) == 'LOSS'
    assert r.paired([-.01]*5) == 'TIE'
    assert r.paired([-.01, -.02, -.1, -.1, -.15]) == 'INSUFFICIENT'


def test_label():
    with pytest.raises(Refusal, match='label=S4'):
        r.observation(log('S4').replace('label=S4', 'label=default'), r.config('S4'))


def test_harness_flags_and_all_samples(tmp_path, monkeypatch, capsys):
    import types
    from examples.llm import vllm_bench as bench
    calls, sampling = [], []
    cc = types.SimpleNamespace(cudagraph_mode='FULL', pass_config={})
    cfg = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(async_scheduling=True),
                                compilation_config=cc, kernel_config={}, attention_config=types.SimpleNamespace(backend='TEST'))
    class LLM:
        def __init__(self, **kw):
            calls.append(kw)
            self.llm_engine = types.SimpleNamespace(vllm_config=cfg)
            self.n = 0
        def generate(self, prompts, sp, **kw):
            self.n += 1
            # The first measured sample fails; the last succeeds.
            toks = [9 if self.n == 3 else 1] * sp['max_tokens']
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=toks)])]
    def params(**kw):
        sampling.append(kw)
        return kw
    monkeypatch.setitem(sys.modules, 'vllm', types.SimpleNamespace(LLM=LLM, SamplingParams=params, __version__='0.27.1'))
    monkeypatch.setitem(sys.modules, 'vllm.inputs', types.SimpleNamespace(TokensPrompt=lambda **kw: kw))
    golden = tmp_path / 'golden'
    golden.write_text('prompt ' + ' '.join(['1']*1024) + '\ngen ' + ' '.join(['1']*32))
    monkeypatch.setattr(sys, 'argv', ['bench', '--model', 'model', '--golden', str(golden), '--repeats', '5',
                                    '--engine-kw', '{"attention_backend":"TEST"}', '--no-detok', '--echo-config'])
    monkeypatch.setenv('VLLM_TEST', 'yes')
    ticks = iter(range(100))
    monkeypatch.setattr(bench.time, 'perf_counter', lambda: next(ticks))
    # Full and one calls need distinct elapsed times to avoid a zero median.
    ticks = iter([0, 32, 32, 33] * 6)
    bench.main()
    text = capsys.readouterr().out
    assert calls[0]['attention_backend'] == 'TEST'
    assert all(p['detokenize'] is False for p in sampling)
    assert 'VLLM-SAMPLE sample=0 first_divergence=0/32' in text
    assert 'VLLM-SAMPLE sample=4 first_divergence=32/32' in text
    assert bench.CONFIG_PREFIX == 'VLLM-CONFIG '
    line = r.one_line(text, r.CONFIG_PREFIX, 'bench')
    echo = bench.decode_config(line)
    assert bench.encode_config(echo) == line
    assert bench.decode_config(bench.encode_config(echo)) == echo
    # Independent wire-format oracle catches a shared prefix mutation.
    assert json.loads(line[len('VLLM-CONFIG '):]) == echo
    with pytest.raises(ValueError):
        bench.decode_config(line[1:])
    assert echo['env']['VLLM_TEST'] == 'yes'
    assert echo['async_scheduling'] is True


def test_vendor_pinned_bytes():
    source = Path(__file__).resolve().parents[1] / 'examples/vllm_llm/vendor/benchmark_moe.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '230151de56d177ac22920ff9dada010f4361933b34b54e18d5981367723a8bcf'


@pytest.mark.parametrize('deltas,want', [
    ([-.1]*5, True), ([-.1, -.1, -.1, -.1, -.05], True),
    ([-.1, -.1, -.1, -.1, -.049], False),
    ([-.01]*5, True), ([0]*5, False),
    ([-.05, -.025, -.025, -.025, -.025], False),
])
def test_spread_bound_diagnostic(deltas, want):
    assert r.spread_bound(deltas) is want


def test_spread_bound_does_not_change_verdict():
    deltas = [-.2, -.2, -.2, -.2, -.1]
    assert not r.spread_bound(deltas)
    assert r.paired(deltas) == 'LEAD'


def test_report_spread_diagnostic_roundtrip(tmp_path, capsys):
    from tests.test_import_certificate import fixture_session
    from tests.test_nvref_report import write
    fixture_session(tmp_path)
    search = dict(tested=['S0'], winners={}, excluded={})
    (tmp_path / 'final.json').write_text(json.dumps(dict(final=r.config('S0'), search=search)))
    write(tmp_path, 'docker_pull', 'IMAGE ' + r.IMAGE, 'None')
    write(tmp_path, 'golden_sha', '/synthetic/golden_c1024.txt: OK', 'None')
    for rnd in range(1, 6):
        write(tmp_path, f'S0_r{rnd}', log(ms=5 if rnd < 5 else 4.9))
    assert r.main([str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert 'SPREAD-BOUND fused−R: False' in output
    assert '→ LEAD' in output
    assert 'FULL-CLAIM ELIGIBLE' in output
