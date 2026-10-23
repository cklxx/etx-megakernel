"""Synthetic search evidence; no vLLM, GPU, image or network required."""
import json
from pathlib import Path
import hashlib
import sys

import pytest

from examples.vllm_llm import vlbest_report as r
from examples.vllm_llm.pairs import Refusal


PLAN = '\n'.join(f'{i} attn {i} attn 1 1 1 0 1 0 0 0 1 0 -1 0 1' for i in range(48))


def proof(text, cfg):
    return r.proof(text, cfg, expected=r.expected_replacements(PLAN))


def observation(text, cfg):
    return r.observation(text, cfg, expected=r.expected_replacements(PLAN))


@pytest.fixture(autouse=True)
def graph_fixture(tmp_path):
    (tmp_path / 'vl_plan.txt').write_text(PLAN)


def log(arm='S0', ms=5, cfg=None):
    cfg = cfg or r.config(arm)
    env = {k: v.replace('$' + 'HOME', '/synthetic') for k, v in cfg['env'].items()}
    echo = dict(async_scheduling=True, cudagraph_mode='FULL_AND_PIECEWISE', pass_config={},
                kernel_config={}, attention_backend=cfg['kw'].get('attention_backend'), env=env)
    lines = [r.CONFIG_PREFIX + json.dumps(echo), 'VLLM-SAMPLING detok=0']
    if 'attention_backend' in cfg['kw']:
        lines += [f"Using {cfg['kw']['attention_backend']} backend (selected via --attention-backend)"]
    lines += ['Enabled custom fusions: norm_quant, act_quant, qk_norm_rope_kvcache',
              'QK-Norm+RoPE+KVCache fusion: replaced 48 pattern(s)',
              'Capturing CUDA graphs (decode, FULL)', 'Using TritonExperts MoE backend', "rms_norm=['native']"]
    if 'VLLM_TUNED_CONFIG_FOLDER' in env:
        lines += [f"Using configuration from {env['VLLM_TUNED_CONFIG_FOLDER']}/E=128,N=768,device_name=AMD_Instinct_MI300X.json for MoE layer"]
    lines += [f'VLLM-SAMPLE sample={i} first_divergence=32/32' for i in range(5)]
    lines += [f"VLLM label={arm} aiter={env.get('VLLM_ROCM_USE_AITER', '0')} batch=1 version=0.27.1 context=1024 first_divergence=32/32 median_ms={ms}"]
    return '\n'.join(lines)


@pytest.mark.parametrize('arm', [f'S{i}' for i in range(10)])
def test_proven(arm):
    assert observation(log(arm), r.config(arm)) == 5


@pytest.mark.parametrize('arm,remove', [
    ('S0', '"async_scheduling": true, '),
    ('S4', 'Using ROCM_AITER_FA backend (selected via --attention-backend)'),
    ('S4', 'Enabled custom fusions: norm_quant, act_quant, qk_norm_rope_kvcache'),
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
        observation(text.replace(remove, ''), r.config(arm))


@pytest.mark.parametrize('arm,extra', [('S7', '(EngineCore pid=123)'), ('S1', 'Using default MoE config')])
def test_contradictory_proof(arm, extra):
    with pytest.raises(Refusal, match='NOT-APPLIED'):
        observation(log(arm) + '\n' + extra, r.config(arm))


@pytest.mark.parametrize('bad', ['31/32', '31/31', '32/33', '0/32'])
def test_every_sample(bad):
    text = log().replace('sample=0 first_divergence=32/32', 'sample=0 first_divergence=' + bad)
    with pytest.raises(Refusal, match='failed 32/32'):
        observation(text, r.config('S0'))


def test_missing_sample():
    with pytest.raises(Refusal, match='failed 32/32'):
        observation(log().replace('VLLM-SAMPLE sample=0 first_divergence=32/32', ''), r.config('S0'))


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
    assert observation(log('C', cfg=cfg), cfg) == 5


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
        observation(log('S4').replace('label=S4', 'label=default'), r.config('S4'))


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
    from etx.testing.certificates import certified_import_session as fixture_session
    from tests.test_nvref_report import write
    fixture_session(tmp_path)
    search = dict(tested=['S0'], winners={}, excluded={})
    (tmp_path / 'final.json').write_text(json.dumps(dict(final=r.config('S0'), search=search)))
    write(tmp_path, 'docker_pull', 'IMAGE ' + r.IMAGE, 'None')
    write(tmp_path, 'golden_sha', '/synthetic/golden_c1024.txt: OK', 'None')
    write(tmp_path, 'record', 'GPU[0] : Card Series: Synthetic GPU\nROCM-VERSION test-version', 'None')
    (tmp_path / 'screen.json').write_text(json.dumps({'combination': None}))
    for rnd in (1, 2):
        for end in ('open', 'close'):
            write(tmp_path, f'a_r{rnd}_S0_{end}', log())
        for arm in r.SCREEN_ARMS:
            write(tmp_path, f'a_r{rnd}_{arm}', log(arm, ms=6))
    for rnd in range(1, 6):
        write(tmp_path, f'S0_r{rnd}', log(ms=5 if rnd < 5 else 4.9))
    assert r.main([str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert 'SPREAD-BOUND fused−R: False' in output
    assert '→ LEAD' in output
    assert 'FULL-CLAIM ELIGIBLE' in output
    assert 'ETX: step mean; vLLM: sample median' in output
    assert 'FUSION step-mean basis' in output
    assert 'FUSION RESULT wall_ms basis (pairs.py)' in output
    assert 'Synthetic GPU | ROCm test-version' in output


@pytest.mark.parametrize('arm', ['S4f', 'S5f', 'S6f'])
def test_fixed_fusion_positive(arm):
    cfg = r.config(arm)
    assert cfg['kw']['compilation_config'] == {'use_inductor_graph_partition': True}
    base = r.config(arm[:-1])
    assert cfg['env'] == base['env'] and cfg['flags'] == base['flags']
    assert cfg['kw']['attention_backend'] == base['kw']['attention_backend']
    assert observation(log(arm), cfg) == 5


def test_captured_disabled_fusion():
    captured = Path(__file__).parent / 'fixtures/vlbest/a_r1_S4.log'
    text = captured.read_text()
    assert 'Disabling fuse_qk_norm_rope_kvcache' in text
    # This historical capture predates VLLM_USE_LAYERNAME; env requirements
    # are checked before component disabling, so preserve both refusal checks.
    captured_config = r.config('S4f')
    captured_config['env'] = captured_config['env'].copy()
    captured_config['env'].pop('VLLM_USE_LAYERNAME')
    assert proof(text, captured_config) == 'Disabling fuse_qk_norm_rope_kvcache'
    assert proof(text, r.config('S4f')) == 'env VLLM_USE_LAYERNAME'
    with pytest.raises(Refusal, match='NOT-APPLIED'):
        observation(text, r.config('S4f'))


@pytest.mark.parametrize('brackets', [False, True])
def test_fusion_banner_newline_regression(brackets):
    text = log('S4f')
    if brackets:
        text = text.replace('fusions: norm_quant, act_quant, qk_norm_rope_kvcache',
                            'fusions: [norm_quant, act_quant, qk_norm_rope_kvcache]')
    assert proof(text, r.config('S4f')) is None
    assert proof(text.replace('act_quant, qk_', 'act_quant,\nqk_'), r.config('S4f')) is not None


@pytest.mark.parametrize('count', [0, 1, 47])
def test_fusion_replacement_is_primary(count):
    text = log('S4f').replace('replaced 48', f'replaced {count}')
    # Even with the enabled banner, zero or partial replacement is not proof.
    assert proof(text, r.config('S4f')) == ('replaced 0 patterns' if count == 0 else f'replacement counts [{count}]; expected 48 per full graph')
    # If both positive proofs are absent, report the missing replacement first.
    text = text.replace('Enabled custom fusions:', 'Other fusions:')
    assert proof(text, r.config('S4f')) == ('replaced 0 patterns' if count == 0 else f'replacement counts [{count}]; expected 48 per full graph')


@pytest.mark.parametrize('arm', ['S4f', 'S5f', 'S6f'])
@pytest.mark.parametrize('remove', [
    'Enabled custom fusions: norm_quant, act_quant, qk_norm_rope_kvcache',
    'QK-Norm+RoPE+KVCache fusion: replaced 48 pattern(s)',
    'Capturing CUDA graphs (decode, FULL)',
])
def test_fixed_missing_fusion_proof(arm, remove):
    with pytest.raises(Refusal, match='NOT-APPLIED'):
        observation(log(arm).replace(remove, ''), r.config(arm))


@pytest.mark.parametrize('arm', ['S4f', 'S5f', 'S6f'])
def test_fixed_disabling_warning_overrides_positive(arm):
    assert proof(log(arm) + '\nDisabling fuse_qk_norm_rope_kvcache', r.config(arm)) is not None


def test_c2_union():
    cfg = r.combination({a: winner(a, -.1) for a in ('S1', 'S6f')})
    assert cfg['id'] == 'C2'
    assert set(cfg['components']) == {'S1', 'S6f'}
    assert proof(log('C2', cfg=cfg), cfg) is None
    assert proof(log('C2', cfg=cfg) + '\nDisabling fuse_qk_norm_rope_kvcache', cfg)


def test_fusion_guard_mutations(monkeypatch):
    import inspect
    source = inspect.getsource(r.proof)
    cases = [
        ("if 'Disabling fuse_qk_norm_rope_kvcache' in text:", 'if False:',
         lambda: test_fixed_disabling_warning_overrides_positive('S4f')),
        (r'[^\n]*qk_norm_rope_kvcache', r'[^\\r\\n]*qk_norm_rope_kvcache',
         lambda: test_fusion_banner_newline_regression(False)),
    ]
    for old, new, oracle in cases:
        assert old in source
        namespace = dict(r.__dict__)
        exec(source.replace(old, new), namespace)
        with monkeypatch.context() as patch:
            patch.setattr(r, 'proof', namespace['proof'])
            with pytest.raises(AssertionError):
                oracle()


def test_changed_round_configuration_refused(tmp_path, monkeypatch):
    from tests.test_nvref_report import write
    for rnd in (1, 2):
        for end in ('open', 'close'):
            write(tmp_path, f'a_r{rnd}_S0_{end}', log())
        write(tmp_path, f'a_r{rnd}_S4f', log('S4f'))
    cfg = r.config('S4f')
    cfg['kw']['compilation_config'] = {'splitting_ops': []}
    (tmp_path / 'a_r2_S4f.config.json').write_text(json.dumps(cfg))
    result = r.screen(tmp_path, candidates=[r.config('S4f')])
    assert 'configuration changed' in result['excluded']['S4f']
    assert result['tested'] == ['S0']


def test_changed_final_configuration_refused(tmp_path, capsys):
    from etx.testing.certificates import certified_import_session as fixture_session
    from tests.test_nvref_report import write
    fixture_session(tmp_path)
    (tmp_path / 'final.json').write_text(json.dumps(dict(final=r.config('S0'), search={})))
    for rnd in range(1, 6):
        write(tmp_path, f'S0_r{rnd}', log())
    (tmp_path / 'F_r1.config.json').write_text(json.dumps(r.config('S4f')))
    assert r.main([str(tmp_path)]) == 2
    assert 'finalist configuration changed' in capsys.readouterr().out


def test_configuration_guard_mutations(tmp_path, monkeypatch, capsys):
    import inspect
    for func, old, oracle in [
        (r.screen, 'resolved[0] != resolved[1]',
         lambda patch: test_changed_round_configuration_refused(tmp_path, patch)),
        (r.report, "path.exists() and json.loads(path.read_text()) != cfg",
         lambda patch: test_changed_final_configuration_refused(tmp_path, capsys)),
    ]:
        source = inspect.getsource(func)
        assert old in source
        namespace = dict(r.__dict__)
        exec(source.replace(old, 'False'), namespace)
        with monkeypatch.context() as patch:
            patch.setattr(r, func.__name__, namespace[func.__name__])
            with pytest.raises(AssertionError):
                oracle(patch)


def test_budget_omission_overrides_evidence(tmp_path, monkeypatch):
    (tmp_path / 'screen_budget.log').write_text('SCREEN S4f SKIPPED budget\n')
    monkeypatch.setattr(r.Session, 'read', lambda *a, **kw: 'valid')
    monkeypatch.setattr(r, 'observation', lambda *a, **kw: 5)
    result = r.screen(tmp_path, candidates=[r.config('S4f')])
    assert result['tested'] == ['S0']
    assert result['excluded']['S4f'] == 'NOT-APPLIED S4f: budget admission'


def test_budget_omission_mutation(tmp_path, monkeypatch):
    import inspect
    namespace = dict(r.__dict__)
    exec(inspect.getsource(r.not_applied).replace('if match:', 'if False:'), namespace)
    monkeypatch.setattr(r, 'not_applied', namespace['not_applied'])
    with pytest.raises(AssertionError):
        test_budget_omission_overrides_evidence(tmp_path, monkeypatch)


def test_graph_count_and_counter_semantics():
    plan = '\n'.join(PLAN.splitlines()[:3])
    assert r.expected_replacements(plan) == 3
    text = log('S4f').replace('replaced 48', 'replaced 3')
    assert r.proof(text, r.config('S4f'), expected=3) is None
    assert r.proof(text, r.config('S4f')) == 'no graph coverage count'
    # Separate pass invocations must not be summed into full coverage.
    text = text.replace('replaced 3', 'replaced 1')
    text += '\nQK-Norm+RoPE+KVCache fusion: replaced 2 pattern(s)'
    assert 'expected 3' in r.proof(text, r.config('S4f'), expected=3)
    text = log('S4f').replace('QK-Norm+RoPE+KVCache fusion: replaced 48 pattern(s)', '')
    assert r.proof(text, r.config('S4f'), expected=3) == 'no replacement-count line'


def test_recompute_stale_search_and_both_round_failures(tmp_path):
    from tests.test_nvref_report import write
    (tmp_path / 'screen.json').write_text(json.dumps({'combination': None, 'search': {'winners': {'S1': {}}}}))
    for rnd in (1, 2):
        for end in ('open', 'close'):
            write(tmp_path, f'a_r{rnd}_S0_{end}', log(ms=5))
        write(tmp_path, f'a_r{rnd}_S1', log('S1', ms=5.3))
        write(tmp_path, f'a_r{rnd}_S3', log('S3', ms=4.8))
        for arm in ('S4f', 'S5f', 'S6f'):
            text = log(arm).replace('replaced 48', 'replaced 0')
            if rnd == 2:
                text = text.replace('QK-Norm+RoPE+KVCache fusion: replaced 0 pattern(s)', '')
            write(tmp_path, f'a_r{rnd}_{arm}', text)
        write(tmp_path, f'b_r{rnd}_C2', 'NOT-APPLIED C2: fewer than two compatible winning axes')
    search = r.recompute_search(tmp_path)
    assert list(search['winners']) == ['S3']
    assert search['excluded']['S1'] == 'NO-WINNER'
    assert 'fewer than two compatible' in search['excluded']['C2']
    assert 'r1: NOT-APPLIED S4f: replaced 0 patterns' in search['excluded']['S4f']
    assert 'r2: NOT-APPLIED S4f: no replacement-count line' in search['excluded']['S4f']


def test_survival_and_machine():
    assert '0.228 ms' in r.survival([-.278] * 5)
    assert 'uniform reference improvement only' in r.survival([-.278] * 5)
    assert 'recomputing every pair' in r.survival([-.278] * 5)
    assert r.machine('GPU[0] : Card Series: Another GPU\nROCM-VERSION 9.1') == 'Another GPU | ROCm 9.1'
    with pytest.raises(Refusal):
        r.machine('GPU[0] : Card Series: Another GPU')


@pytest.mark.parametrize('extra', ['', '\nUsing AiterExperts MoE backend', '\nUsing AITER MoE backend'])
def test_vlbest3_exact_tuned_hit(extra):
    cfg = r.config('M1')
    text = log('M1', cfg=cfg).replace(' for MoE layer', ' for MoE layer.')
    assert (r.proof(text + extra, cfg) is None) is (not extra)
    for wrong in ('E=128,N=768,device_name=AMD_Instinct_MI300X.json.bak',
                  'E=128,N=768,device_name=AMD_Instinct_MI300X,dtype=float16.json'):
        assert r.proof(text.replace(r.MOE_FILENAME, wrong), cfg)
    assert r.proof(text.replace('/synthetic/moecfg/tuned/' + r.MOE_FILENAME,
                                '/other/moecfg/tuned/' + r.MOE_FILENAME), cfg)


@pytest.mark.parametrize('deltas,won', [([-.05, -.05], True), ([-.1, .001], False),
                                        ([-.049, -.049], False), ([-.08, -.04], True)])
def test_vlbest3_selection(tmp_path, deltas, won):
    from examples.sessions import vlbest3 as v3
    for rnd, delta in enumerate(deltas, 1):
        for arm, ms in [('S0_open', 5), ('S3', 4.8), ('M1', 4.8+delta), ('S0_close', 5)]:
            name = f'a_r{rnd}_{arm}'
            cfg = r.config(arm.split('_')[0])
            text = log(cfg['id'], ms, cfg).replace(' for MoE layer', ' for MoE layer.')
            (tmp_path / f'{name}.log').write_text(f'# job {name} gpu=0\n' + text)
    data = v3.select(tmp_path)
    assert data['final']['id'] == ('S9' if won else 'S3')
    assert 'S9' in data['search']['tested']
    assert r.recompute_search(tmp_path) == data['search']
    assert r.finalist(data['search']['winners']) == data['final']
    if not won:
        assert 'tested-and-not-won' in data['search']['excluded']['S9']
    target = tmp_path / 'a_r2_M1.log'
    target.write_text(target.read_text().replace('sample=0 first_divergence=32/32', 'sample=0 first_divergence=31/32'))
    with pytest.raises(Refusal, match='32/32'):
        v3.select(tmp_path)
