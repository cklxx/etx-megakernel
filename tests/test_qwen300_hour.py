"""CPU refusals, observations and end-to-end control flow; no GPU mocks are results."""
import copy
import json
import os
import subprocess
from pathlib import Path
import sys
import struct
import types

import pytest

from examples.sessions import qwen300_hour as h


def valid_perturbation():
    return h.stamp_perturbation(dict(perturbation=dict(sampled_bodies=48, max_abs_body_change_us=.5)), True)


def trace(previous=True, ambiguous=False):
    p = dict(types={'attn_0': dict(id=0), 'prep_0': dict(id=1), 'reduce_0': dict(id=2)},
             descs=[dict(id=i, type=i, coord=[0], domain=0, mode='static') for i in range(3)],
             domains=1, workers_per_domain=2, events=[dict(name='done', offset=0, shape=[1])],
             ev_inst=[['done', [0], [0]]], task_in=[[], [], [0]])
    r = dict(version=1, tick_ns=10, waves_per_task=1,
             legacy=[[100, 110, 200, 0], [150, 160, 250, 1], [260, 290, 400, 1 if previous else 2]],
             poll=[[0], [0], [280]], events=[[215, 220, 0]],
             waves=[[180, 190], [220, 230], [350, 380]])
    if ambiguous:
        r['events'] = [[251, 255, 0]]
    return p, r


@pytest.mark.parametrize('previous,ambiguous,want', [(True, False, 'busy'), (False, False, 'free'), (True, True, 'ambiguous')])
def test_edge_classification_does_not_invent_tail_ownership(previous, ambiguous, want):
    p, r = trace(previous, ambiguous)
    result = h.edge_prices(p, r)
    assert result['edges'][0]['edge_class'] == want
    assert result['edges'][0]['gap_bounds_us'][0] < 1
    assert result['all_edges'][want]['below_floor_count'] == 1
    assert result['all_edges'][want]['above_floor_lower_ms'] == 0
    assert result['all_edges'][want]['signed_excess_ms'] < 0
    assert result['edges'][0]['nonfinal_producer_arrivals'].startswith('UNOBSERVED')
    assert result['inbody']['acceptance']['exclusive_attributed_fraction'] == 0


def test_missing_stamp_refuses_edge_price():
    p, r = trace(); r['events'][0] = [0, 0, 0]
    with pytest.raises(ValueError): h.edge_prices(p, r)


def test_previous_owned_arrival_proves_busy_after_legacy_body_end():
    p, r = trace()
    p['events'].append(dict(name='prev_done', offset=1, shape=[1]))
    p['ev_inst'].append(['prev_done', [0], [1]])
    p['task_in'][2] = [0, 1]
    r['events'].append([265, 270, 1])
    r['legacy'][1] = [150, 160, 210, 1]; r['waves'][1] = [180, 190]
    r['legacy'][2] = [280, 320, 450, 1]; r['poll'][2] = [300]; r['waves'][2] = [400, 420]
    result = h.edge_prices(p, r)
    edge = result['edges'][0]
    assert edge['edge_class'] == 'busy'
    assert edge['previous_last_known_busy_tick'] == 270
    # Publication stays in the legacy gap; do not silently subtract it from
    # the registered 1-us floor/pool basis by switching to B-A.
    assert result['critical_edges'][0]['priced_gap_us'] == pytest.approx(1.1)
    assert result['critical_pools']['busy']['above_floor_lower_ms'] == pytest.approx(.0001)


def test_a2_prices_critical_body_and_is_not_attainability():
    rows = [dict(layer=l, phase=phase, body_us=cost) for l in range(48)
            for phase, cost in [('attn', 5.360), ('reduce', 5.285)]]
    result = h.attention_price(dict(phases=rows))
    assert result['body_ms'] == pytest.approx(.51096)
    assert result['available_cut_estimate_ms'] == pytest.approx(.23592)
    assert result['attainable_cut_ms'] is None
    with pytest.raises(ValueError): h.attention_price(dict(phases=rows[:-1]))


@pytest.mark.parametrize('failure', ['none', 'certify', 'run', 'analysis', 'coverage'])
def test_a2_failure_preserves_g_and_missing_a1_is_incomplete(tmp_path, monkeypatch, failure):
    s = session(tmp_path); s.c['builds']['stamps'] = '/unused/stamps'
    def certify(s, arm):
        if arm == 'stages' or (arm == 'stamps' and failure == 'certify'):
            raise ValueError('diagnostic certification failed')
        s.certified.add(arm)
    monkeypatch.setattr(h, 'certify', certify)
    def host(*args):
        if failure == 'run': raise RuntimeError('stamp run failed')
        return host_text()+'HOUR-SNAPSHOT run=0 step=1024\n'
    monkeypatch.setattr(h, 'host', host)
    phases = [dict(layer=l, phase=p, body_us=5.) for l in range(48) for p in ('attn', 'reduce')]
    if failure == 'coverage': phases.pop()
    def analysis(name, cmd, **kw):
        if failure == 'analysis': raise ValueError('analysis failed')
        pool = dict(above_floor_lower_ms=.2, above_floor_upper_ms=.2, count=1)
        result = dict(critical_pools=dict(busy=pool, free=pool, ambiguous=dict(count=0)),
                      gap_atoms=dict(G_ms=.324), stamp_perturbation=valid_perturbation(), inbody=dict(phases=phases))
        Path(cmd[-2]).with_suffix('.summary.json').write_text(json.dumps(result))
    monkeypatch.setattr(s, 'step', analysis)
    complete_g = failure in ('none', 'coverage')
    assert h.tranche_a(s) is complete_g
    assert s.rows['A.stop']['status'] == ('PASS' if complete_g else 'NOT-COMPLETE')
    if complete_g:
        assert s.rows['A.stop']['value']['G_ms'] == .324
        assert h.second_stop(s)
        assert s.rows['A2.route']['status'] == ('NOT-COMPLETE' if failure == 'coverage' else 'HEADROOM-PASS')
        assert s.rows['B.stop']['value']['R2_ms'] == 0.
    if failure == 'certify':
        control = s.rows['A.stop']['value']['stamp_perturbation']['original_off_control']
        assert control['original_off_within_spread'] is None
        assert 'stamps_OFF' in control['reason']


def host_text(mode='fused', cost=4.18):
    text = 'ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH=\n'
    text += 'mode: '+('ETX megakernel' if mode == 'fused' else 'unfused HIP graph')+'\n'
    for t in range(1024, 1055):
        text += 'HOUR-WALL '+json.dumps(dict(run=0, step=t, ms=cost))+'\n'
    text += 'run 0: 32/32 generated tokens match HF greedy\n'
    text += f'RESULT model=qwen requests=1 matched=32 gen=32 median_ms=4.000 wall_ms={cost:.3f} matched_each=32\n'
    return text


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'mismatch', 'extra_repeat', 'nan', 'wrongmode'])
def test_partial_timing_never_becomes_a_result(mutation):
    text = host_text()
    if mutation == 'missing': text = '\n'.join(l for l in text.splitlines() if '"step": 1024' not in l)
    if mutation == 'duplicate': text += 'HOUR-WALL {"run":0,"step":1024,"ms":4.18}\n'
    if mutation == 'mismatch': text = text.replace('run 0: 32/32', 'run 0: 31/32')
    if mutation == 'extra_repeat': text += 'run 1: 32/32 generated tokens match HF greedy\n'
    if mutation == 'nan': text = text.replace('"ms": 4.18', '"ms": NaN')
    if mutation == 'wrongmode': text = text.replace('ETX megakernel', 'unfused HIP graph')
    with pytest.raises((ValueError, h.Refusal)): h.timing(text)


def test_complete_wall_basis_and_keep_bars():
    assert h.timing(host_text())['wall_ms'] == 4.18
    assert h.keep([.030, .030, .030])['keep']
    assert not h.keep([.040, .010, .040])['keep']
    assert not h.keep([.040, -.010, .040])['keep']
    assert not h.keep([.029, .029, .029])['keep']
    with pytest.raises(ValueError): h.keep([.2, .2])


def stage_text():
    out = []
    for t in range(1023, 1055):
        out += ['HOUR-STAGE '+json.dumps(dict(run=0, step=t, upload_us=5, reset_submit_us=49,
            residency_us=1, launch_us=15, wait_us=3900, retire_readback_us=20,
            instrument_other_us=0, reset_device_us=40, kernel_device_us=3855,
            wall_us=3990, device_tail_reset=0)),
            'HOUR-DEVICE '+json.dumps(dict(run=0, step=t, ms=3.855))]
    return '\n'.join(out)


def test_stage_accounting_subtracts_kernel_once():
    r = h.stage_prices(stage_text())
    assert r['host_excess_ms'] == pytest.approx(.135)
    assert sum(r['named_host_ms'].values()) == pytest.approx(.135)
    assert r['accounted_beyond_priced_ms'] == pytest.approx(.086)
    with pytest.raises(ValueError): h.stage_prices(stage_text().replace('"device_tail_reset": 0', '"device_tail_reset": 1'))
    with pytest.raises(ValueError): h.stage_prices(stage_text().replace('"step": 1023', '"step": 1024'))


def session(tmp_path):
    return h.Session(dict(manifest=str(h.MANIFEST), builds={'kept': '/unused'}, model='/unused'), tmp_path)


@pytest.mark.parametrize('missing', [None, 'packed', 'hf'])
def test_prewarm_reads_staged_weights_before_any_session_clock(tmp_path, monkeypatch, capsys, missing):
    model = tmp_path/'model'; model.mkdir()
    hf = model/'hf'; hf.mkdir()
    if missing != 'packed': (model/'vl.bin').write_bytes(b'packed weights')
    if missing != 'hf': (hf/'model-00001.safetensors').write_bytes(b'HF weights')
    config = tmp_path/'prepared.json'
    config.write_text(json.dumps(dict(model=str(model), hf=str(hf))))
    monkeypatch.setattr(h, 'Session', lambda *args: pytest.fail('prewarm must precede session clock'))
    if missing:
        with pytest.raises(ValueError, match='weight'): h.main(['--prewarm', str(config)])
        assert 'CPU-PREWARM begin' not in capsys.readouterr().out
    else:
        assert h.main(['--prewarm', str(config)]) == 0
        out = capsys.readouterr().out
        assert str(model/'vl.bin') in out and str(hf/'model-00001.safetensors') in out
        from examples.vllm_llm.import_certificate import sha
        assert json.loads(config.read_text())['weight_sha256'] == {str(p): sha(p) for p in (model/'vl.bin', hf/'model-00001.safetensors')}
        assert 'CPU-PREWARM complete' in out


def test_ineligible_c_lends_exactly_sixty_seven_minutes(tmp_path):
    s = session(tmp_path)
    s.m['C'] = []
    assert h.tranche_c(s) == ([], 4020)
    assert s.rows['C.fusion']['status'] == 'SKIP'


def test_timeout_retains_unfiltered_log_and_reports_stuck_state(tmp_path, monkeypatch):
    s = session(tmp_path)
    monkeypatch.setattr(s, 'probe', lambda out: out.write('STUCK-STATE synthetic\n'))
    with pytest.raises(TimeoutError):
        s.step('timeout', [sys.executable, '-u', '-c', 'import time; print("FULL stderr/error"); time.sleep(5)'], timeout=.1)
    assert 'FULL stderr/error' in (tmp_path/'timeout.log').read_text()
    assert 'STUCK-STATE' in (tmp_path/'timeout.log').read_text()
    s.finish()
    assert all(r['status'] == 'NOT-COMPLETE' for r in s.rows.values())


def test_attempt_catches_certificate_refusal_and_keeps_working(tmp_path):
    s = session(tmp_path)
    def refuse(): raise h.Refusal('stale certificate')
    assert s.attempt('broken', refuse) is None
    assert s.attempt('next', lambda: 123) == 123
    assert s.arms['broken']['status'] == 'NOT-COMPLETE'


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('extra', [0, 4020])
def test_d_runs_five_adjacent_pinned_pairs_per_baseline(tmp_path, monkeypatch, partial, extra):
    s = session(tmp_path); s.certified.add('kept'); calls = []
    monkeypatch.setattr(h, 'certify', lambda *args: None)
    def fake_host(s, arm, name, args):
        assert s.deadline == s.end
        mode = 'graph' if '--unfused' in args else 'fused'; calls.append(mode)
        if partial and name == 'D_fusion_r4_graph': raise TimeoutError('synthetic deadline')
        text = host_text(mode, 3.4 if mode == 'graph' else 3.3)
        (s.out/(name+'.log')).write_text('# job '+name+' gpu=0\n'+text)
        return text
    monkeypatch.setattr(h, 'host', fake_host)
    def fake_system(s, name): calls.append('S3'); return dict(wall_ms=3.5)
    monkeypatch.setattr(h, 'system', fake_system)
    h.tranche_d(s, [], extra)
    assert s.deadline == s.end
    assert calls[1:] == ['fused', 'graph', 'fused', 'S3', 'graph', 'fused', 'S3', 'fused',
                         'fused', 'graph', 'fused', 'S3', 'graph', 'fused', 'S3', 'fused',
                         'fused', 'graph', 'fused', 'S3']
    if partial:
        assert s.arms['D']['status'] == 'NOT-COMPLETE'
        assert 'D.target' not in s.rows
        assert 'D.fusion' not in s.rows
    else:
        assert s.arms['D']['status'] == 'COMPLETE'
        assert s.rows['D.target']['status'] == 'PASS'
        assert s.rows['D.fusion']['status'] == 'ESTABLISHED'
        assert s.rows['D.system']['status'] == 'LEAD'


def test_does_not_sum_overlap_without_prebuilt_combination(tmp_path, monkeypatch):
    s = session(tmp_path); s.arms.update(C1=dict(median_ms=.05), C5=dict(median_ms=.04))
    arms = h.manifest()['B']
    assert h.select_final(s, arms) == 'C1'


def test_changed_schedule_is_not_a_shape_candidate():
    p, _ = trace(); q = copy.deepcopy(p); q['workers_per_domain'] = 3
    assert h.schedule_signature(p) != h.schedule_signature(q)


def test_b_each_pair_has_gate_and_failed_arm_does_not_hide_next(tmp_path, monkeypatch):
    s = session(tmp_path); s.certified.add('kept'); calls = []
    monkeypatch.setattr(h, 'eligible_arm', lambda s, a: a)
    monkeypatch.setattr(h, 'certify', lambda s, arm: s.certified.add(arm))
    def fake_host(s, arm, name, args):
        calls.append(arm)
        return host_text(cost={'kept': 4.18, 'C1': 4.13, 'C5': 4.20}[arm])
    monkeypatch.setattr(h, 'host', fake_host)
    survivors = h.tranche_b(s)
    assert [a['id'] for a in survivors] == ['C1']
    assert calls == ['kept', 'C1', 'C1', 'kept', 'kept', 'C1', 'kept', 'C5', 'C5', 'kept', 'kept', 'C5']
    assert s.arms['C5']['status'] == 'REJECT'


def test_first_stop_does_not_spend_rest_on_missing_deciding_pools(tmp_path, monkeypatch):
    s = session(tmp_path)
    monkeypatch.setattr(h, 'preflight', lambda s: None)
    monkeypatch.setattr(h, 'environment_guard', lambda s: True)
    monkeypatch.setattr(h, 'qualify_c', lambda s: [])
    from examples.vllm_llm import x5
    monkeypatch.setattr(x5, 'run', lambda s: None)
    monkeypatch.setattr(h, 'tranche_a', lambda s: h.first_stop(s, None))
    monkeypatch.setattr(h, 'tranche_b', lambda s: pytest.fail('B must not run after stop'))
    h.execute(s)
    assert s.rows['A.stop']['status'] == 'NOT-COMPLETE'
    assert 'close target route' not in s.rows['A.stop']['value']['decision']
    assert s.rows['D.target']['status'] == 'NOT-RUN'


def test_stale_source_seal_cannot_produce_a_fresh_certificate(tmp_path, monkeypatch):
    s = session(tmp_path)
    b = tmp_path/'prepared'; b.mkdir(); (b/'hour_build.json').write_text('{}')
    s.c['builds']['kept'] = str(b); s.c['build_seals'] = dict(kept='wrong')
    with pytest.raises(ValueError, match='unsealed/stale'): h.verify_seal(s, 'kept')


def test_manifest_can_drop_b_without_harness_rewrite(tmp_path, monkeypatch):
    s = session(tmp_path); s.m['B'] = []; s.certified.add('kept')
    assert h.tranche_b(s) == []
    assert s.rows['B.keep']['status'] == 'NO-KEEP'


# The --run command reporting every registered threshold without staging is executed in
# tests/e2e/test_api_qwen300_hour.py.


@pytest.mark.parametrize('bad_warmup', [False, True])
def test_vllm_gates_warmup_before_timing_on_cpu(tmp_path, monkeypatch, capsys, bad_warmup):
    from examples.llm import vllm_bench
    golden = tmp_path/'golden.txt'
    golden.write_text('prompt 1 2 3\ngolden '+' '.join(map(str, range(32)))+'\n')
    calls = []
    class Sampling:
        def __init__(self, **kw): self.max_tokens = kw['max_tokens']
    class LLM:
        def __init__(self, **kw): pass
        def generate(self, prompts, sp, **kw):
            calls.append(sp.max_tokens)
            tokens = list(range(sp.max_tokens))
            if bad_warmup and len(calls) == 1: tokens[0] = -999
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=tokens)])]
    fake = types.ModuleType('vllm'); fake.LLM = LLM; fake.SamplingParams = Sampling; fake.__version__ = '0.27.1'
    inp = types.ModuleType('vllm.inputs'); inp.TokensPrompt = lambda **kw: kw
    monkeypatch.setitem(sys.modules, 'vllm', fake); monkeypatch.setitem(sys.modules, 'vllm.inputs', inp)
    monkeypatch.setattr(sys, 'argv', ['bench', '--model', 'unused', '--golden', str(golden), '--repeats', '1', '--gate-warmup'])
    if bad_warmup:
        with pytest.raises(RuntimeError, match='timing not started'): vllm_bench.main()
        assert calls == [32]
        assert 'VLLM label=' not in capsys.readouterr().out
    else:
        vllm_bench.main()
        assert calls == [32, 1, 32, 1]
        out = capsys.readouterr().out
        assert out.index('VLLM-WARMUP-GATE OK') < out.index('VLLM-SAMPLE')


def test_bfloat16_compute_dtype_cannot_hide_fp8_checkpoint(tmp_path):
    s = session(tmp_path); hf = tmp_path/'hf'; hf.mkdir()
    (hf/'config.json').write_text(json.dumps(dict(model_type='qwen3_moe', num_hidden_layers=48,
        hidden_size=2048, num_key_value_heads=4, num_experts=128, num_experts_per_tok=8,
        torch_dtype='bfloat16', quantization_config={'quant_method': 'fp8'})))
    gates = tmp_path/'gates.json'
    gates.write_text(json.dumps(dict(pytest=True, gfx942=True, dry_run=True, shell=True,
        local_matrix_16=True, rocm_build_check=True, source_hashes=h.source_hashes())))
    s.c.update(hf=str(hf), cpu_gates=str(gates))
    with pytest.raises(ValueError, match='FP8 checkpoint forbidden'): h.preflight(s)


def test_table_keeps_threshold_scalar_after_large_distribution(tmp_path, capsys):
    s = session(tmp_path)
    s.row('A1.busy', dict(gaps_us=list(range(1000)), above_floor_lower_ms=.280,
                         above_floor_upper_ms=.290, count=123), 'PASS', h.REGISTERED['A1.busy'])
    s.finish()
    table = capsys.readouterr().out.split('REGISTERED THRESHOLDS', 1)[1]
    assert '"above_floor_lower_ms": 0.28' in table
    assert 'gaps_us' not in table
    assert json.loads((tmp_path/'report.json').read_text())['thresholds']['A1.busy']['value']['gaps_us'] == list(range(1000))


def test_c_positive_body_keep_cannot_bypass_net_fusion_bar(tmp_path, monkeypatch):
    s = session(tmp_path)
    a = dict(id='proto', build='proto', kind='fusion', native_resources=dict(vgpr=128, lds_bytes=29736),
             expected_vgpr=128, expected_lds_bytes=29736)
    s.qualified_c = [a]
    def fake_pairs(s, a, tranche, publish=True):
        result = dict(h.keep([.040, .040, .040]), status='KEEP')
        s.arms[a['id']] = result
        return result
    monkeypatch.setattr(h, 'candidate_pairs', fake_pairs)
    assert h.tranche_c(s) == ([], 0)
    assert s.arms['proto']['status'] == 'REJECT'
    assert s.rows['C.fusion']['status'] == 'NO-KEEP'
    assert s.rows['C.fusion']['value']['proto']['native_resources']['lds_bytes'] == 29736


# The --analyze command is executed in tests/e2e/test_api_qwen300_hour.py.


@pytest.mark.parametrize('change,status,message', [(.5, 'NOT-COMPLETE', 'NOT-COMPLETE: original/OFF spread not established'),
                                                 (.501, 'INVALID', 'INVALID: load stamps exceed 0.5 us/body')])
def test_analysis_cli_masks_g_and_keeps_perturbation_message(tmp_path, monkeypatch, capsys, change, status, message):
    from etx.tools import inbody
    plan=tmp_path/'plan.json';plan.write_text('{}')
    raw=tmp_path/'raw.bin';(tmp_path/'stamps_load_off.bin').touch()
    def report(body):
        return dict(phases=[], critical_task_ids=[0], tasks=[dict(body_us=body,phase=next(iter(inbody.LOAD_ROLES)))])
    monkeypatch.setattr(inbody,'read_raw',lambda p: {})
    monkeypatch.setattr(inbody,'load_phases',lambda *args: {})
    monkeypatch.setattr(inbody,'analyze',lambda *args: report(1.))
    monkeypatch.setattr(h,'edge_prices',lambda *args: dict(critical_pools={},floor_us=1.,gap_atoms=dict(G_ms=.100),
                                                       basis='test',inbody=report(1.+change)))
    out=tmp_path/'analysis.json'
    assert h.main(['--analyze',str(plan),str(raw),str(out)])==0
    assert message in capsys.readouterr().out
    for path in (out,out.with_suffix('.summary.json')):
        result=json.loads(path.read_text())
        assert result['stamp_perturbation']['status']==status
        assert result['gap_atoms']['G_ms'] is None
        assert result['load_phases']['validity']==message


@pytest.mark.parametrize('g,passes', [(None, False), (.087999, False), (.088, True), (.100, True), (.324, True)])
@pytest.mark.parametrize('busy,free', [('PASS', 'PASS'), ('DEAD', 'UNRESOLVED')])
def test_s1_uses_gap_atoms_instead_of_pool_labels(tmp_path, g, passes, busy, free):
    s = session(tmp_path); s.certified.add('kept')
    s.rows.update({'A1.busy': dict(status=busy), 'A1.free': dict(status=free)})
    assert h.first_stop(s, None if g is None else dict(G_ms=g, atoms={}), valid_perturbation()) is passes
    assert s.rows['A.stop']['status'] == ('NOT-COMPLETE' if g is None else 'PASS' if passes else 'STOP')
    assert ':301-302' in s.rows['A.stop']['source']


@pytest.mark.parametrize('certified,g', [(False, .324), (True, float('nan')), (True, float('inf'))])
def test_s1_missing_correctness_or_measurement_does_not_close_route(tmp_path, certified, g):
    s = session(tmp_path)
    if certified: s.certified.add('kept')
    assert not h.first_stop(s, dict(G_ms=g), valid_perturbation())
    assert s.rows['A.stop']['status'] == 'NOT-COMPLETE'
    assert s.rows['A.stop']['value']['decision'] == 'END SESSION: NOT-COMPLETE'


@pytest.mark.parametrize('off,change,status', [(None, .5, 'NOT-COMPLETE'), (False, .5, 'NOT-COMPLETE'),
                                              (True, .501, 'INVALID'), (True, .5, 'VALID')])
def test_s1_consumes_perturbation_before_any_later_tranche(tmp_path, monkeypatch, capsys, off, change, status):
    s = session(tmp_path); s.c['builds']['stamps'] = '/unused/stamps'
    evidence = h.stamp_perturbation(dict(perturbation=dict(sampled_bodies=48, max_abs_body_change_us=change)), off)
    assert evidence['status'] == status
    monkeypatch.setattr(h, 'preflight', lambda s: None)
    monkeypatch.setattr(h, 'environment_guard', lambda s: True)
    monkeypatch.setattr(h, 'qualify_c', lambda s: [])
    def certify(s, arm):
        if arm == 'stages': raise ValueError('no stage observation')
        s.certified.add(arm)
    monkeypatch.setattr(h, 'certify', certify)
    monkeypatch.setattr(h, 'registered_reset', lambda s: None)
    monkeypatch.setattr(h, 'host', lambda *args: host_text()+'HOUR-SNAPSHOT run=0 step=1024\n')
    def analysis(name, cmd, **kw):
        pool = dict(above_floor_lower_ms=.2, above_floor_upper_ms=.2, count=1)
        # The analyzer masks invalid G while retaining raw atom observations.
        result = dict(critical_pools=dict(busy=pool, free=pool, ambiguous=dict(count=0)),
                      gap_atoms=dict(G_ms=None if status == 'INVALID' else .100,
                                     atoms={'raw': dict(above_floor_ms=.09274, counted=True)}),
                      stamp_perturbation=evidence, inbody=dict(phases=[]))
        Path(cmd[-2]).with_suffix('.summary.json').write_text(json.dumps(result))
    monkeypatch.setattr(s, 'step', analysis)
    calls = []
    from examples.vllm_llm import x5
    monkeypatch.setattr(x5, 'run', lambda s: calls.append('X5') or {})
    # S2 is outside this regression; let it pass so every later tranche is observable.
    monkeypatch.setattr(h, 'second_stop', lambda s: calls.append('S2') or True)
    monkeypatch.setattr(h, 'tranche_b', lambda s: calls.append('B') or [])
    monkeypatch.setattr(h, 'tranche_c', lambda s: (calls.append('C') or [], 0))
    monkeypatch.setattr(h, 'tranche_d', lambda *args: calls.append('D'))
    h.execute(s)
    row = s.rows['A.stop']
    assert row['value']['stamp_perturbation'] == evidence
    if status == 'VALID':
        assert row['status'] == 'PASS' and row['value']['G_ms'] == .100
        assert calls == ['X5', 'S2', 'B', 'C', 'D']
    elif status in ('INVALID', 'NOT-COMPLETE'):
        assert row['status'] == 'NOT-COMPLETE'
        assert row['value']['G_ms'] is None and row['value']['gap_atoms']['G_ms'] is None
        assert calls == ['X5', 'B', 'C', 'D']
        assert s.rows['B.stop']['status'] == 'NOT-COMPLETE'
        assert 'session' not in s.arms
        out = capsys.readouterr().out
        assert 'S1 verdict UNJUDGED' in out
        assert 'raw above-floor opportunity=0.09274 ms (uncertified); session continues to B/C/D' in out
        s.finish()
        report = json.loads((s.out/'report.json').read_text())
        assert report['thresholds']['A.stop']['status'] == 'NOT-COMPLETE'
        assert report['thresholds']['A.stop']['value']['G_ms'] is None
    else:
        assert row['status'] == 'NOT-COMPLETE'
        assert s.arms['session']['status'] == 'NOT-COMPLETE'
        assert row['value']['G_ms'] is None and row['value']['gap_atoms']['G_ms'] is None
        assert row['value']['decision'] == 'END SESSION: NOT-COMPLETE'
        assert calls == []
        assert all(s.rows[k]['status'] == 'NOT-RUN' for k in ('X5', 'B.keep', 'B.stop', 'C.body', 'D.target'))
        s.finish()
        report = json.loads((s.out/'report.json').read_text())
        assert report['stamp_perturbation'] == evidence
        assert report['thresholds']['A.stop']['value']['G_ms'] is None


def test_s1_missing_or_contradictory_perturbation_fails_closed(tmp_path):
    s = session(tmp_path); s.certified.add('kept')
    for evidence in (None, {}, dict(valid_perturbation(), load_stamps_within_bar=False)):
        assert not h.first_stop(s, dict(G_ms=.100), evidence)
        assert s.rows['A.stop']['status'] == 'NOT-COMPLETE'
        assert s.rows['A.stop']['value']['G_ms'] is None


@pytest.mark.parametrize('change_ticks,original_off,status,passes', [
    (352, None, 'INVALID', False),  # paid session: 3.52 us, not clock resolution
    (352, True, 'INVALID', False),
    (52, True, 'INVALID', False),   # 0.52 us still exceeds the unchanged bar
    (48, None, 'NOT-COMPLETE', False),
    (48, True, 'VALID', True),
    (50, True, 'VALID', True),
])
def test_paid_gap_opportunity_needs_both_perturbation_gates(
        tmp_path, change_ticks, original_off, status, passes):
    evidence = h.stamp_perturbation(dict(perturbation=dict(
        sampled_bodies=382, max_abs_body_change_us=change_ticks * 10 / 1000)), original_off)
    assert evidence['status'] == status
    s = session(tmp_path); s.certified.add('kept')
    assert h.first_stop(s, machine_figures(), evidence) is passes
    row = s.rows['A.stop']
    assert row['status'] == ('PASS' if passes else 'NOT-COMPLETE')
    assert row['value']['G_ms'] == (.356727 if passes else None)
    assert row['value']['setup_atom_ms'] == .09274
    assert row['value']['named_above_floor_sum_ms'] == pytest.approx(.356727)
    assert row['value']['G_figure'] == h.G_FIGURE
    assert row['value']['G_figure_disagreement'] is True
    assert row['value']['decision'] == ('CONTINUE to S2' if passes else 'END SESSION: NOT-COMPLETE')


def machine_figures():
    # Only the setup and total are the supplied machine figures. The split of
    # the other two atoms below is synthetic CPU test data, not GPU evidence.
    atoms = {name: dict(above_floor_ms=value, counted=True) for name, value in
             [('P10.arrive', .100), ('P5/P11.observation', .163987), ('P4/P6/P7.setup', .09274)]}
    return dict(atoms=atoms, G_ms=.356727, **h.gap_figure(dict(atoms=atoms)))


def control_text(values):
    return '\n'.join(f'run {i}: 32/32 generated tokens match HF greedy\nINBODY-RUN '+
                     json.dumps(dict(run=i, matched=32, gen=32, wall_ms=x)) for i, x in enumerate(values))


@pytest.mark.parametrize('off,expected', [([4.1, 4.2, 4.3, 4.2, 4.2], True),
                                         ([4.5]*5, False), ([4.2]*4, None), ([float('nan')]*5, None)])
def test_runner_measures_full_original_off_distributions(tmp_path, monkeypatch, off, expected):
    s = session(tmp_path); s.certified.update(('kept', 'stamps'))
    calls = []
    def host(s, arm, name, args, extra):
        calls.append((arm, name, args, extra))
        assert 'ETX_INBODY_RAW' not in s.env and 'ETX_HOUR_TRACE_STEP' not in s.env
        return control_text([4., 4.1, 4.2, 4.1, 4.1] if arm == 'kept' else off)
    monkeypatch.setattr(h, 'host', host)
    result = h.measure_original_off(s)
    assert result['original_off_within_spread'] is expected
    assert result['status'] == ('PASS' if expected is True else 'FAIL' if expected is False else 'NOT-COMPLETE')
    assert [a for a, *_ in calls] == ['kept', 'stamps']
    assert all(args == ['--repeat', '5'] and extra == {'ETX_INBODY_LOADS': '0'} for _, _, args, extra in calls)
    assert json.loads((s.out/'A_original_off.json').read_text())['status'] == result['status']
    if expected is None: assert 'stamps_OFF' in result['reason']


@pytest.mark.parametrize('control_case,change,passes', [('missing', .48, False), ('outside', .48, False),
                                                    ('within', .501, False), ('within', .48, True)])
def test_runner_control_reaches_analyzer_and_named_s1_row(tmp_path, monkeypatch, capsys, control_case, change, passes):
    from etx.tools import inbody
    s = session(tmp_path); s.c['builds']['stamps'] = str(tmp_path/'stamps')
    Path(s.c['builds']['stamps']).mkdir()
    (Path(s.c['builds']['stamps'])/'plan.json').write_text('{}')
    (s.out/'stamps_load_off.bin').touch()
    def certify(s, arm):
        if arm == 'stages': raise ValueError('no stage observation')
        s.certified.add(arm)
    monkeypatch.setattr(h, 'certify', certify)
    monkeypatch.setattr(h, 'registered_reset', lambda s: None)
    def host(s, arm, name, args, extra):
        if name.startswith('A_control_'):
            if arm == 'stamps' and control_case == 'missing': raise TimeoutError('OFF control deadline')
            return control_text([4.5]*5 if arm == 'stamps' and control_case == 'outside' else [4.1]*5)
        return host_text()+'HOUR-SNAPSHOT run=0 step=1024\n'
    monkeypatch.setattr(h, 'host', host)
    def report(body):
        return dict(phases=[], critical_task_ids=[0], tasks=[dict(body_us=body, phase=next(iter(inbody.LOAD_ROLES)))])
    monkeypatch.setattr(inbody, 'read_raw', lambda p: {})
    monkeypatch.setattr(inbody, 'load_phases', lambda *args: {})
    monkeypatch.setattr(inbody, 'analyze', lambda *args: report(1.))
    pool = dict(above_floor_lower_ms=.2, above_floor_upper_ms=.2, count=1)
    monkeypatch.setattr(h, 'edge_prices', lambda *args: dict(
        critical_pools=dict(busy=pool, free=pool, ambiguous=dict(count=0)), floor_us=1.,
        gap_atoms=machine_figures(), basis='synthetic test fixture', inbody=report(1.+change)))
    def analyze(name, cmd, **kw):
        assert name == 'A_analysis'
        assert h.main(list(map(str, cmd[4:]))) == 0
    monkeypatch.setattr(s, 'step', analyze)
    assert h.tranche_a(s) is passes
    row = s.rows['A.stop']['value']
    assert row['G_ms'] == (.356727 if passes else None)
    assert row['G_figure'] == h.G_FIGURE
    assert row['G_figure_disagreement'] is True
    assert row['setup_atom_ms'] == .09274
    assert row['named_above_floor_sum_ms'] == pytest.approx(.356727)
    assert row['stamp_perturbation']['original_off_control']['original_off_within_spread'] is (
        None if control_case == 'missing' else control_case == 'within')
    out = capsys.readouterr().out
    log = json.loads(next(line.removeprefix('S1-G ') for line in out.splitlines() if line.startswith('S1-G ')))
    assert log['gate_compared_ms'] == row['G_ms'] and log['G_figure'] == row['G_figure']
    import inspect
    assert log['G_code'] in inspect.getsource(h.gap_atom_prices)
    if control_case == 'missing': assert 'stamps_OFF' in row['stamp_perturbation']['original_off_control']['reason']
    for file in ('A_edges.json', 'A_edges.summary.json'):
        result = json.loads((s.out/file).read_text())
        assert result['gap_atoms']['G_ms'] == row['G_ms']
        assert result['gap_atoms']['G_figure'] == row['G_figure']


@pytest.mark.parametrize('g,r2,host,passes', [
    (.100, None, .030, False), (.088, .236, 0., True),
    (.100, .193, .030, False), (.100, .194, .030, True),
    (.324, None, 0., True), (.200, None, .135, False)])
def test_s2_combines_gap_attention_and_attributable_host(tmp_path, g, r2, host, passes):
    s = session(tmp_path); s.certified.add('kept')
    assert h.first_stop(s, dict(G_ms=g), valid_perturbation())
    s.rows['A2.route'] = dict(value=dict(attainable_cut_ms=r2, available_cut_estimate_ms=.900))
    s.rows['A3.host'] = dict(value=dict(named_host_ms=dict(launch_us=host, reset_submit_us=.049,
        kernel_outside_token_timers_ms=.900, instrument_other_ms=.900), accounted_beyond_priced_ms=.900))
    assert h.second_stop(s) is passes
    value = s.rows['B.stop']['value']
    assert value['R2_ms'] == (r2 or 0.)
    assert value['H_ms'] == pytest.approx(min(.086, host))
    assert value['total_ms'] == pytest.approx(g+(r2 or 0.)+value['H_ms'])
    assert ':306-308' in s.rows['B.stop']['source']


@pytest.mark.parametrize('g,runs_b,runs_cd', [(.087, False, False), (.100, False, False), (.324, True, True)])
def test_execute_obeys_both_stops_with_time_remaining(tmp_path, monkeypatch, capsys, g, runs_b, runs_cd):
    s = session(tmp_path); s.certified.add('kept'); calls = []
    monkeypatch.setattr(h, 'preflight', lambda s: None)
    monkeypatch.setattr(h, 'environment_guard', lambda s: True)
    monkeypatch.setattr(h, 'qualify_c', lambda s: [])
    from examples.vllm_llm import x5
    monkeypatch.setattr(x5, 'run', lambda s: None)
    monkeypatch.setattr(h, 'tranche_a', lambda s: h.first_stop(s, dict(G_ms=g), valid_perturbation()))
    def b(s):
        assert s.rows['B.stop']['status'] == 'PASS'
        calls.append('B'); return ['body-keep']
    def c(s): calls.append('C'); return [], 4020
    def d(s, survivors, extra):
        assert survivors == ['body-keep']
        calls.append('D')
    monkeypatch.setattr(h, 'tranche_b', b)
    monkeypatch.setattr(h, 'tranche_c', c)
    monkeypatch.setattr(h, 'tranche_d', d)
    h.execute(s)
    assert calls == (['B', 'C', 'D'] if runs_cd else ['B'] if runs_b else [])
    assert s.end-h.time.monotonic() > 3900
    if not runs_cd:
        assert all(s.rows[k]['status'] == 'NOT-RUN' for k in ('C.fusion', 'C.body', 'D.target', 'D.fusion', 'D.system'))
    if g >= .088: assert s.rows['B.stop']['status'] == ('PASS' if runs_cd else 'STOP')
    if not runs_b: assert s.rows['B.keep']['status'] == 'NOT-RUN'
    if g < .088:
        out = capsys.readouterr().out
        assert 'END after A: STOP-SPEND: close target route; G_ms=0.087' in out
        assert 'UNJUDGED' not in out


def test_invalid_s1_does_not_override_session_deadline(tmp_path, monkeypatch, capsys):
    s = session(tmp_path); s.certified.add('kept')
    monkeypatch.setattr(h, 'preflight', lambda s: None)
    monkeypatch.setattr(h, 'environment_guard', lambda s: True)
    monkeypatch.setattr(h, 'qualify_c', lambda s: [])
    evidence = h.stamp_perturbation(dict(perturbation=dict(sampled_bodies=48, max_abs_body_change_us=.501)), True)
    def a(s):
        s.end = h.time.monotonic()-1
        return h.first_stop(s, dict(G_ms=.09274), evidence)
    monkeypatch.setattr(h, 'tranche_a', a)
    from examples.vllm_llm import x5
    monkeypatch.setattr(x5, 'run', lambda s: pytest.fail('deadline must stop before X5'))
    monkeypatch.setattr(h, 'tranche_b', lambda s: pytest.fail('deadline must stop before B'))
    h.execute(s)
    assert s.rows['A.stop']['status'] == 'NOT-COMPLETE'
    assert s.rows['D.target']['status'] == 'NOT-RUN'
    assert s.arms['session']['reason'] == 'session deadline reached'
    assert 'END after A: session deadline reached' in capsys.readouterr().out


def test_gap_atoms_clip_to_critical_gap_and_apply_class_floor(tmp_path):
    def task(phase='qkv', critical=True):
        return dict(critical=critical, arrive_is_synthetic=False, phase=phase,
                    record=[0, 200, 300, 0, 120, 170], arrive_bounds_ticks=[90, 120], gap_us=10.)
    report = dict(raw=dict(tick_ns=100), tasks=[task() for _ in range(48)])
    # Gap starts at tick 100: only 2 us of arrive, 5 of observation, 3 of setup.
    result = h.gap_atom_prices(report)
    assert result['atoms']['P10.arrive']['above_floor_ms'] == pytest.approx(48*(2-.30)/1000)
    assert result['atoms']['P5/P11.observation']['above_floor_ms'] == pytest.approx(48*(5-.951)/1000)
    assert result['atoms']['P4/P6/P7.setup']['above_floor_ms'] == pytest.approx(48*(3-.15)/1000)
    total = sum(a['above_floor_ms'] for a in result['atoms'].values() if a['counted'])
    assert result['G_ms'] == total
    s = session(tmp_path); s.certified.add('kept')
    assert h.first_stop(s, result, valid_perturbation())
    value = s.rows['A.stop']['value']
    assert value['G_ms'] == total == value['named_above_floor_sum_ms']
    assert value['G_figure'] == result['G_figure'] == h.G_FIGURE
    assert value['setup_atom_ms'] != value['G_ms'] and value['G_figure_disagreement']
    report['tasks'] += [task('attn'), task('reduce'), task(critical=False)]
    assert h.gap_atom_prices(report) == result  # booked boundaries/parallel bodies excluded
    report['tasks'] = [task()]
    result = h.gap_atom_prices(report)
    assert result['G_ms'] == 0.  # every class below 0.625 us/layer; raw excess retained
    assert result['atoms']['P10.arrive']['above_floor_ms'] > 0.


@pytest.mark.parametrize('excess_ticks,passes', [(0, False), (20, False), (30, True)])
def test_s1_stops_at_protocol_floor_and_counts_only_excess(tmp_path, excess_ticks, passes):
    # 339 critical edges: pre .05, arrive .30, observation .50, setup .15 us.
    task = dict(critical=True, arrive_is_synthetic=False, phase='qkv',
                record=[0, 100+excess_ticks, 200+excess_ticks, 0, 35+excess_ticks, 85+excess_ticks],
                arrive_bounds_ticks=[5, 35+excess_ticks], gap_us=1.+excess_ticks*.01)
    result = h.gap_atom_prices(dict(raw=dict(tick_ns=10), tasks=[task for _ in range(339)]))
    assert result['G_ms'] == pytest.approx(339*excess_ticks*.01/1000)
    assert result['atoms']['P10.arrive']['idle_us'] == .30
    assert result['atoms']['P4/P6/P7.setup']['idle_us'] == .15
    assert result['atoms']['P5/P11.observation']['idle_us'] == .951
    s = session(tmp_path); s.certified.add('kept')
    assert h.first_stop(s, result, valid_perturbation()) is passes
    assert s.rows['A.stop']['status'] == ('PASS' if passes else 'STOP')


def test_140_minute_cap_and_22_minute_b_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(h.time, 'monotonic', lambda: 100.)
    s = session(tmp_path); s.certified.add('kept'); s.m['B'] = []
    assert h.MAX_SESSION_SECONDS == h.JOBS[0]['timeout'] == 8400
    assert s.end-s.start == 8390  # existing ten-second reporting margin
    assert h.tranche_b(s) == []
    assert s.deadline == 1300  # X5 already reserved 120s of the 1320s B tranche
    limited = h.Session({}, tmp_path/'limited', seconds=1000)
    assert limited.end-limited.start == 990


# The --dry-run cap validation is executed in tests/e2e/test_api_qwen300_hour.py.


def test_full_hour_cap_reaches_pinned_image_preflight(tmp_path):
    hf = tmp_path/'hf'; hf.mkdir()
    (hf/'config.json').write_text(json.dumps(dict(model_type='qwen3_moe', num_hidden_layers=48,
        hidden_size=2048, num_key_value_heads=4, num_experts=128, num_experts_per_tok=8,
        torch_dtype='bfloat16')))
    gates = tmp_path/'gates.json'
    gates.write_text(json.dumps(dict(pytest=True, gfx942=True, dry_run=True, shell=True,
        local_matrix_16=True, rocm_build_check=True, source_hashes=h.source_hashes())))
    config = dict(hf=str(hf), model=str(tmp_path), cpu_gates=str(gates), image='wrong-image')
    code = f'''
import types
from examples.sessions import qwen300_hour as h
from examples.vllm_llm.qwen300_config import IMAGE
assert h.IMAGE == IMAGE
assert 'examples/vllm_llm/qwen300_config.py' in h.source_hashes()
try:
    h.preflight(types.SimpleNamespace(c={config!r}))
except ValueError as exc:
    assert str(exc) == 'strongest legal reference requires pinned vLLM image', str(exc)
else:
    raise AssertionError('unpinned image accepted')
'''
    result = subprocess.run([sys.executable, '-c', code],
                            env=dict(os.environ, ETX_SESS_MAX_SECONDS='8400'),
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_s2_does_not_attribute_pending_reset_device_work_to_host(tmp_path):
    s = session(tmp_path); s.certified.add('kept')
    assert h.first_stop(s, dict(G_ms=.260), valid_perturbation())
    s.rows['A3.host'] = dict(value=h.stage_prices(stage_text()))
    assert not h.second_stop(s)
    assert s.rows['B.stop']['value']['H_ms'] == pytest.approx(.046)


def wait_arms():
    return [a for a in h.manifest()['C'] if a['id'] in ('NO_TOUCH', 'ISOLATE_POLLER')]


def test_wait_arms_are_registered_staged_and_budgeted():
    a, b = wait_arms()
    assert a['defines'] == ['ETX_NO_TOUCH']
    assert b['flags']['ETX_PREFETCH_ISOLATE_POLLER'] == '1'
    assert a['diagnostic_build'] in h.template()['builds']
    assert all(x['build'] in h.template()['builds'] for x in (a, b))
    assert a['enabled'] is True and a['budget_s'] == 1620
    assert a['estimated_required_s'] == 1575 < a['budget_s']
    assert b['enabled'] is True and b['estimated_required_s'] == 1140
    assert b['budget_s'] == 1200
    assert sum(x['estimated_required_s'] for x in (a, b)) < h.TRANCHE_SECONDS['C']
    for x in (a, b):
        assert x['keep_threshold']['complete_adjacent_pinned_pairs'] == 3
        assert x['keep_threshold']['median_wall_gain_ms_ge'] == .030
        assert 'C.'+x['id'] in h.REGISTERED


@pytest.mark.parametrize('partial', [False, True])
def test_c_wait_pairs_are_adjacent_and_partial_never_has_aggregate(tmp_path, monkeypatch, partial):
    monkeypatch.setattr(h.time, 'monotonic', lambda: 100.)
    s = session(tmp_path); s.certified.add('kept')
    checked = []
    def eligible(s, a):
        checked.append(a['id'])
        return a
    monkeypatch.setattr(h, 'c_eligibility', eligible)
    s.qualified_c = h.qualify_c(s)
    assert checked == ['NO_TOUCH', 'ISOLATE_POLLER', 'C6']
    assert all(s.arms['C.eligibility.'+a]['status'] == 'QUALIFIED' for a in checked)
    calls = []; diagnostics = []
    monkeypatch.setattr(h, 'certify', lambda s, arm: s.certified.add(arm))
    monkeypatch.setattr(h, 'no_touch_diagnostic', lambda s, a: diagnostics.append(a['id']))
    monkeypatch.setattr(h, 'isolation_round_trip', lambda s, a: dict(median_us=1.9, pass_lt_2us=True))
    def fake_host(s, arm, name, args):
        calls.append((arm, s.deadline))
        if partial and name == 'C_NO_TOUCH_r2_NO_TOUCH': raise TimeoutError('missing final sample')
        return host_text(cost=4.18 if arm == 'kept' else 4.12 if arm == 'C6' else 4.14)
    monkeypatch.setattr(h, 'host', fake_host)
    survivors, extra = h.tranche_c(s)
    assert extra == 0
    assert diagnostics == ['NO_TOUCH']
    assert [arm for arm, _ in calls] == [who for arm in checked for who in ('kept', arm, arm, 'kept', 'kept', arm)]
    assert [t-s.start for _, t in calls] == [1620]*6+[1200]*12
    assert s.rows['C.budget']['status'] == 'ESTIMATE'
    assert s.rows['C.budget']['value']['available_s'] == 4020
    assert s.rows['C.budget']['value']['fits_estimate'] is True
    assert [x['id'] for x in survivors] == (['ISOLATE_POLLER', 'C6'] if partial else checked)
    if partial:
        assert s.rows['C.NO_TOUCH']['status'] == 'NOT-COMPLETE'
        assert s.rows['C.NO_TOUCH']['value'] is None
        assert 'median_ms' not in s.arms['NO_TOUCH']
    else:
        assert len(s.rows['C.NO_TOUCH']['value']['rounds']) == 3
        assert s.rows['C.NO_TOUCH']['status'] == 'KEEP'


@pytest.mark.parametrize('rt,status', [(1.999, 'KEEP'), (2.0, 'REJECT'), (2.01, 'REJECT'), (None, 'NOT-COMPLETE')])
def test_isolation_cannot_keep_on_wall_gain_alone(tmp_path, monkeypatch, rt, status, capsys):
    s = session(tmp_path); s.certified.add('kept'); s.qualified_c = [wait_arms()[1]]
    monkeypatch.setattr(h, 'certify', lambda *args: None)
    monkeypatch.setattr(h, 'host', lambda s, arm, name, args: host_text(cost=4.18 if arm == 'kept' else 4.14))
    def trip(s, a):
        if rt is None: raise ValueError('probe not staged')
        return dict(median_us=rt, pass_lt_2us=rt < 2.)
    monkeypatch.setattr(h, 'isolation_round_trip', trip)
    survivors, _ = h.tranche_c(s)
    assert s.rows['C.ISOLATE_POLLER']['status'] == status
    assert bool(survivors) == (status == 'KEEP')
    assert 'round trip <2.0 us' in s.rows['C.ISOLATE_POLLER']['rule']
    if status != 'KEEP': assert 'THRESHOLD C.ISOLATE_POLLER KEEP' not in capsys.readouterr().out


@pytest.mark.parametrize('mutation', ['none', 'partial', 'session', 'build', 'flag', 'nan'])
def test_isolation_requires_fresh_complete_probe_output(tmp_path, monkeypatch, mutation):
    from examples.vllm_llm import import_certificate as cert
    s = session(tmp_path); a = wait_arms()[1]; s.c['builds'][a['build']] = '/unused'
    exe = tmp_path/'probe'; exe.write_bytes(b'prepared probe')
    s.c['isolation_probe'] = dict(argv=[str(exe)], sha256=cert.sha(exe))
    monkeypatch.setattr(cert, 'verified_build', lambda *args: dict(frozen='candidate'))
    identity = cert.digest(dict(frozen='candidate'))
    rows = [dict(sample=i, session=str(s.out), build_sha256=identity, isolate_poller=1, round_trip_us=1.9) for i in range(3)]
    if mutation == 'partial': rows.pop()
    if mutation == 'session': rows[0]['session'] = 'old'
    if mutation == 'build': rows[0]['build_sha256'] = 'other'
    if mutation == 'flag': rows[0]['isolate_poller'] = 0
    if mutation == 'nan': rows[0]['round_trip_us'] = float('nan')
    monkeypatch.setattr(s, 'step', lambda *args, **kw: '\n'.join('POLLER-ROUND-TRIP '+json.dumps(r) for r in rows))
    if mutation == 'none': assert h.isolation_round_trip(s, a)['pass_lt_2us']
    else:
        with pytest.raises(ValueError): h.isolation_round_trip(s, a)


@pytest.mark.parametrize('mutation', ['none', 'plan', 'workers', 'precision', 'flag', 'imports', 'touch_off', 'profile'])
def test_wait_variant_rejects_changed_plan_math_or_flags(tmp_path, monkeypatch, mutation):
    from examples.vllm_llm import import_certificate as cert
    s = session(tmp_path); a = wait_arms()[1]
    for arm in ('kept', a['build']):
        b = tmp_path/arm; b.mkdir(); s.c['builds'][arm] = str(b)
        (b/'plan.json').write_text(json.dumps(dict(workers_per_domain=32, precision='bf16')))
    base = dict(check_enabled=True, hipcc_flags=h.BASE_FLAGS,
                overrides=dict(ETX_VL_TOUCH='1', ETX_PREFETCH_ISOLATE_POLLER='0'), files=dict(imports='same'))
    actual = copy.deepcopy(base); actual['overrides']['ETX_PREFETCH_ISOLATE_POLLER'] = '1'
    if mutation in ('plan', 'workers', 'precision'):
        (tmp_path/a['build']/'plan.json').write_text(json.dumps(dict(workers_per_domain=31, precision='fp8' if mutation == 'precision' else 'bf16')))
    if mutation == 'flag': actual['overrides']['ETX_VL_UC'] = '1'
    if mutation == 'imports': actual['files']['imports'] = 'changed'
    if mutation == 'touch_off': base['overrides']['ETX_VL_TOUCH'] = '0'
    if mutation == 'profile': actual['hipcc_flags'] += ' -DETX_HOUR_PROFILE'
    monkeypatch.setattr(h, 'verify_seal', lambda *args: dict(flags=dict(ETX_INBODY_BUILD='0')))
    monkeypatch.setattr(cert, 'verified_build', lambda b, m: base if b.name == 'kept' else actual)
    if mutation == 'none': assert h.verify_wait_variant(s, a) == actual
    else:
        with pytest.raises(ValueError): h.verify_wait_variant(s, a)


def test_all_c_arms_fit_full_checks_and_pairs_before_reserved_d(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(h.time, 'monotonic', lambda: clock[0])
    s = session(tmp_path)
    s.certified.add('kept')
    s.qualified_c = s.m['C']
    clock[0] += h.TRANCHE_SECONDS['A'] + h.TRANCHE_SECONDS['B']
    calls = []
    def spend(seconds):
        clock[0] += seconds
        assert clock[0] <= s.deadline <= s.end
    def certify(s, arm):
        if arm not in s.certified:
            for gate in ('p0', 'p1040', 'dsl_tokens'):
                calls.append((arm, gate)); spend(120)
            s.certified.add(arm)
    monkeypatch.setattr(h, 'certify', certify)
    def diagnostic(s, a):
        certify(s, a['diagnostic_build'])
        spend(120); spend(45)
    monkeypatch.setattr(h, 'no_touch_diagnostic', diagnostic)
    def host(s, arm, name, args):
        calls.append((arm, name)); spend(120)
        return host_text(cost=4.18 if arm == 'kept' else 4.12)
    monkeypatch.setattr(h, 'host', host)
    def round_trip(s, a):
        spend(60)
        return dict(median_us=1.9, pass_lt_2us=True)
    monkeypatch.setattr(h, 'isolation_round_trip', round_trip)
    survivors, _ = h.tranche_c(s)
    assert [a['id'] for a in survivors] == ['NO_TOUCH', 'ISOLATE_POLLER', 'C6']
    for a in survivors:
        arm = a['build']
        assert [gate for who, gate in calls if who == arm and not gate.startswith('C_')] == ['p0', 'p1040', 'dsl_tokens']
        assert len(s.rows['C.'+a['id']]['value']['rounds']) == 3
    assert s.end-clock[0] >= h.TRANCHE_SECONDS['D']


def test_enabled_c6_without_staged_prototype_remains_incomplete(tmp_path):
    s = session(tmp_path)
    s.m['C'] = [a for a in s.m['C'] if a['id'] == 'C6']
    assert s.m['C'][0]['enabled']
    assert h.qualify_c(s) == []
    assert s.arms['C.eligibility.C6']['status'] == 'NOT-COMPLETE'
    assert 'C6' not in s.certified


# The --template command is executed in tests/e2e/test_api_qwen300_hour.py.


@pytest.mark.parametrize('arm', ['C6', 'reset'])
@pytest.mark.parametrize('key', ['builds', 'build_seals', 'build_commits'])
def test_optional_prepared_arm_names_missing_key(tmp_path, arm, key):
    config = h.template()
    config['build_seals'][arm] = 'sealed'
    config['build_commits'][arm] = 'committed'
    config[key].pop(arm)
    s = h.Session(config, tmp_path)
    with pytest.raises(ValueError, match=rf'{arm}.*missing {key}\.{arm}'):
        h.verify_seal(s, arm)


def test_template_unprepared_optional_arms_have_named_rows(tmp_path, capsys):
    s = h.Session(h.template(), tmp_path)
    s.m['C'] = [a for a in s.m['C'] if a['id'] == 'C6']
    assert h.qualify_c(s) == []
    assert h.registered_reset(s) is None
    for name, arm in [('C.eligibility.C6', 'C6'), ('RESET.keep', 'reset')]:
        row = s.rows[name]
        assert row['status'] == 'NOT-COMPLETE'
        assert f'missing build_seals.{arm}' in row['value']['reason']
    assert 'KeyError' not in capsys.readouterr().out


@pytest.mark.parametrize('key', ['prototype_certificate', 'prototype_commit', 'expected_vgpr', 'expected_lds_bytes'])
def test_optional_prototype_names_missing_field(tmp_path, monkeypatch, key):
    s = h.Session(h.template(), tmp_path)
    arm = dict(id='C6', build='C6', kind='fusion', prototype_certificate='/unused',
               prototype_commit='a'*40, expected_vgpr=128, expected_lds_bytes=29752)
    arm.pop(key)
    monkeypatch.setattr(h, 'eligible_arm', lambda *args: None)
    with pytest.raises(ValueError, match=f'C6.*missing {key}'):
        h.c_eligibility(s, arm)
