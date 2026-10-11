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


def test_disabled_c_lends_exactly_twenty_minutes(tmp_path):
    s = session(tmp_path)
    assert h.tranche_c(s) == ([], 1200)
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
def test_d_runs_five_adjacent_pinned_pairs_per_baseline(tmp_path, monkeypatch, partial):
    s = session(tmp_path); s.certified.add('kept'); calls = []
    monkeypatch.setattr(h, 'certify', lambda *args: None)
    def fake_host(s, arm, name, args):
        mode = 'graph' if '--unfused' in args else 'fused'; calls.append(mode)
        if partial and name == 'D_fusion_r4_graph': raise TimeoutError('synthetic deadline')
        text = host_text(mode, 3.4 if mode == 'graph' else 3.3)
        (s.out/(name+'.log')).write_text('# job '+name+' gpu=0\n'+text)
        return text
    monkeypatch.setattr(h, 'host', fake_host)
    def fake_system(s, name): calls.append('S3'); return dict(wall_ms=3.5)
    monkeypatch.setattr(h, 'system', fake_system)
    h.tranche_d(s, [], 1200)
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
    monkeypatch.setattr(h, 'tranche_a', lambda s: False)
    monkeypatch.setattr(h, 'tranche_b', lambda s: pytest.fail('B must not run after stop'))
    h.execute(s)
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


def test_missing_staging_still_prints_all_registered_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('ETX_QWEN300_CONFIG', str(tmp_path/'missing.json'))
    monkeypatch.setenv('ETX_SESS_LOGDIR', str(tmp_path/'out'))
    assert h.main(['--run']) == 2
    report = json.loads((tmp_path/'out/report.json').read_text())
    assert set(h.REGISTERED) <= report['thresholds'].keys()
    assert all(r['value'] is None for r in report['thresholds'].values())
    assert 'REGISTERED THRESHOLDS' in capsys.readouterr().out


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
    def fake_pairs(s, a, tranche):
        result = dict(h.keep([.040, .040, .040]), status='KEEP')
        s.arms[a['id']] = result
        return result
    monkeypatch.setattr(h, 'candidate_pairs', fake_pairs)
    assert h.tranche_c(s) == ([], 0)
    assert s.arms['proto']['status'] == 'REJECT'
    assert s.rows['C.fusion']['status'] == 'NO-KEEP'
    assert s.rows['C.fusion']['value']['proto']['native_resources']['lds_bytes'] == 29736


def test_analysis_cli_emits_small_decision_view_under_its_deadline(tmp_path):
    from etx.tools.inbody import MAGIC
    plan, raw = trace()
    p = tmp_path/'plan.json'; p.write_text(json.dumps(plan))
    words = [MAGIC, 1, 3, 1, 1, 10]
    for k in ('legacy', 'poll', 'events', 'waves'): words += [v for row in raw[k] for v in row]
    r = tmp_path/'raw.bin'; r.write_bytes(struct.pack('<'+'Q'*len(words), *words))
    out = tmp_path/'edges.json'
    assert h.main(['--analyze', str(p), str(r), str(out)]) == 0
    full = json.loads(out.read_text()); small = json.loads(out.with_suffix('.summary.json').read_text())
    assert full['inbody']['raw'] == raw
    assert small['critical_pools'] == full['critical_pools']
    assert 'raw' not in small['inbody']
    assert small['inbody']['phases'] == full['inbody']['phases']


@pytest.mark.parametrize('g,passes', [(None, False), (.087999, False), (.088, True), (.100, True), (.324, True)])
@pytest.mark.parametrize('busy,free', [('PASS', 'PASS'), ('DEAD', 'UNRESOLVED')])
def test_s1_uses_gap_atoms_instead_of_pool_labels(tmp_path, g, passes, busy, free):
    s = session(tmp_path); s.certified.add('kept')
    s.rows.update({'A1.busy': dict(status=busy), 'A1.free': dict(status=free)})
    assert h.first_stop(s, None if g is None else dict(G_ms=g, atoms={})) is passes
    assert s.rows['A.stop']['status'] == ('PASS' if passes else 'STOP')
    assert ':301-302' in s.rows['A.stop']['source']


@pytest.mark.parametrize('g,r2,host,passes', [
    (.100, None, .030, False), (.088, .236, 0., True),
    (.100, .193, .030, False), (.100, .194, .030, True),
    (.324, None, 0., True), (.200, None, .135, False)])
def test_s2_combines_gap_attention_and_attributable_host(tmp_path, g, r2, host, passes):
    s = session(tmp_path); s.certified.add('kept')
    assert h.first_stop(s, dict(G_ms=g))
    s.rows['A2.route'] = dict(value=dict(attainable_cut_ms=r2, available_cut_estimate_ms=.900))
    s.rows['A3.host'] = dict(value=dict(named_host_ms=dict(launch_us=host, reset_submit_us=.049,
        kernel_outside_token_timers_ms=.900, instrument_other_ms=.900), accounted_beyond_priced_ms=.900))
    assert h.second_stop(s) is passes
    value = s.rows['B.stop']['value']
    assert value['R2_ms'] == (r2 or 0.)
    assert value['H_ms'] == pytest.approx(min(.086, host))
    assert value['total_ms'] == pytest.approx(g+(r2 or 0.)+value['H_ms'])
    assert ':306-308' in s.rows['B.stop']['source']


@pytest.mark.parametrize('g,runs_b,runs_cd', [(.087, False, False), (.100, True, False), (.324, True, True)])
def test_execute_obeys_both_stops_with_time_remaining(tmp_path, monkeypatch, g, runs_b, runs_cd):
    s = session(tmp_path); s.certified.add('kept'); calls = []
    monkeypatch.setattr(h, 'preflight', lambda s: None)
    monkeypatch.setattr(h, 'tranche_a', lambda s: h.first_stop(s, dict(G_ms=g)))
    def b(s): calls.append('B'); return ['body-keep']
    def c(s): calls.append('C'); return [], 1200
    def d(s, survivors, extra): calls.append('D')
    monkeypatch.setattr(h, 'tranche_b', b)
    monkeypatch.setattr(h, 'tranche_c', c)
    monkeypatch.setattr(h, 'tranche_d', d)
    h.execute(s)
    assert calls == (['B', 'C', 'D'] if runs_cd else ['B'] if runs_b else [])
    assert s.end-h.time.monotonic() > 3900
    if not runs_cd:
        assert all(s.rows[k]['status'] == 'NOT-RUN' for k in ('C.fusion', 'C.body', 'D.target', 'D.fusion', 'D.system'))
    if runs_b: assert s.rows['B.stop']['status'] == ('PASS' if runs_cd else 'STOP')


def test_gap_atoms_clip_to_critical_gap_and_apply_class_floor():
    def task(phase='qkv', critical=True):
        return dict(critical=critical, arrive_is_synthetic=False, phase=phase,
                    record=[0, 200, 300, 0, 120, 170], arrive_bounds_ticks=[90, 120], gap_us=10.)
    report = dict(raw=dict(tick_ns=100), tasks=[task() for _ in range(48)])
    # Gap starts at tick 100: only 2 us of arrive, 5 of observation, 3 of setup.
    result = h.gap_atom_prices(report)
    assert result['atoms']['P10.arrive']['above_floor_ms'] == pytest.approx(.096)
    assert result['atoms']['P5/P11.observation']['above_floor_ms'] == pytest.approx(48*(5-.951)/1000)
    assert result['atoms']['P4/P6/P7.setup']['above_floor_ms'] == pytest.approx(.144)
    report['tasks'] += [task('attn'), task('reduce'), task(critical=False)]
    assert h.gap_atom_prices(report) == result  # booked boundaries/parallel bodies excluded
    report['tasks'] = [task()]
    result = h.gap_atom_prices(report)
    assert result['G_ms'] == 0.  # every class below 0.625 us/layer; raw excess retained
    assert result['atoms']['P10.arrive']['above_floor_ms'] > 0.


def test_67_minute_cap_and_22_minute_b_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(h.time, 'monotonic', lambda: 100.)
    s = session(tmp_path); s.certified.add('kept'); s.m['B'] = []
    assert h.MAX_SESSION_SECONDS == h.JOBS[0]['timeout'] == 4020
    assert s.end-s.start == 4010  # existing ten-second reporting margin
    assert h.tranche_b(s) == []
    assert s.deadline == 1420
    limited = h.Session({}, tmp_path/'limited', seconds=1000)
    assert limited.end-limited.start == 990


@pytest.mark.parametrize('seconds,accepted', [('4020', True), ('3600', True), ('4021', False), ('0', False), ('nan', False)])
def test_cap_environment_validation(seconds, accepted):
    env = dict(os.environ, ETX_SESS_MAX_SECONDS=seconds)
    result = subprocess.run([sys.executable, '-c', 'from examples.sessions import qwen300_hour'],
                            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert (result.returncode == 0) is accepted
    if not accepted: assert 'cap must be positive and <=4020 seconds' in result.stdout


def test_s2_does_not_attribute_pending_reset_device_work_to_host(tmp_path):
    s = session(tmp_path); s.certified.add('kept')
    assert h.first_stop(s, dict(G_ms=.260))
    s.rows['A3.host'] = dict(value=h.stage_prices(stage_text()))
    assert not h.second_stop(s)
    assert s.rows['B.stop']['value']['H_ms'] == pytest.approx(.046)
