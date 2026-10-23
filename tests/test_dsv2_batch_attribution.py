"""Lightweight CPU contracts for the registered B4 follow-up; no simulated result is a GPU measurement."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from etx.models.dsv2lite import batch_attribution as a
from etx.models.dsv2lite.batch_session import BatchSession
from examples.dsv2lite.routing_overlap import validate_prefix

ROOT = Path(__file__).resolve().parents[1]


def row(i=0, value='10', start=10, end=20):
    return dict(Counter_Name='FETCH_SIZE', Counter_Value=value, Kernel_Name='etx_megakernel_d0',
                Dispatch_Id=str(i), Agent_Id='1', Process_Id='42', Start_Timestamp=str(start), End_Timestamp=str(end))


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_measured_traffic_requires_dispatches_and_finite_values():
    got = a.traffic([row(), row(1, '20', 30, 60)], expected=2)
    assert got['total_fetch_kb'] == 30 and got['dispatches'] == 2
    assert got['dispatch_duration_sum_us'] == .04
    assert got['device_span_us'] == .05
    for rows, match in [([], 'missing'), ([row(), row()], 'duplicate'), ([row(value='nan')], 'nonfinite'),
                         ([row(value='0')], 'nonpositive'), ([row(end=1)], 'span')]:
        with pytest.raises(ValueError, match=match):
            a.traffic(rows)
    with pytest.raises(ValueError, match='expected 64'):
        a.traffic([row()], expected=64)


def marker_fixture(tmp_path, b=4):
    phases = ['warmup','warmup_short', *(p for k in range(5) for p in (f'sample{k}',f'sample{k}_short'))]
    markers, counters = [], []
    for i, phase in enumerate(phases):
        markers.append(dict(Function=f'ds_b4:b{b}:{phase}', Start_Timestamp=str(i*100), End_Timestamp=str(i*100+90)))
        counters.append(row(i, '10' if phase.endswith('short') else '41', i*100+10, i*100+80))
    write_csv(tmp_path/'test_marker_api_trace.csv', markers)
    write_csv(tmp_path/'test_counter_collection.csv', counters)
    write_csv(tmp_path/'test_kernel_trace.csv', [
        {k: v for k, v in r.items() if k not in ('Counter_Name', 'Counter_Value')} for r in counters])
    return markers, counters


def vllm_log(config, batches, legal, failed_phase='sample0_short'):
    lines = []
    for b in batches:
        for phase in ['warmup', 'warmup_short', *(p for k in range(5) for p in (f'sample{k}', f'sample{k}_short'))]:
            n = 1 if phase.endswith('short') else 32
            for r in range(b):
                ok = b in legal or phase != failed_phase or r != 0
                lines.append(f'VLLMSEQ label={config} batch={b} phase={phase} seq={r} '
                             f'first_divergence={n if ok else 0}/{n} tokens={n}')
        lines.append(f'VLLMB label={config} batch={b} per_token_ms=4,4,4,4,4')
    return '\n'.join(lines)


@pytest.mark.parametrize('b1_config', ['default', 'aiter'])
def test_review_r1_mixed_legal_subsets_complete_all_selected_pairs(tmp_path, monkeypatch, b1_config):
    s = bare_session(tmp_path)
    s.root, s.data = ROOT, tmp_path/'references'
    monkeypatch.setattr(s, 'require_gates', lambda: None)
    other = 'aiter' if b1_config == 'default' else 'default'
    assigned = {b1_config: (1,), other: (2, 4)}
    (tmp_path/'vsel.json').write_text(json.dumps(dict(selection={
        str(b): dict(config=b1_config if b == 1 else other) for b in a.BATCHES})))
    for b in a.BATCHES:
        (tmp_path/f'graph_gate_{a.BATCH_ARMS[b]}.json').write_text('{"passed":false}')
    monkeypatch.setattr(s, 'etx', lambda b, name: dict(step_ms='3', latency_ms='3'))
    calls = []

    def run(name, command, **kw):
        config = next(c for c in assigned if name.endswith('_'+c))
        batches = tuple(map(int, command[command.index('--batches')+1].split(',')))
        calls.append((config, batches))
        return vllm_log(config, batches, assigned[config])

    monkeypatch.setattr(s, 'run', run)
    s.rounds()
    s.check_rounds()
    result = json.loads((tmp_path/'brounds.json').read_text())
    assert result['status'] == 'COMPLETE' and len(result['rounds']) == 5
    assert all(bs == assigned[c] for c, bs in calls)
    assert all(calls.count((c, bs)) == 5 for c, bs in assigned.items())


@pytest.mark.parametrize('phase', ['warmup', 'warmup_short', 'sample2', 'sample2_short'])
def test_selected_call_failure_stops_rounds_and_capture_precondition(tmp_path, monkeypatch, phase):
    s = bare_session(tmp_path)
    s.root, s.data = ROOT, tmp_path/'references'
    monkeypatch.setattr(s, 'require_gates', lambda: None)
    (tmp_path/'vsel.json').write_text(json.dumps(dict(selection={
        str(b):dict(config='default' if b == 1 else 'aiter') for b in a.BATCHES})))
    for b in a.BATCHES:
        (tmp_path/f'graph_gate_{a.BATCH_ARMS[b]}.json').write_text('{"passed":false}')
    monkeypatch.setattr(s, 'etx', lambda b, name: dict(step_ms='3', latency_ms='3'))
    calls = []

    def run(name, command, **kw):
        calls.append(name)
        config = 'default' if name.endswith('_default') else 'aiter'
        batches = tuple(map(int, command[command.index('--batches')+1].split(',')))
        return vllm_log(config, batches, {1, 2}, failed_phase=phase)

    monkeypatch.setattr(s, 'run', run)
    with pytest.raises(RuntimeError, match='remaining timings and captures stopped'):
        s.rounds()
    assert len(calls) == 2
    with pytest.raises(ValueError, match='before attribution'):
        s.check_rounds()
    with pytest.raises(ValueError, match='before attribution'):
        s.captures()
    assert len(calls) == 2
    assert json.loads((tmp_path/'attribution.json').read_text())['status'] == 'INCOMPLETE'


@pytest.mark.parametrize('reported', [(), (1, 2)])
def test_selected_vllm_requires_exact_invoked_batch_outputs(tmp_path, monkeypatch, reported):
    s = bare_session(tmp_path)
    s.root, s.data = ROOT, tmp_path/'references'
    monkeypatch.setattr(s, 'run', lambda *args, **kw: vllm_log('aiter', reported, {1, 2}))
    result = s.vllm('aiter', 'selected', batches=(2,))
    assert 'error' in result and 'invoked subset' in result['error']


def test_review_r2_missing_one_counter_of_two_traced_dispatches_is_refused(tmp_path):
    _, counters = marker_fixture(tmp_path)
    trace = [{k:v for k,v in r.items() if k not in ('Counter_Name', 'Counter_Value')} for r in counters]
    trace.append({k:v for k,v in row(100, start=240, end=260).items() if k not in ('Counter_Name', 'Counter_Value')})
    write_csv(tmp_path/'test_kernel_trace.csv', trace)
    with pytest.raises(ValueError, match='dispatch coverage'):
        a.ranged_traffic(tmp_path, 4)


def test_review_r2_partial_dispatch_capture_stays_incomplete(tmp_path, monkeypatch):
    from etx.tools import timeline
    s = bare_session(tmp_path)
    s.capture_batch = None
    monkeypatch.setattr(s, 'require_gates', lambda: None)
    monkeypatch.setattr(s, 'check_rounds', lambda: None)
    monkeypatch.setattr(s, 'identity', lambda b: dict(batch=b))
    monkeypatch.setattr(s, 'out', lambda arm: tmp_path/arm)
    monkeypatch.setattr(s, 'etx_args', lambda b: [])
    monkeypatch.setattr(a.batch, 'etx_result', lambda *args: {})
    for f in ('load_plan', 'read_raw', 'build_trace', 'latency_report'):
        monkeypatch.setattr(timeline, f, lambda *args: {})
    (tmp_path/'vsel.json').write_text(json.dumps(dict(selection={str(b):dict(config='aiter') for b in a.BATCHES})))
    for b in a.BATCHES:
        (tmp_path/f'bgate_{a.BATCH_ARMS[b]}.json').write_text('{"passed":true}')

    def run(name, command, **kw):
        if name.startswith('traffic_'):
            directory = tmp_path/name
            directory.mkdir()
            write_csv(directory/'test_counter_collection.csv', [row(i, start=i*100+10, end=i*100+20) for i in range(64)])
        return ''

    def vllm(config, name, **kw):
        b = s.capture_batch
        directory = tmp_path/name
        directory.mkdir()
        _, counters = marker_fixture(directory, b)
        trace = [{k:v for k,v in r.items() if k not in ('Counter_Name', 'Counter_Value')} for r in counters]
        trace.append({k:v for k,v in row(100, start=240, end=260).items() if k not in ('Counter_Name', 'Counter_Value')})
        write_csv(directory/'test_kernel_trace.csv', trace)
        return {b: dict(legal=True)}

    monkeypatch.setattr(s, 'run', run)
    monkeypatch.setattr(s, 'vllm', vllm)
    with pytest.raises(ValueError, match='dispatch coverage'):
        s.captures()
    result = json.loads((tmp_path/'attribution.json').read_text())
    assert result['status'] == 'INCOMPLETE' and 'dispatch coverage' in result['error']
    assert s.capture_batch is None


def test_vllm_ranges_preserve_measured_calls_and_label_subtraction(tmp_path):
    marker_fixture(tmp_path)
    got = a.ranged_traffic(tmp_path, 4)
    assert len(got['calls']) == 12
    assert got['decode_fetch_kb_estimate'] == [1]*5
    assert 'not independently proved' in got['limitation']
    assert all(r['dispatch_coverage']['status'] == 'COMPLETE' for r in got['calls'].values())


@pytest.mark.parametrize('defect', ['extra_counter', 'duplicate_counter', 'duplicate_trace', 'wrong_process',
                                  'wrong_agent', 'wrong_dispatch', 'timestamp', 'trace_crossing',
                                  'missing_trace', 'missing_scope', 'overlapping_ranges'])
def test_dispatch_coverage_refuses_unmatched_scopes_and_records(tmp_path, defect):
    markers, counters = marker_fixture(tmp_path)
    trace = [{k:v for k,v in r.items() if k not in ('Counter_Name', 'Counter_Value')} for r in counters]
    if defect == 'extra_counter':
        counters.append(row(100, start=240, end=260))
    elif defect == 'duplicate_counter':
        counters.append(counters[2].copy())
    elif defect == 'duplicate_trace':
        trace.append(trace[2].copy())
    elif defect in ('wrong_process', 'wrong_agent', 'wrong_dispatch', 'missing_scope'):
        field = {'wrong_process':'Process_Id', 'wrong_agent':'Agent_Id',
                 'wrong_dispatch':'Dispatch_Id', 'missing_scope':'Process_Id'}[defect]
        counters[2][field] = '' if defect == 'missing_scope' else '999'
    elif defect == 'timestamp':
        counters[2]['End_Timestamp'] = '279'
    elif defect == 'trace_crossing':
        trace[2]['Start_Timestamp'] = '199'
    elif defect == 'overlapping_ranges':
        markers[2]['Start_Timestamp'] = '180'
    write_csv(tmp_path/'test_counter_collection.csv', counters)
    write_csv(tmp_path/'test_marker_api_trace.csv', markers)
    write_csv(tmp_path/'test_kernel_trace.csv', trace)
    if defect == 'missing_trace':
        (tmp_path/'test_kernel_trace.csv').unlink()
    with pytest.raises(ValueError):
        a.ranged_traffic(tmp_path, 4)


def test_complete_zero_measurements_and_scoped_dispatch_ids_are_preserved(tmp_path):
    _, counters = marker_fixture(tmp_path)
    # Same numeric dispatch ID in another agent/process is a distinct dispatch.
    for field in ('Agent_Id', 'Process_Id'):
        extra = row(2, value='0', start=240, end=260)
        extra[field] = '99'
        counters.append(extra)
    # A measured all-zero short range is present, not a missing record.
    counters[3]['Counter_Value'] = '0'
    write_csv(tmp_path/'test_counter_collection.csv', counters)
    write_csv(tmp_path/'test_kernel_trace.csv', [
        {k:v for k,v in r.items() if k not in ('Counter_Name', 'Counter_Value')} for r in counters])
    got = a.ranged_traffic(tmp_path, 4)
    assert got['calls']['sample0']['dispatches'] == 3
    assert got['calls']['sample0_short']['total_fetch_kb'] == 0
    assert got['calls']['sample0_short']['dispatch_coverage']['measured'] == 1
    assert got['decode_fetch_kb_estimate'][0] == 41/31
    assert 'estimate' in got['limitation']


@pytest.mark.parametrize('defect', ['missing','crossing','nonpositive'])
def test_vllm_ranges_refuse_incomplete_attribution(tmp_path, defect):
    markers, counters = marker_fixture(tmp_path)
    if defect == 'missing':
        write_csv(tmp_path/'test_marker_api_trace.csv', markers[:-1])
    else:
        counters[2]['Start_Timestamp' if defect == 'crossing' else 'Counter_Value'] = '199' if defect == 'crossing' else '1'
        write_csv(tmp_path/'test_counter_collection.csv', counters)
    with pytest.raises(ValueError):
        a.ranged_traffic(tmp_path, 4)


def test_exact_routing_prefix_cannot_use_a_rotated_or_shortened_trajectory():
    validate_prefix([[1,2,3],[4,5,6]], [[1,2,3,7],[4,5,6,8]], 2)
    for actual in ([[1,2],[4,5]], [[1,2,4],[4,5,6]], [[4,5,6],[1,2,3]]):
        with pytest.raises(ValueError, match='bound timed prefix'):
            validate_prefix(actual, [[1,2,3],[4,5,6]], 2)


def bare_session(tmp_path):
    s = a.AttributionSession.__new__(a.AttributionSession)
    s.logs = tmp_path
    return s


# The final lane is the `final` command in tests/e2e/test_api_dsv2_attribution_final.py.

def test_wrong_binary_or_failed_gate_stops_performance(tmp_path, monkeypatch):
    s = bare_session(tmp_path)
    monkeypatch.setattr(s, 'identity', lambda b: {'b': b})
    for b in a.BATCHES:
        for kind in ('bgate','graph_gate'):
            (tmp_path/f'{kind}_{a.BATCH_ARMS[b]}.json').write_text(json.dumps(dict(passed=True, identity={'b':b})))
    s.require_gates()
    p = tmp_path/'bgate_x4.json'
    p.write_text(json.dumps(dict(passed=False, identity={'b':4})))
    with pytest.raises(ValueError, match='no timing'):
        s.require_gates()
    p.write_text(json.dumps(dict(passed=True, identity={'b':'other binary'})))
    with pytest.raises(ValueError, match='identity'):
        s.require_gates()


def test_b4_reuses_five_rounds_with_no_eight_request_dependency(tmp_path, monkeypatch):
    s = bare_session(tmp_path)
    monkeypatch.setattr(s, 'require_gates', lambda: None)
    (tmp_path/'vsel.json').write_text(json.dumps(dict(selection={str(b):dict(config='aiter') for b in a.BATCHES})))
    for b in a.BATCHES:
        (tmp_path/f'bgate_{a.BATCH_ARMS[b]}.json').write_text('{"passed":true}')
        (tmp_path/f'graph_gate_{a.BATCH_ARMS[b]}.json').write_text('{"passed":true}')
    calls = []
    monkeypatch.setattr(s, 'etx', lambda b, name: calls.append(('etx',b)) or {'step_ms':'3'})
    monkeypatch.setattr(s, 'graph', lambda b, name: calls.append(('graph',b)) or {'step_ms':'4'})
    monkeypatch.setattr(s, 'vllm', lambda c, name, *, batches: calls.append(('vllm',c)) or {b:{'legal':True} for b in batches})
    s.rounds()
    order = [(kind,b) for b in (1,2,4) for kind in ('etx','graph')] + [('vllm','aiter')]
    assert calls == order + order[::-1] + order + order[::-1] + order
    assert BatchSession.batches == (1,2,4,8)
    result = json.loads((tmp_path/'brounds.json').read_text())
    assert len(result['rounds']) == 5 and result['skipped'] == []
    assert all(x['status'] == 'COMPLETE' and x['throughput']['outcome'] == 'WIN' for x in result['same_kernel_graph'].values())


def test_session_declares_all_producers_and_bounded_cpu_gpu_paths():
    spec = importlib.util.spec_from_file_location('b4_test_session', ROOT/'examples/sessions/ds_b4_attribution.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    jobs = {j['name']:j for j in mod.JOBS}
    assert mod.MAX_SESSION_SECONDS == 7200
    assert not any('x8' in j['name'] for j in mod.JOBS)
    assert jobs['b4_result']['lane'] == 'final' and '?captures' in jobs['b4_result']['reads']
    assert jobs['captures']['reads'][0:2] == ['vsel.json','brounds.json']
    assert 'prep_ref' not in jobs['prep_pack']['reads']
    for b in a.BATCHES:
        arm = a.BATCH_ARMS[b]
        assert f'b_{arm}/run' in jobs['bgate_'+arm]['reads']
    for j in mod.JOBS:
        assert 0 < j['timeout'] <= mod.MAX_SESSION_SECONDS


def test_first_timed_failure_stops_remaining_pairs(tmp_path, monkeypatch):
    s = bare_session(tmp_path)
    monkeypatch.setattr(s, 'require_gates', lambda: None)
    (tmp_path/'vsel.json').write_text(json.dumps(dict(selection={str(b):dict(config='aiter') for b in a.BATCHES})))
    for b in a.BATCHES:
        (tmp_path/f'graph_gate_{a.BATCH_ARMS[b]}.json').write_text('{"passed":false}')
    monkeypatch.setattr(s, 'etx', lambda *args: {'error':'hidden gate failed'})
    monkeypatch.setattr(s, 'graph', lambda *args: pytest.fail('timing after hidden failure'))
    monkeypatch.setattr(s, 'vllm', lambda *args: pytest.fail('timing after hidden failure'))
    with pytest.raises(RuntimeError, match='remaining timings'):
        s.rounds()
    result = json.loads((tmp_path/'brounds.json').read_text())
    assert result['status'] == 'INCOMPLETE' and len(result['rounds']) == 1
    assert all(x['status'] == 'UNAVAILABLE' for x in result['same_kernel_graph'].values())


def test_unfused_host_wall_does_not_report_device_event_time(tmp_path):
    import shutil
    import subprocess
    preferred = Path('/opt/homebrew/opt/llvm/bin/clang++')
    compiler = str(preferred) if preferred.exists() else shutil.which('c++')
    if compiler is None:
        pytest.skip('portable C++ compiler absent')
    source = (ROOT/'examples/dsv2lite/host.hip').read_text()
    block = source[source.index('      const auto step_begin ='):source.index('    }\n    for (int r = 0; r < R;', source.index('      const auto step_begin ='))]
    program = r"""
#include <chrono>
#include <thread>
#include <cstdio>
#include <vector>
struct Host { double last_ms=.001; bool run_unfused() { std::this_thread::sleep_for(std::chrono::milliseconds(3)); return true; }
 bool run(double) { std::this_thread::sleep_for(std::chrono::milliseconds(3)); return true; }};
int main() { for (bool unfused : {false,true}) { Host hst; int t=0; double wall_total=0; std::vector<double> step_ms;
""" + block + r"""
if (step_ms[0] < 1 || step_ms[0] == hst.last_ms) return 1;
} return 0; }
"""
    path = tmp_path/'clock.cpp'
    path.write_text(program)
    subprocess.run([compiler, '-std=c++17', str(path), '-o', str(tmp_path/'clock')], check=True, capture_output=True)
    subprocess.run([str(tmp_path/'clock')], check=True, timeout=10)


def test_routing_aggregate_uses_each_numerical_batch_run(tmp_path, monkeypatch):
    s = bare_session(tmp_path)
    s.data, s.fleet = tmp_path/'references', tmp_path/'fleet'
    monkeypatch.setattr(s, 'require_gates', lambda: None)
    monkeypatch.setattr(s, 'check_rounds', lambda: None)
    monkeypatch.setattr(s, 'identity', lambda b: dict(batch=b))
    calls = []
    def fake_run(name, command, **kw):
        calls.append(command)
        b = int(command[command.index('--batch')+1])
        path = Path(command[command.index('--json')+1])
        path.write_text(json.dumps(dict(batch=b, bound_prefix=True, reference_sha256={},
                                       executed_prefix={str(b): b*100, 'other_subset':-999})))
    monkeypatch.setattr(s, 'run', fake_run)
    s.routing()
    result = json.loads((tmp_path/'routing.json').read_text())
    assert result['executed_prefix'] == {'1':100,'2':200,'4':400}
    assert result['status'] == 'COMPLETE' and not result['native_etx_routing_measured']
    assert [c[c.index('--batch')+1] for c in calls] == ['1','2','4']
    assert all(c[c.index('-m')+1] == 'examples.dsv2lite.routing_overlap' for c in calls)


@pytest.mark.parametrize('symbol', ['fleet_expert_down_cached','fleet_d4_routed_down','unreviewed_body'])
def test_graph_baseline_refuses_lds_handoffs_and_unknown_tasks_before_gpu(tmp_path, monkeypatch, symbol):
    s = bare_session(tmp_path)
    monkeypatch.setattr(s, 'identity', lambda b: dict(batch=b))
    directory = tmp_path/'b_x2'
    directory.mkdir()
    (directory/'plan.json').write_text(json.dumps(dict(types={'down_1':{'symbol':symbol}})))
    monkeypatch.setattr(s, 'run', lambda *args, **kw: pytest.fail('known-invalid GPU graph launched'))
    result = s.graph(2, 'refused')
    assert result['status'] == 'UNAVAILABLE' and 'error' in result
    if symbol != 'unreviewed_body':
        assert 'LDS' in result['eligibility']['reason']


def test_refused_graph_baselines_do_not_block_corrected_vllm_attribution_rounds(tmp_path, monkeypatch):
    s = bare_session(tmp_path)
    monkeypatch.setattr(s, 'require_gates', lambda: None)
    (tmp_path/'vsel.json').write_text(json.dumps(dict(selection={str(b):dict(config='aiter') for b in a.BATCHES})))
    for b in a.BATCHES:
        (tmp_path/f'graph_gate_{a.BATCH_ARMS[b]}.json').write_text('{"passed":false,"result":{"status":"UNAVAILABLE"}}')
    calls = []
    monkeypatch.setattr(s, 'etx', lambda b, name: calls.append(('etx',b)) or {'step_ms':'3'})
    monkeypatch.setattr(s, 'vllm', lambda c, name, *, batches: calls.append(('vllm',c)) or {b:{'legal':True} for b in batches})
    monkeypatch.setattr(s, 'graph', lambda *args: pytest.fail('refused graph invoked'))
    s.rounds()
    assert len(calls) == 20
    result = json.loads((tmp_path/'brounds.json').read_text())
    assert result['status'] == 'COMPLETE'
    assert all(x['status'] == 'UNAVAILABLE' for x in result['same_kernel_graph'].values())


# The final lane is the `final` command in tests/e2e/test_api_dsv2_attribution_final.py.

def test_profiled_vllm_command_is_inside_pinned_container_and_single_batch(tmp_path, monkeypatch):
    s = bare_session(tmp_path)
    s.data, s.capture_batch = tmp_path/'references', 2
    s.root = ROOT
    commands = []
    monkeypatch.setattr(BatchSession, 'run', lambda self, name, command, **kw: commands.append(command) or '')
    s.vllm('aiter', 'capture', batches=(2,))
    command = commands[0]
    assert command[command.index('--batches')+1] == '2'
    assert '--profile-ranges' in command
    assert command.index('rocprofv3') > command.index(a.BatchSession.vllm.__globals__['IMAGE'])
    assert command[command.index('--pmc')+1] == 'FETCH_SIZE'
    assert len(command[command.index('--goldens')+1:command.index('--batches')]) == 2
