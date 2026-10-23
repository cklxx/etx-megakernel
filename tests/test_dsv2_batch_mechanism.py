"""CPU contracts of the DeepSeek batch-mechanism analysis and session; no simulated value is a GPU measurement."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from etx.tools.timeline import Task, Trace
from etx.models.dsv2lite import batch_mechanism as m

ROOT = Path(__file__).resolve().parents[1]
# what plan_stats lowers for the four arms (examples/dsv2lite/batch_mechanism.py, ~100 s locally); the offline test
# checks the retained traces against these, the lowering itself is exercised on a small graph below
PLANS = {1: dict(tasks=53022, slots=296, grids=271, requests_per_slot=1, tasks_per_slot_mean=179.1, tasks_per_slot_max=186),
         2: dict(tasks=109892, slots=592, grids=438, requests_per_slot=1, tasks_per_slot_mean=185.6, tasks_per_slot_max=219),
         4: dict(tasks=219784, slots=592, grids=876, requests_per_slot=2, tasks_per_slot_mean=371.3, tasks_per_slot_max=438),
         8: dict(tasks=439568, slots=592, grids=1752, requests_per_slot=4, tasks_per_slot_mean=742.5, tasks_per_slot_max=876)}


def test_span_identity_attributes_every_microsecond_of_the_retained_steps():
    o = m.offline(m.TRACES, PLANS)
    for key, s in o['steps'].items():
        assert abs(sum(s[n]['us'] for n in ('tasks', 'body_per_task', 'slots', 'busy')) - s['total_us']) < 1e-6, key
        assert abs(s['check_us']) < 1e-6
    b2, b4 = o['batches']['2'], o['batches']['4']
    assert (b2['span_us'], b4['span_us']) == (4339.2, 8670.16)
    assert b4['slots'] == b2['slots'] == 592
    assert 0.99 < b4['body_per_task_us'] / b2['body_per_task_us'] < 1.0
    assert abs(b4['busy'] - b2['busy']) < 0.005
    assert abs(o['steps']['2->4']['tasks']['us'] - 4339.2) < 1e-6        # doubling tasks at fixed rate = one B2 span
    assert o['projection_b8']['kind'].startswith('ESTIMATE')
    assert '| tasks per step | 53022 | 109892 | 219784 | 439568 |' in m.render(o)


def test_offline_refuses_a_trace_that_is_not_the_lowered_plan():
    plans = {**PLANS, 4: dict(PLANS[4], tasks=219783)}
    with pytest.raises(ValueError, match='B4: traced 219784 tasks'):
        m.offline(m.TRACES, plans)


def _trace(tasks, deps):
    by_id = {t.id: t for t in tasks}
    return Trace(tasks=tasks, by_id=by_id, type_names=[], n_domains=1, workers_per_domain=2, deps=deps,
                 t0=0.0, t1=max(t.done for t in tasks))


def test_worker_metrics_separate_dependency_wait_from_cross_request_queue_blocking():
    tasks = [Task(0, 0, 'r0_qkv_5', 0, 0, 0.0, 0.0, 10.0),      # slot 0: r0 body 0-10
             Task(1, 0, 'r2_qkv_5', 0, 0, 10.0, 10.0, 20.0),    # r2 inputs ready at 2, blocked 8 behind r0's task
             Task(2, 1, 'r0_attn_5', 0, 0, 20.0, 25.0, 30.0),   # r0 waits for its producer (task 3) until 25
             Task(3, 0, 'r1_qkv_5', 0, 1, 0.0, 0.0, 25.0),
             Task(4, 2, 'r1_norm_5', 0, 1, 25.0, 25.0, 27.0)]  # same request behind its own producer: neither
    deps = [[], [5], [3], [], [3]]
    tr = _trace(tasks, deps)
    tr.by_id[5] = Task(5, 3, 'r2_embed', 0, 2, 0.0, 0.0, 2.0)  # a producer outside the slot queues
    w = m.worker_metrics(tr)
    assert w['queue_cross_us'] == 8.0
    assert w['dep_wait_us'] == 5.0
    assert w['body_us'] == 10 + 10 + 5 + 25 + 2
    # a producer-free task behind another request's task is blocked from the window start
    tasks[4] = Task(4, 2, 'r3_embed', 0, 1, 25.0, 25.0, 27.0)
    assert m.worker_metrics(_trace(tasks, [[], [], [3], [], []]))['queue_cross_us'] == 10.0 + 25.0


def test_isolation_blocks_must_be_alone_other_slot_queued():
    tasks, i = [], 0
    spans = {'gate_up_5': (34.0, 52.0, 68.0), 'qkv_5': (20.0, 28.0, 40.0), 'attn_5': (15.0, 20.0, 30.0)}
    for p, phase in enumerate(m.ISO_PHASES):
        for k, (reqs, span) in enumerate(zip(((0,), (0, 1), (0, 2)), spans[phase])):
            for rep in range(2):
                for r in reqs:
                    tasks.append(Task(i, 0, f'iso{3 * p + k}x{rep}_r{r}_{phase}', 0, r % 2, 100.0 * i, 100.0 * i,
                                      100.0 * i + (span if r == reqs[-1] else 1.0)))
                    i += 1
    tr = _trace(tasks, None)
    blocks = m.iso_spans(tr)
    assert blocks['1']['requests'] == [0, 1] and blocks['2']['requests'] == [0, 2]
    with pytest.raises(ValueError, match='not A/B/C'):
        m.iso_ratios(dict(blocks, **{'2': dict(blocks['2'], requests=[0, 1])}))


def test_evaluate_applies_the_registered_rules_and_stays_incomplete_without_inputs(tmp_path):
    assert m.evaluate(tmp_path)['status'] == 'INCOMPLETE'

    def trace(body, busy, cross):
        return dict(status='COMPLETE', runs=[dict(trace=dict(slots=dict(body_us=body, body_share=busy,
                                                                     queue_cross_share=cross))) for _ in range(3)])
    (tmp_path / 'trace_x2.json').write_text(json.dumps(trace(1000.0, 0.75, 0.01)))
    (tmp_path / 'trace_x4.json').write_text(json.dumps(trace(2000.0, 0.751, 0.12)))
    for arm, fetch, dram, us in (('x2', 100.0, 10.0, 4.0), ('x4', 200.0, 20.0, 8.0)):
        (tmp_path / f'counters_{arm}.json').write_text(json.dumps(dict(groups=dict(
            fetch=dict(values=dict(FETCH_SIZE=dict(median=fetch, dispatch_us_median=us))),
            dram=dict(values=dict(TCC_EA0_RDREQ_DRAM_sum=dict(median=dram, dispatch_us_median=us)))))))
    ratios = {p: dict(alone_us=1, other_slot_us=1.5, same_slot_us=2, co_resident=1.5, queued=2.0) for p in m.ISO_PHASES}
    (tmp_path / 'iso_x4iso.json').write_text(json.dumps(dict(status='COMPLETE', ratios=ratios)))
    (tmp_path / 'trace_x8.json').write_text(json.dumps(dict(status='NOT-RUN', reason='bgate_x8 did not pass')))
    r = m.evaluate(tmp_path)
    p = r['predictions']
    assert p['P1_slot_serialization']['status'] == 'HOLDS'
    assert p['P2_traffic']['status'] == 'HOLDS'
    assert p['P3_queue_vs_co_residency']['branch'].startswith('RESIDENCY-CAP')
    assert r['b8']['status'] == 'NOT-RUN' and r['status'] == 'COMPLETE'


def test_counters_need_every_megakernel_dispatch(tmp_path):
    import csv
    d = tmp_path / 'c' / '0'
    d.mkdir(parents=True)
    rows = [dict(Counter_Name='FETCH_SIZE', Counter_Value='5', Kernel_Name='etx_megakernel_d0', Dispatch_Id=str(i),
                 Agent_Id='1', Process_Id='9', Start_Timestamp=str(1000 * i), End_Timestamp=str(1000 * i + 500))
            for i in range(m.DISPATCHES)]
    with (d / 'x_counter_collection.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows[:-1])
    with pytest.raises(ValueError, match='63 records'):
        m.counter_values(tmp_path / 'c', ('FETCH_SIZE',))


def test_isolation_arm_lowers_with_r2_queued_on_r0s_slot(monkeypatch):
    from etx.passes import PassOptions, compile_graph
    from examples.dsv2lite.arms import ARMS
    from examples.dsv2lite import graph
    assert ARMS['x4iso'].env['ETX_DSV2_ISOLATE'] == m.ISO_SPEC
    for k, v in ARMS['x4iso'].env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv('ETX_DSV2_ISOLATE_REPS', '1')
    g = graph.build(layers=6, routing_cached=True)
    plan = compile_graph(g, 'gfx942', {}, {}, PassOptions(relay='on'))
    per = plan.workers_per_domain
    slot = {}
    for t in plan.tasks:
        if t.grid.startswith('iso2x0_'):            # block C of gate_up: r0 and r2
            slot.setdefault(m._request(t.grid.split('_', 1)[1]), set()).add((t.worker - t.domain * per) // 38)
    assert slot == {0: {0}, 2: {0}}


def test_session_orders_correctness_microbenchmark_traces_then_counters():
    spec = importlib.util.spec_from_file_location('mech_session', ROOT / 'examples/sessions/ds_batch_mechanism.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = [j['name'] for j in mod.JOBS]
    gpu = [j['name'] for j in mod.JOBS if j['lane'] == 'gpu']
    assert gpu == ['references', 'bgate_x2', 'bgate_x4', 'iso_x4iso', 'trace_x2', 'trace_x4', 'bgate_x8', 'trace_x8',
                   'counters_x2', 'counters_x4', 'counters_x8']
    assert not any('vllm' in j['cmd'] or 'docker' in j['cmd'] for j in mod.JOBS)
    assert {'build_x2', 'build_x4', 'build_x8', 'build_x4iso'} <= set(names)
    assert next(j for j in mod.JOBS if j['name'] == 'report')['lane'] == 'final'
    assert '--requests 8' in next(j for j in mod.JOBS if j['name'] == 'references')['cmd']
    assert all(0 < j['timeout'] <= mod.MAX_SESSION_SECONDS for j in mod.JOBS)
