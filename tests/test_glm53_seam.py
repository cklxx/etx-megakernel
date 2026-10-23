"""CPU-only seam shape, receipt parser, CLI and bounded-runner tests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from etx.ir import Scope
from etx.passes import compile_graph
from etx.passes.plan import verify_plan
from etx.experiments.jobs import load_session
from etx.models.glm53 import seam

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('batch', seam.BATCHES)
def test_production_row_graph(batch):
    g = seam.build()
    p = compile_graph(g, 'gfx942', seam.bindings(B=batch), {})
    assert not verify_plan(p)
    assert p.n_devices == 2 and len(p.tasks) == 3*batch
    assert set(p.events) == {'ready', 'feedback'}
    for event in p.events.values():
        assert event.scope == Scope.SYSTEM and event.memory == 'fine_grained'
        assert event.shape == (batch,)
    assert p.tensor_placement['mailbox'] == p.tensor_placement['token'] == 'fine_grained'
    for grid in g.grids:
        assert grid.grid == ('B',)
        assert grid.resource.threads == 256 and grid.resource.vgpr == 64
        for edge in list(grid.in_edges.values()) + list(grid.out_edges.values()):
            assert edge.text == 'i->i'
    assert [grid.device for grid in g.grids] == [0, 1, 0]
    assert g.tensors['mailbox'].shape == ('B', 4, 4096)
    assert g.tensors['mailbox'].dtype == 'bf16'
    assert g.tensors['mailbox'].nbytes({'B': batch}) == batch*32768
    assert g.tensors['returned'].dtype == 'int32'
    assert 'input' not in g.tensors  # producer fills direct layer output, not host memcpy


@pytest.mark.parametrize('batch', seam.BATCHES)
@pytest.mark.parametrize('control', [False, True])
def test_full_row_oracle(batch, control):
    values, received, tokens = seam.oracle(batch, 123, control)
    assert len(values) == len(received) == batch*16384
    for r in range(batch):
        row = received[r*16384:(r+1)*16384]
        expected = values[r*16384:(r+1)*16384]
        assert row[0] == expected[0] and row[-1] == expected[-1]
        assert row[1:-1] == ([seam.POISON]*16382 if control else expected[1:-1])
        assert tokens[r] == expected[0]+expected[-1]
    assert values != seam.oracle(batch, 124, control)[0]


@pytest.mark.parametrize('batch', [0, 3, 16])
def test_invalid_batch(batch):
    with pytest.raises(ValueError):
        seam.bindings(B=batch)


def receipts(batch=2, trace=False, iterations=2, rounds=2, warmup=1):
    config = dict(schema=1, label=seam.LABEL, batch=batch, trace=trace,
                  iterations=iterations, rounds=rounds, warmup=warmup, row_bytes=32768)
    records = [('SEAM-CONFIG', config)]
    for rnd in range(rounds):
        for arm in ('payload', 'control'):
            for phase, n in (('warmup', warmup), ('sample', iterations)):
                for step in range(n):
                    x = 100 if arm == 'payload' else 80
                    rec = dict(round=rnd, arm=arm, phase=phase, iteration=step,
                               correct=True, rows_checked=batch, elements_checked=batch*16384,
                               wall_us=x+rnd+step, rank0_span_us=x-10, rank1_span_us=x-20)
                    if trace:
                        # Deliberately uncalibrated GPU clocks: rank1 precedes rank0 numerically.
                        rec['trace_rows'] = [dict(row=r, send=[9000000, 9000100, 9000300],
                                                  consume=[1, 101, 301],
                                                  feedback=[9000200, 9000900, 9001000]) for r in range(batch)]
                    records.append(('SEAM-STEP', rec))
    records.append(('SEAM-RESULT', dict(status='PASS', steps_checked=rounds*2*(iterations+warmup))))
    return records


def write_log(tmp_path, records, name='run.log'):
    path = tmp_path/name
    path.write_text('full unfiltered chatter\n' + '\n'.join(f'{k} {json.dumps(v)}' for k, v in records)+'\n')
    return path


@pytest.mark.parametrize('trace', [False, True])
def test_summary_complete_and_paired_attribution(tmp_path, trace):
    path = write_log(tmp_path, receipts(trace=trace))
    report = seam.summarize([path])
    assert report['status'] == 'COMPLETE'
    assert not report['cross_gpu_clock_subtraction']
    run = report['runs'][0]
    stats = run['round_medians_us']
    assert stats['payload']['wall_us'] == dict(median=101, spread=1)
    assert stats['payload_minus_control']['wall_us'] == dict(median=20, spread=0)
    if trace:
        assert stats['payload']['rank0_row_roundtrip_us']['median'] == 8
        assert stats['payload']['consumer_wait_acquire_us']['median'] == 1
        assert stats['payload']['consumer_body_drain_us']['median'] == 2
    else:
        assert 'rank0_row_roundtrip_us' not in stats['payload']


@pytest.mark.parametrize('mutation', [
    lambda rs: rs.pop(),
    lambda rs: rs.pop(1),
    lambda rs: rs.insert(2, deepcopy(rs[1])),
    lambda rs: rs.insert(1, deepcopy(rs[0])),
    lambda rs: rs[-1][1].update(status='FAIL'),
    lambda rs: rs[-1][1].update(steps_checked=1),
    lambda rs: rs[1][1].update(correct=False),
    lambda rs: rs[1][1].update(rows_checked=1),
    lambda rs: rs[1][1].update(elements_checked=16384),
    lambda rs: rs[1][1].update(arm='unknown'),
    lambda rs: rs[1][1].update(phase='unknown'),
    lambda rs: rs[1][1].update(iteration=999),
    lambda rs: rs[1][1].update(round=-1),
    lambda rs: rs[1][1].update(wall_us=float('nan')),
    lambda rs: rs[1][1].update(rank0_span_us=0),
    lambda rs: rs[0][1].update(batch=3),
    lambda rs: rs[0][1].update(label='full model latency'),
    lambda rs: rs[0][1].update(row_bytes=512),
    lambda rs: rs[0][1].update(rounds=True),
    lambda rs: rs.append(deepcopy(rs[-1])),
    lambda rs: rs.append(deepcopy(rs[1])),
])
def test_parser_rejects_incomplete_failed_or_ambiguous(tmp_path, mutation):
    records = receipts()
    mutation(records)
    with pytest.raises(ValueError):
        seam.parse_log(write_log(tmp_path, records))


@pytest.mark.parametrize('mutation', [
    lambda r: r.pop('trace_rows'),
    lambda r: r['trace_rows'].pop(),
    lambda r: r['trace_rows'][0].update(row=1),
    lambda r: r['trace_rows'][0].update(send=[2, 1, 3]),
    lambda r: r['trace_rows'][0].update(consume=[0, 10, 20]),
    lambda r: r['trace_rows'][0].update(feedback=[10, 20, 30]),
])
def test_parser_requires_valid_same_device_row_traces(tmp_path, mutation):
    records = receipts(trace=True)
    mutation(records[1][1])
    with pytest.raises(ValueError):
        seam.parse_log(write_log(tmp_path, records))


def test_no_instrumentation_mixing_or_duplicate_batch(tmp_path):
    untraced = write_log(tmp_path, receipts(), 'untraced.log')
    traced = write_log(tmp_path, receipts(trace=True), 'traced.log')
    assert len(seam.summarize([untraced, traced])['runs']) == 2
    with pytest.raises(ValueError, match='duplicate'):
        seam.summarize([untraced, untraced])
    records = receipts()
    records[1][1]['trace_rows'] = []
    with pytest.raises(ValueError):
        seam.parse_log(write_log(tmp_path, records))


@pytest.mark.parametrize('text', ['SEAM-STEP {bad json}', 'SEAM-FAIL payload mismatch', 'SEAM-CONFIG []'])
def test_malformed_failure_records(tmp_path, text):
    path = tmp_path/'bad.log'
    path.write_text(text)
    with pytest.raises(ValueError):
        seam.parse_log(path)








def test_cli_defaults(tmp_path, monkeypatch):
    # Internal probe: the parsed defaults handed to the runner (the dry run itself is in tests/e2e).
    captured = []
    monkeypatch.setattr(seam, 'run', captured.append)
    assert seam.main(['run', '--build-root', str(tmp_path/'build'), '--logdir', str(tmp_path/'default-logs')]) == 0
    args = captured[0]
    assert (args.batches, args.iterations, args.rounds, args.warmup) == ([1, 2, 4, 8], 1000, 5, 20)


def test_native_source_has_watchdogs_and_untimed_validation():
    host = (ROOT/'examples/glm53/seam_host.hip').read_text()
    assert host.index('host.wait_stream(5.0, start)') < host.index('const double wall_us')
    assert host.index('const double wall_us') < host.index('host.after_wait()')
    assert 'if (ms-heartbeat.load() > 10000)' in host
    assert 'stage=%s sequence=%d' in host
    assert 'step < iterations' in host and 'j < 16384' in host
    assert 'if (received[i] != expected)' in host and 'if (returned[r] != expected)' in host
    assert 'std::_Exit(0)' in host
    tiles = (ROOT/'examples/glm53/seam_tiles.hip').read_text()
    assert 'j += 256' in tiles and '17*epoch' in tiles
    assert 'input' not in tiles and 'hipMemcpy' not in tiles
    assert 'c->coord[0] * 256' not in tiles


def test_session_is_separate_two_gpu_opt_in():
    jobs, limit = load_session(ROOT/'examples/sessions/glm53_seam.py')
    assert limit == 2400 and len(jobs) == 2
    assert jobs[1].lane == 'cpu' and jobs[1].env['HIP_VISIBLE_DEVICES'] == '0,1'
    assert jobs[1].quiet_s == 15 and '--iterations 1000 --rounds 5 --warmup 20' in jobs[1].cmd
    assert len(jobs[0].writes) == len(jobs[1].reads) == 4
    assert 'boundary' not in jobs[0].cmd + jobs[1].cmd
