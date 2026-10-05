"""CPU arithmetic and protocol accounting; no hardware correctness claims."""
from dataclasses import replace
import hashlib
import json

import pytest

from etx.tools.fresh_handoff import matrix
from etx.tools.fresh_sim import Constant, check_constants, constants, report, simulate_case


def case(**updates):
    return dict(dict(id='small', bytes=64, handoffs=3, topology='same',
                     load='idle', mode='cached', repeat=0), **updates)


def test_hand_built_arithmetic():
    p = constants()
    # Deliberately artificial arithmetic fixture, not hardware estimates.
    values = dict(bw_bytes_us=1024, mall_idle_us=2, l2_idle_us=3,
                  l1_idle_us=1, launch_us=5, inv_idle_us=7,
                  check_us_per_word=.25, extra_flow_us=.5, host_s_per_case=1)
    p.update({k: replace(p[k], value=v, status='assumed', note='Artificial test fixture')
              for k, v in values.items()})
    r = simulate_case(case(), p)
    # Store 2, local drain included, flag 2, poll 4, first 3,
    # bulk 1, one parallel check trip .25, ACK store 2, ACK poll 4, extra .5.
    assert r['predicted_p50_full_flow_us'] == 18.75
    assert r['reader_first_load_us'] == 3
    assert r['predicted_wall_s_per_launch'] == (5 + 7 + 3 * 18.75) / 1e6
    assert r['predicted_wall_s_per_case'] == 2 * r['predicted_wall_s_per_launch'] + 1
    assert r['bandwidth_floor_s_per_launch'] == 3 * 128 / 1024 / 1e6
    assert r['payload_allocation_bytes'] == 3 * 128


def test_frozen_matrix_used_unchanged():
    frozen = matrix()['cases']
    assert len(frozen) == 144
    assert [r['case'] for r in report()['cases']] == frozen
    # Locks sizes, counts, pairing order, repeats, modes and case IDs.
    assert hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() == '8cb6aca648d2d89745c0ff46264be5b3215adcfed7b9c4e9b2dc2ad65ce9c6a7'


def test_protocol_allocation_and_traffic():
    r = report()
    for size, rounded in ((64, .12), (8192, 7.63), (65536, 61.04)):
        rows = [x for x in r['cases'] if x['case']['bytes'] == size]
        for x in rows:
            negative = x['case']['mode'] in ('forced_old', 'primed_stale')
            if negative:
                assert x['slots'] == 1
                assert x['payload_allocation_bytes'] == max(128, size)
            else:
                assert x['slots'] == 1_000_000
                assert round(x['payload_allocation_bytes'] / 2**30, 2) == rounded
    # The timed matrix covers all 144 cases, not the positives: positives alone are
    # 5.31 TB written / 15.94 TB read, the controls add 1.77 TB / 5.31 TB timed, and
    # their priming pass doubles the controls' own traffic (35.42 TB matrix total).
    assert r['timed_matrix_written_bytes'] == 7_084_032_000_000
    assert r['timed_matrix_read_bytes'] == 21_252_096_000_000
    assert round(r['timed_matrix_written_bytes'] / 1e12, 1) == 7.1
    assert r['timed_matrix_read_bytes'] / 1e12 == pytest.approx(21.2, abs=.1)
    assert (r['timed_matrix_written_bytes'] + r['timed_matrix_read_bytes']) / 1e12 == pytest.approx(28.336128)
    assert r['positive_written_bytes'] == r['timed_matrix_written_bytes'] * .75
    assert r['positive_read_bytes'] == r['timed_matrix_read_bytes'] * .75
    assert r['matrix_written_bytes'] == r['timed_matrix_written_bytes'] * 1.25
    assert r['matrix_read_bytes'] == r['timed_matrix_read_bytes'] * 1.25
    assert r['timed_matrix_bandwidth_floor_s'] == pytest.approx(28.336128 / 4.3)


@pytest.mark.parametrize('topology', ['same', 'cross', 'fanout'])
@pytest.mark.parametrize('mode', ['cached', 'bypass'])
def test_slots_read_only_after_publish(topology, mode):
    r = simulate_case(case(topology=topology, mode=mode))
    events = {e['stage']: e for e in r['stages']}
    publish = events['flag_publish']
    assert events['writer_store']['end_us'] <= publish['start_us']
    fence = events['local_waitcnt' if topology == 'same' else 'writeback_wait']
    assert fence['end_us'] <= publish['start_us']
    for epoch in range(2):
        slots = set()
        for slot in range(r['case']['handoffs']):
            assert slot not in slots
            slots.add(slot)
            offset = slot * r['predicted_p50_full_flow_us']
            for read in ('first_payload_load', 'bulk_payload_load'):
                assert offset + events[read]['start_us'] >= offset + publish['end_us']
            assert events['ack_publish']['start_us'] >= events['per_word_check']['end_us']
            assert events['writer_ack_poll']['end_us'] <= r['predicted_p50_full_flow_us']


def test_controls_prime_outside_writer_timer_and_share_one_slot():
    positive = simulate_case(case())
    for mode in ('forced_old', 'primed_stale'):
        r = simulate_case(case(mode=mode))
        assert r['prime_us_per_handoff'] > 0
        assert r['slots'] == 1
        assert r['written_bytes_per_launch'] == 2 * positive['written_bytes_per_launch']
        assert r['read_bytes_per_launch'] == 2 * positive['read_bytes_per_launch']
        assert r['predicted_p50_full_flow_us'] == positive['predicted_p50_full_flow_us']
        assert r['predicted_wall_s_per_launch'] > positive['predicted_wall_s_per_launch']


def test_busy_fanout_and_bypass_paths():
    idle = simulate_case(case(bytes=65536))
    busy = simulate_case(case(bytes=65536, load='synthetic_busy'))
    fanout = simulate_case(case(bytes=65536, load='synthetic_busy', topology='fanout'))
    assert busy['helpers'] == 302
    assert fanout['helpers'] == 296
    assert busy['busy_working_set_bytes'] == 302 * 2**21
    assert busy['predicted_p50_full_flow_us'] > idle['predicted_p50_full_flow_us']
    assert fanout['read_bytes_per_launch'] == 7 * busy['read_bytes_per_launch']
    assert fanout['predicted_p50_full_flow_us'] > busy['predicted_p50_full_flow_us']
    for r in report()['cases']:
        if r['case']['mode'] == 'cached':
            paired = simulate_case({**r['case'], 'mode': 'bypass'})
            assert r['paired_bypass_minus_cached_us'] == pytest.approx(
                paired['predicted_p50_full_flow_us'] - r['predicted_p50_full_flow_us'])


def test_unsourced_constant_rejected():
    p = constants()
    p['fabricated_hardware_latency'] = Constant(.123, '', 'measured', 'No evidence')
    with pytest.raises(ValueError, match='unsourced named constant'):
        check_constants(p)


def test_report_provenance_and_budget():
    r = report()
    assert r['budget_s'] == 28800
    assert r['predicted_matrix_wall_s'] == sum(c['predicted_wall_s_per_case'] for c in r['cases'])
    assert r['conditional_fits_budget'] == (r['predicted_matrix_wall_s'] <= 28800)
    assert r['actual_budget_fit'] == 'unknown'
    assert 'Staleness is a hardware property' in r['evidence']
    unknowns = {u['name'] for u in r['unknowns']}
    assert {k for k, c in constants().items() if c.status == 'assumed'} <= unknowns
    check_constants(constants())
    json.dumps(r, allow_nan=False)
