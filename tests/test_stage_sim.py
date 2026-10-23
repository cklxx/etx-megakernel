"""CPU arithmetic for the o_proj LDS staging estimate; no hardware claims."""
from dataclasses import replace
import json

import pytest

from etx.tools.costs import stage_sim
from etx.tools.stage_sim import Constant, check_constants, constants, report, simulate


def with_values(**values):
    p = constants()
    p.update({k: replace(p[k], value=float(v), status='assumed', note='Artificial test fixture') for k, v in values.items()})
    return p


def test_every_constant_is_sourced_and_resolves():
    named = constants()
    check_constants(named)
    assert {c.status for c in named.values()} <= {'measured', 'derived', 'assumed'}
    for name in ('oproj_bytes', 'oproj_crit_body_us', 'lds_used_bytes', 'empty_task_us', 'wg_rate_kb_us'):
        assert named[name].status == 'measured'


def test_missing_evidence_fails_closed(monkeypatch):
    real = stage_sim.source
    monkeypatch.setattr(stage_sim, 'source', lambda path, text: real(path, text + ' NOT IN THE FILE'))
    with pytest.raises(ValueError, match='missing source'):
        constants()


def test_unsourced_constant_rejected():
    named = constants()
    named['bad'] = Constant(1.0, 'README.md:999999', 'measured', 'x')
    with pytest.raises(ValueError):
        check_constants(named)


def test_slice_and_staged_bytes():
    r = simulate()
    assert r['tasks_per_layer'] == 256
    assert r['slice_bytes'] == 65536                     # 8 rows x 4096 x bf16
    assert r['free_lds_bytes'] == 65536 - 29736
    assert r['staged_bytes'] == 32768                    # 8 whole 4 KB workgroup DMA steps, 50%
    assert r['dma_instructions_per_lane'] == 8


def test_hand_built_arithmetic():
    # Artificial fixture: 1 layer, 1 task of 8000 B, 4000 B free, 1000 B/us DMA, body 10 us.
    p = with_values(layers=1, oproj_tasks=1, oproj_bytes=8000, oproj_crit_body_us=10, empty_task_us=2,
                    lds_total_bytes=5000, lds_used_bytes=1000, threads=1000, lds_dma_bytes_per_lane=1,
                    wg_rate_kb_us=1, lds_bytes_per_clk=1000, clock_mhz=1,
                    prep_crit_body_us=10, attn_crit_body_us=0, reduce_crit_body_us=0, keep_ms=0)
    r = simulate(p)
    assert r['staged_bytes'] == 4000 and r['staged_share'] == .5
    assert r['stream_us_per_layer'] == 8
    assert r['hidden_us_scenario_a'] == 4                # half of the streaming part
    assert r['hidden_us_scenario_b'] == 4                # 4000 B at 1000 B/us, capped at 8
    assert r['added_cost_us'] == 4                       # 4000 B at 1000 B/us of LDS reads
    assert r['net_us_per_layer'] == (0, 0)
    assert r['stage_time_us'] == 4 and r['fits_window']


def test_vgpr_path_is_infeasible_at_128():
    r = simulate(path='vgpr')
    assert r['vgpr_limit'] == 128 and r['extra_vgpr_per_lane'] == 8
    assert not r['feasible'] and r['verdict'] == 'INFEASIBLE' and r['gain_ms_per_token'] == (0, 0)


def test_no_free_lds_is_infeasible():
    r = simulate(with_values(lds_used_bytes=65536))
    assert r['staged_bytes'] == 0 and r['verdict'] == 'INFEASIBLE'


def test_window_too_short_is_infeasible():
    r = simulate(with_values(prep_crit_body_us=0, attn_crit_body_us=0, reduce_crit_body_us=0))
    assert not r['fits_window'] and not r['feasible']


def test_current_prediction_and_sweep_are_monotone():
    r = simulate()
    lo, hi = r['gain_ms_per_token']
    assert 0 < lo <= hi < r['crit_body_us_per_layer'] * 48 / 1000
    assert r['verdict'] == 'ABOVE_THRESHOLD'
    gains = [row['gain_ms_per_token'][1] for row in report()['sweep']]
    assert gains == sorted(gains, reverse=True)


def test_report_serializes_and_lists_unknowns():
    out = report()
    json.dumps(out)
    names = {u['name'] for u in out['unknowns']}
    assert {'current_group_segment', 'lds_dma_rate', 'body_split', 'kv_interference', 'touch_overlap'} <= names
    assert out['evidence'].startswith('CPU simulation only')
