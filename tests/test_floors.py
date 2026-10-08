"""CPU arithmetic for the hardware floors of docs/FLOORS-2026-10-08.md; no hardware claims."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess

import pytest

from etx.tools import floors as fl

ROOT = Path(__file__).resolve().parents[1]
RTTS = (0.50, 0.69)


def test_tables_resolve_with_one_tag_per_phase():
    meas, hops, probes = fl.round11_table(), fl.hop_table(), fl.probes_us()
    assert set(meas) == set(hops) == set(fl.PHASES)
    assert all(h.tag in fl.TAGS for h in hops.values())
    assert all(probes[p][0] is not None for p, h in hops.items() if h.probe_used)
    assert {p for p, h in hops.items() if h.tag == 'hops x RTT'} == {'prep', 'attn', 'reduce', 'router', 'topk'}
    assert hops['o_proj'].tag == 'probe' and probes['o_proj'][0] == 5.30
    assert all(h.where.startswith(fl.DOC) for h in hops.values())


def test_prices_come_from_the_cited_lines():
    (lo, hi), where = fl.rtt_range_us()
    assert (lo, hi) == RTTS and where == 'docs/TILES-2026-10-07.md:32'
    assert fl.cached_hop_us()[0] == pytest.approx(0.1043)
    assert fl.handoff_us_per_layer()[0] == 9.02
    assert fl.measured_span_us()[0] == 5081.5


def test_missing_evidence_fails_closed(monkeypatch):
    real = fl.source
    monkeypatch.setattr(fl, 'source', lambda path, text, after=None: real(path, text + ' NOT IN THE FILE', after))
    with pytest.raises(ValueError, match='missing source'):
        fl.floors()


def test_no_floor_comes_from_a_measured_body(monkeypatch):
    base = fl.floors(4.3, 0.50, 'min')
    doubled = {k: replace(v, crit_body=2 * v.crit_body, mean_body=2 * v.mean_body) for k, v in fl.round11_table().items()}
    monkeypatch.setattr(fl, 'round11_table', lambda: doubled)
    r = fl.floors(4.3, 0.50, 'min')
    for key in ('bandwidth_floor_us', 'hop_chain_us', 'overlap_floor_us', 'serial_floor_us'):
        assert r[key] == base[key]


@pytest.mark.parametrize('bw', fl.BANDWIDTHS_TBS)
@pytest.mark.parametrize('rtt', RTTS)
def test_min_hop_floors_are_ordered_and_below_each_body(bw, rtt):
    r = fl.floors(bw, rtt, 'min')
    assert r['bandwidth_floor_us'] <= r['overlap_floor_us'] <= r['serial_floor_us'] < r['measured_span_us']
    assert r['floor_above_measured_body'] == []
    assert all(row['phase_floor_us'] <= row['crit_body_us'] for row in r['per_layer'].values())
    split = r['split_of_non_prefetch_gain_us']
    assert sum(split.values()) == pytest.approx(r['non_prefetch_gain_max_us'])
    assert 0 <= split['rest'] < 0.02 * r['measured_span_us']


def test_bandwidth_sets_the_overlap_floor_with_minimum_hops():
    for rtt in RTTS:
        r = fl.floors(4.3, rtt, 'min')
        assert r['overlap_bound_by'] == 'bandwidth' and r['hop_chain_us'] < r['bandwidth_floor_us']


def test_maximum_hops_put_qkv_above_its_body():
    for rtt in RTTS:
        r = fl.floors(4.3, rtt, 'max')
        assert r['floor_above_measured_body'] == ['qkv']
        assert r['overlap_bound_by'] == 'hop chain'


def test_unknown_phases_are_listed_and_the_floor_is_a_lower_bound():
    r = fl.floors()
    assert r['unknown_phases'] == ['argmax', 'in_norm', 'prologue'] and r['lower_bound']


def test_bandwidth_floor_matches_the_ledger():
    assert fl.floors(4.3)['bandwidth_floor_us'] == pytest.approx(6.1904e3 / 4.3, abs=0.5)


def test_layer_parts_add_up_to_the_period():
    meas = fl.round11_table().values()
    hand = fl.handoff_us_per_layer()[0]
    assert sum(m.delta for m in meas) == pytest.approx(101.83, abs=0.05)
    assert sum(m.crit_body + m.start for m in meas) + hand == pytest.approx(101.83, abs=0.05)


def test_grid_covers_both_bandwidths_rtts_and_hop_bounds():
    keys = {(r['hops'], r['rtt_us'], r['bandwidth_tbs']) for r in fl.grid()}
    assert keys == {(h, t, b) for h in fl.HOPS for t in RTTS for b in fl.BANDWIDTHS_TBS}


def _git_show(rev: str, path: str) -> list[str] | None:
    p = subprocess.run(['git', 'show', f'{rev}:{path}'], cwd=ROOT, capture_output=True, text=True)
    return p.stdout.splitlines() if p.returncode == 0 else None


def test_tile_evidence_lines_at_the_round11_commit():
    files = {}
    for path, line, text in fl.TILE_EVIDENCE:
        if path not in files:
            files[path] = _git_show(fl.TILE_REV, path)
        if files[path] is None:
            pytest.skip(f'commit {fl.TILE_REV} is not in this clone')
        assert text in files[path][line - 1], f'{fl.TILE_REV}:{path}:{line}'


def test_vllm_evidence_lines():
    src = os.environ.get('VLLM_SRC')
    if not src:
        pytest.skip('set VLLM_SRC to a vLLM v0.27.1 checkout')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=src, capture_output=True, text=True).stdout.strip()
    assert head == fl.VLLM_REV
    for path, line, text in fl.VLLM_EVIDENCE:
        assert text in (Path(src) / path).read_text().splitlines()[line - 1], f'{path}:{line}'


def test_round11_table_matches_the_raw_trace():
    base = Path(os.environ.get('ETX_DATA_ROOT', str(Path.home() / 'code/etx'))) / 'results_2026-10-01/round11'
    plan, raw = base / 'plan_r8base.json', base / 'trace_r11_decode.bin'
    if not (plan.is_file() and raw.is_file()):
        message = 'Round11 floors check needs ETX_DATA_ROOT with the archived pair'
        if os.environ.get('ETX_REQUIRE_DATA') == '1':
            pytest.fail(message)
        pytest.skip(message)
    got = fl.round11_from_trace(plan, raw)
    for name, m in fl.round11_table().items():
        g = got[name]
        assert g['tasks'] == m.tasks
        for key in ('crit_body', 'mean_body', 'delta', 'start'):
            assert g[key] == pytest.approx(getattr(m, key), abs=0.006), (name, key)
