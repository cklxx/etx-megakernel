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
    assert set(meas) == set(fl.PHASES) and set(fl.PHASES) <= set(hops)
    assert set(hops) - set(fl.PHASES) == {'prologue', 'in_norm', 'argmax'}
    assert all(h.tag in fl.TAGS for h in hops.values())
    assert all(probes[p][0] is not None for p in fl.PHASES if hops[p].tag.startswith('probe'))
    assert {p for p in fl.PHASES if hops[p].tag == 'ISA hops x RTT'} == {'prep', 'attn', 'reduce', 'router', 'topk'}
    assert {p for p in fl.PHASES if hops[p].tag == 'probe + ISA hops x RTT'} == {'qkv', 'moe_up', 'moe_down'}
    assert hops['o_proj'].tag == 'probe' and probes['o_proj'][0] == 5.30
    assert all(h.where.startswith(fl.DOC) and fl.ISA_FILE in h.where for h in hops.values())


def test_isa_counts_are_exact_integers_with_the_documented_prices():
    hops = fl.hop_table()
    # qkv and router run the same inlined fold body, so they have the same chain
    q, r = hops['qkv'], hops['router']
    assert (q.coherent, q.plain, q.scalar, q.scratch) == (r.coherent, r.plain, r.scalar, r.scratch)
    # attn is the only phase with a scratch reload on its chain (spill 14 of the matching build)
    assert {p for p, h in hops.items() if h.scratch} == {'attn'}
    # the imported kernels start after an acquire: plain loads cost a full round trip
    assert {p for p, h in hops.items() if h.preset == 'acquired'} == {'attn', 'reduce', 'in_norm', 'argmax'}
    a = hops['attn']
    assert a.cost(0.50, 0.104) == pytest.approx(a.plain * 0.50 + (a.scalar + a.scratch + a.store_plain) * 0.104)


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
    base = fl.floors(4.3, 0.50)
    doubled = {k: replace(v, crit_body=2 * v.crit_body, mean_body=2 * v.mean_body) for k, v in fl.round11_table().items()}
    monkeypatch.setattr(fl, 'round11_table', lambda: doubled)
    r = fl.floors(4.3, 0.50)
    for key in ('bandwidth_floor_us', 'hop_chain_us', 'overlap_floor_us', 'serial_floor_us'):
        assert r[key] == base[key]


@pytest.mark.parametrize('bw', fl.BANDWIDTHS_TBS)
@pytest.mark.parametrize('rtt', RTTS + (0.28,))
def test_isa_floors_are_ordered_and_below_each_body(bw, rtt):
    r = fl.floors(bw, rtt)
    assert r['bandwidth_floor_us'] <= r['overlap_floor_us'] <= r['serial_floor_us'] < r['measured_span_us']
    assert r['floor_above_measured_body'] == []
    assert all(row['phase_floor_us'] <= row['crit_body_us'] for row in r['per_layer'].values())
    assert all(row['floor_us'] <= row['device_us'] for row in r['tail'].values())
    split = r['split_of_non_prefetch_gain_us']
    assert sum(split.values()) == pytest.approx(r['non_prefetch_gain_max_us'])
    assert 0 <= split['rest'] < 0.02 * r['measured_span_us']


def test_bandwidth_sets_the_overlap_floor():
    for rtt in RTTS:
        for bw in fl.BANDWIDTHS_TBS:
            r = fl.floors(bw, rtt)
            assert r['overlap_bound_by'] == 'bandwidth' and r['hop_chain_us'] < r['bandwidth_floor_us']


def test_no_phase_is_unknown_any_more():
    r = fl.floors()
    assert r['unknown_phases'] == [] and r['lower_bound']
    assert r['tail']['lm_head']['tag'] == 'bytes/BW'
    assert {t for t, row in r['tail'].items() if row['tag'] == 'ISA hops x RTT'} == {'prologue', 'in_norm', 'argmax'}


def test_bandwidth_floor_matches_the_ledger():
    assert fl.floors(4.3)['bandwidth_floor_us'] == pytest.approx(6.1904e3 / 4.3, abs=0.5)


def test_layer_parts_add_up_to_the_period():
    meas = fl.round11_table().values()
    hand = fl.handoff_us_per_layer()[0]
    assert sum(m.delta for m in meas) == pytest.approx(101.83, abs=0.05)
    assert sum(m.crit_body + m.start for m in meas) + hand == pytest.approx(101.83, abs=0.05)


def test_grid_covers_both_bandwidths_and_rtts_and_the_median_sensitivity():
    assert {(r['rtt_us'], r['bandwidth_tbs']) for r in fl.grid()} == {(t, b) for t in RTTS for b in fl.BANDWIDTHS_TBS}
    assert fl.wg_median_rtt_us()[0] == 0.28
    assert {(r['rtt_us'], r['bandwidth_tbs']) for r in fl.sensitivity()} == {(0.28, b) for b in fl.BANDWIDTHS_TBS}


def test_isa_hop_table_matches_the_matching_build():
    """Recount the ISA hops. Needs the matching .s and its inline map (see docs/FLOORS-2026-10-08.md)."""
    s, m = os.environ.get('ETX_ISA_S'), os.environ.get('ETX_ISA_MAP')
    if not (s and m and Path(s).is_file() and Path(m).is_file()):
        message = 'ISA recount needs ETX_ISA_S and ETX_ISA_MAP'
        if os.environ.get('ETX_REQUIRE_DATA') == '1' and os.environ.get('ETX_REQUIRE_ISA') == '1':
            pytest.fail(message)
        pytest.skip(message)
    from etx.tools import isa_phases
    only = ['fold', 'prep', 'topk']
    got = isa_phases.summary(isa_phases.phases(Path(s), isa_phases.load_map(Path(m)), only, after_line=83700))
    hops = fl.hop_table()
    for ph, name in (('fold', 'qkv'), ('prep', 'prep'), ('topk', 'topk')):
        h, g = hops[name], got[ph]['hops']
        assert (g['coherent'], g['plain'], g['scalar'], g['scratch']) == (h.coherent, h.plain, h.scalar, h.scratch), ph


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
