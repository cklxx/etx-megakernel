"""CPU arithmetic for the overlap and serial floors; no hardware claims."""
import pytest

from etx.tools import floors as fl


def test_rows_resolve_from_the_per_layer_table():
    ph = fl.budget_phases()
    assert set(ph) == set(fl.PHASES)
    assert ph['qkv'].crit_body == 11.69 and ph['topk'].achievable is None


def test_missing_evidence_fails_closed(monkeypatch):
    real = fl.source
    monkeypatch.setattr(fl, 'source', lambda path, text, after=None: real(path, text + ' NOT IN THE FILE', after))
    with pytest.raises(ValueError, match='missing source'):
        fl.floors()


@pytest.mark.parametrize('bw', fl.BANDWIDTHS_TBS)
@pytest.mark.parametrize('novmem', [False, True])
def test_floors_are_ordered_and_close(bw, novmem):
    r = fl.floors(bw, novmem)
    assert r['bandwidth_floor_us'] <= r['overlap_floor_us'] <= r['serial_floor_us'] < r['measured_span_us']
    assert r['layer_period_check_us']['budget_parts'] == pytest.approx(101.83, abs=0.1)
    assert abs(r['split_of_non_prefetch_gain_us']['rest']) < 0.01 * r['measured_span_us']


def test_bandwidth_floor_matches_the_ledger():
    assert fl.floors(4.3)['bandwidth_floor_us'] == pytest.approx(1439.0, abs=1.0)


def test_skeleton_chain_sets_the_overlap_floor():
    base, nov = fl.floors(4.3), fl.floors(4.3, use_novmem=True)
    assert base['overlap_floor_us'] == base['skeleton_chain_us'] > base['bandwidth_floor_us']
    assert nov['skeleton_chain_us'] > base['skeleton_chain_us']
