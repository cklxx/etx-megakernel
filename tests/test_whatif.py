"""examples/dsv2lite/whatif.py: the base layout reproduces fleet's layer, and the parser reads overlap_bench's lines."""
import pytest

from examples.dsv2lite import whatif as w


def test_base_and_bounds():
    assert w.base() == pytest.approx(129.3)
    assert w.latency_us() + w.expert_us() == pytest.approx(129.3)
    # a free overlap (f = 1) cannot beat the byte-limited expert stream by more than its moved share
    assert 100 < w.shared_layout(1.0, 1.4, 4.0) < w.base()
    assert w.interleave_period(1.0) == pytest.approx(2 * w.latency_us())
    assert 6 < w.routed_union_units(2) < 12


def test_parse_overlap_lines():
    text = """RESULT stream   17.3 MB  cold 3.90 TB/s  warm 5.10 TB/s  mall 4.60 TB/s  (median of 5; mall pass: 0 of 304 neighbours share an XCD)
RESULT fold place=solo                bg depth 0: fold median   1.80 us  p90   2.00 us  |   8 fg    0 bg workgroups, bg 0.00 TB/s
RESULT fold place=same                bg depth 0: fold median   3.00 us  p90   3.40 us  | 304 fg    0 bg workgroups, bg 0.00 TB/s
RESULT fold place=same                bg depth 4: fold median   3.60 us  p90   4.40 us  | 304 fg  304 bg workgroups, bg 2.50 TB/s
RESULT fold place=split fg 8 CUs/XCD  bg depth 8: fold median   2.40 us  p90   2.90 us  |  64 fg  240 bg workgroups, bg 3.10 TB/s"""
    p = w.parse_overlap(text)
    assert p["stream"][17.3]["mall"] == 4.6
    assert [r["place"] for r in p["fold"]] == ["solo", "same", "same", "split"]
    assert p["fold"][3]["fg"] == 8
    assert w.f_curve(p) == [(2.5, pytest.approx(1.2))]
