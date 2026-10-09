"""The analysis in examples/dsv2lite/routing_stats.py must be right before it measures anything.

CPU-only and fast (~2 s): it runs the module's --selftest, which pins the sample-size behaviour
of the pair-concentration metric against uniform routing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.dsv2lite.routing_stats import (pairs_of, uniform_control, cross_layer,
                                             pair_concentration, __name__ as _m)


def test_pairs_of_a_top_6_token():
    assert len(pairs_of([5, 1, 9, 2, 3, 4])) == 15
    assert pairs_of([3, 1]) == [(1, 3)]


def test_sample_size_controls():
    """Uniform routing at decode scale scores like structure; at Zyphra's scale it does not."""
    small = pair_concentration(uniform_control(32, 26, 6, 64, seed=0), 6, 64)
    big = pair_concentration(uniform_control(50000, 2, 6, 64, seed=0), 6, 64)
    assert small["hottest_pair_ratio"] > 5.0, "a 32-token sample must inflate the hottest pair"
    assert 0.90 <= big["ratio_vs_uniform"] <= 1.25, "50k uniform tokens must score about 1x"
    assert big["hottest_pair_ratio"] < 1.5


def test_cross_layer_needs_support():
    """At 32 tokens/layer every expert is noise-concentrated; at 50k none is."""
    small = cross_layer(uniform_control(32, 26, 6, 64, seed=0), 6, 64)
    big = cross_layer(uniform_control(50000, 2, 6, 64, seed=0), 6, 64)
    assert small["experts_over_30pct"] > 0
    assert big["experts_over_30pct"] == 0


def test_planted_pair_is_detected():
    planted = uniform_control(50000, 2, 6, 64, seed=0).reshape(-1, 6).copy()
    planted[::10, 0], planted[::10, 1] = 0, 1
    pc = pair_concentration(planted.reshape(50000, 2, 6), 6, 64)
    assert pc["hottest_pair_ratio"] > 8.0


def test_selftest_passes():
    from examples.dsv2lite.routing_stats import selftest
    assert selftest() == 0
