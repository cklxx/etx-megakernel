"""examples/dsv2lite/whatif.py: the base layout reproduces fleet's layer, and the parser reads overlap_bench's lines."""
import pytest

from etx.models.dsv2lite import whatif as w


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


def test_touch_stream_overlaps_the_wait_and_shares_bandwidth():
    """Cost.touch_bytes: the touched bytes stream from the moment the worker takes the task (before its event fires), so
    a task whose event fires late finishes earlier by up to that stream; when the event is already ready, the touch does
    not start earlier than the body would have and the step is no slower; the touched bytes still cost bandwidth."""
    import json
    from etx.codegen import emit_plan_json
    from etx.frontends import hip_link
    from etx.ir.types import Graph, Resource
    from etx.passes import compile_graph
    from etx.sim.xcd import Cost, Hw, simulate
    g = Graph("touch_sim")
    g.tensor("w", (8,), role="weight", bytes_per_elem=1)
    g.etensor("E0", (1,), wait_count=1)
    res = Resource(threads=1024)
    g.call_device("first", (1,), hip_link("t.hip", "etx_tile_first"), resource=res, args=["w"], out_edges={"E0": "i->(0)"}, duration_us=1.0)
    g.call_device("second", (8,), hip_link("t.hip", "etx_tile_second", touch="etx_touch_second"), resource=res, args=["w"],
                  in_edges={"E0": "i->(0)"}, duration_us=1.0)
    plan = json.loads(emit_plan_json(compile_graph(g, "gfx942", {}, {})))
    hw = Hw(bw_max=4e6, n_half=0.0, handoff_us={"DOMAIN": 0.4, "DEVICE": 1.8}, wake_us=0.0, rate_cv=0.0, lat_us=0.7)

    def costs(first_chain, touch):
        def f(name, d):
            if name == "first":
                return Cost(chain_us=first_chain)
            return Cost(chain_us=1.0, stream_bytes=128e3, inflight=64e3, touch_bytes=64e3 if touch else 0.0)
        return f
    late_plain = simulate(plan, costs(20.0, False), hw)[1]["makespan_us"]
    late_touch = simulate(plan, costs(20.0, True), hw)[1]["makespan_us"]
    # the event fires at 20 us: one trip (64 KB at the loaded latency, 8 x 64 KB / 4 TB/s = 0.128 us -> L = 0.7) is hidden
    assert late_plain - 0.9 < late_touch < late_plain - 0.5, (late_plain, late_touch)
    # the touch shares the bandwidth: its bytes are not free (the makespan with touch still carries one trip of the body)
    assert late_touch > 20.0 + 1.8 + 1.0 + 0.7 - 1e-6
    soon_plain = simulate(plan, costs(0.0, False), hw)[1]["makespan_us"]
    soon_touch = simulate(plan, costs(0.0, True), hw)[1]["makespan_us"]
    assert soon_touch <= soon_plain + 1e-6
