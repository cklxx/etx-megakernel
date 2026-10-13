"""D4 (docs/DSV2-D1-D5.md): shared experts during routing, routed slices on the shared XCDs, and the build-time placement
check. The negative tests prove the refusal fires: in the rule, in the realised graph, inside graph.build() and through
the real `etx compile` command line."""
from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from examples.dsv2lite import d4_placement as d4

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def d4_graph(monkeypatch):
    monkeypatch.setenv("ETX_DSV2_D4", "1")
    from examples.dsv2lite import graph
    return graph.build()


def _move(stages, unit, at_lo, stage, **changes):
    out, hit = [], 0
    for s in stages:
        if s.unit == unit and s.lo == at_lo and s.stage == stage:
            s = dataclasses.replace(s, **changes)
            hit += 1
        out.append(s)
    assert hit == 1
    return out


def test_the_layout_obeys_every_rule():
    for rows in (0, 32, 64, 128):
        d4.check(d4.layout(rows))
    st = d4.layout(32)
    # every slice's two stages on one XCD; the shared XCDs run their halves without routing
    for s in st:
        twin = [t for t in st if (t.unit, t.lo, t.hi) == (s.unit, s.lo, s.hi) and t.stage != s.stage]
        assert len(twin) == 1 and twin[0].xcd == s.xcd
    assert {s.xcd for s in st if s.unit.startswith("shared")} == {6, 7}
    assert {(s.unit, s.xcd) for s in st if s.lo == 0 and s.unit.startswith("slot")} == \
        {(f"slot{k}", 6 + k // 3) for k in range(6)}


@pytest.mark.parametrize("name,mutate,needle", [
    ("split slice", lambda st: _move(st, "slot1", 0, "down", xcd=7, coord=1), "split slice: slot1 rows [0, 32) gate_up on XCD 6 but down on XCD 7"),
    ("split routed body", lambda st: _move(st, "slot4", 32, "gate_up", xcd=6), "split slice: slot4 rows [32, 1408)"),
    ("rows uncovered", lambda st: _move(st, "slot2", 32, "down", lo=64), "rows [32, 1408): has ['gate_up'] only"),
    ("rows twice", lambda st: st + [dataclasses.replace(s, lo=32, hi=64) for s in st if s.unit == "slot0" and s.lo == 0],
     "slot0 gate_up: rows [32, 64) covered twice"),
    ("misaligned", lambda st: d4.layout(40), "row boundary 40 is not a multiple of 32"),
    ("routing on the shared start", lambda st: _move(st, "shared1", 0, "gate_up", routing=True),
     "shared XCD 7 must start with shared half 1's gate_up and no routing"),
    ("missing down", lambda st: [s for s in st if not (s.unit == "shared0" and s.stage == "down")],
     "shared0 rows [0, 1408): has ['gate_up'] only"),
])
def test_check_refuses_an_invalid_placement(name, mutate, needle):
    with pytest.raises(d4.PlacementError) as e:
        d4.check(mutate(d4.layout(32)))
    assert needle in str(e.value), (name, str(e.value))


def test_the_realised_graph_places_every_stage_where_the_table_says(d4_graph):
    g = d4_graph
    L = 5
    d4.verify_graph(g, L, d4.layout(32), {f"E_norm_{L}"})
    assert set(g.grid(f"shgu_{L}").in_edges) == {f"E_oproj_{L}"}          # starts when o_proj completes, no routing
    assert g.grid(f"router_{L}").grid == (6, 37) and g.grid(f"shgu_{L}").grid == (2, 37)
    assert g.grid(f"sdown_{L}").consts == (L, 0, 32, 3)


def test_verify_graph_refuses_a_grid_that_splits_a_slice(d4_graph):
    """A deliberately invalid realisation: the shared down moved one XCD over, so the slices' h would be read across
    XCDs. The table is fine; the graph is not -- and that is what the build checks."""
    g = d4_graph
    g.grid("sdown_5").domain_map = "xw->(x+5)"
    with pytest.raises(d4.PlacementError, match=r"sdown_5 coordinate 0 .* runs on XCD \[5\], the placement says XCD 6"):
        d4.verify_graph(g, 5, d4.layout(32), {"E_norm_5"})
    g2 = d4_graph
    g2.grid("sdown_5").domain_map = "xw->(x+6)"
    g2.grid("shgu_7").in_edges = {**g2.grid("shgu_7").in_edges, "E_norm_7": "xw->(0)"}
    with pytest.raises(d4.PlacementError, match="shgu_7 waits on the routing"):
        d4.verify_graph(g2, 7, d4.layout(32), {"E_norm_7"})


def test_graph_build_refuses_an_invalid_placement(monkeypatch):
    """The refusal fires inside the build itself: graph.build() (what `etx compile` runs) with a split-slice table."""
    monkeypatch.setenv("ETX_DSV2_D4", "1")
    good = d4.layout
    monkeypatch.setattr(d4, "layout", lambda rows=d4.ROWS: _move(good(rows), "slot3", 0, "down", xcd=6, coord=0))
    from examples.dsv2lite import graph
    with pytest.raises(d4.PlacementError, match="split slice: slot3 rows \\[0, 32\\) gate_up on XCD 7 but down on XCD 6"):
        graph.build()


@pytest.mark.parametrize("rows,needle", [("40", "row boundary 40 is not a multiple of 32"),
                                         ("1408", "rows [1408, 1408): outside [0, 1408) or empty")])
def test_etx_compile_refuses_an_invalid_placement(tmp_path, rows, needle):
    env = dict(os.environ, ETX_DSV2_D4="1", ETX_DSV2_D4_ROWS=rows)
    p = subprocess.run([sys.executable, "-m", "etx", "compile", "examples/dsv2lite/graph.py", "--arch", "gfx942",
                        "--out", str(tmp_path / "plan"), "--inline-tiles", "--relay", "on"],
                       cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert p.returncode != 0 and "PlacementError" in p.stdout and needle in p.stdout, p.stdout[-2000:]
    assert not (tmp_path / "plan" / "megakernel_d0.hip").exists()


def test_d4_refuses_the_coherent_plan(monkeypatch):
    monkeypatch.setenv("ETX_DSV2_D4", "1")
    monkeypatch.setenv("ETX_DSV2_COH", "1")
    from examples.dsv2lite import graph
    with pytest.raises(ValueError, match="ETX_DSV2_D4 with ETX_DSV2_COH is not supported"):
        graph.build()


def test_d4_is_off_by_default(monkeypatch):
    monkeypatch.delenv("ETX_DSV2_D4", raising=False)
    from examples.dsv2lite import graph
    names = {g.name for g in graph.build().grids}
    assert not any(n.startswith(("shgu_", "sdown_")) for n in names) and "gate_up_5" in names


def test_the_tiles_exist_only_in_the_d4_build(monkeypatch):
    """The d4 plan calls four bodies the shim defines only under ETX_DSV2_D4: a d4 plan cannot link without the flag."""
    shim = (ROOT / "examples/dsv2lite/fleet_shim.hip").read_text()
    block = shim[shim.index("#if defined(ETX_DSV2_D4) && ETX_DSV2_D4"):shim.index("#endif\n\n// exported for the host")]
    syms = set(re.findall(r'extern "C" __device__ void (fleet_d4_\w+)\(', shim))
    assert syms == {"fleet_d4_routed_gate_up", "fleet_d4_routed_down", "fleet_d4_shared_gate_up", "fleet_d4_shared_down"}
    assert all(f"void {s}(" in block for s in syms)
    monkeypatch.setenv("ETX_DSV2_D4", "1")
    from examples.dsv2lite import graph
    used = {g.body.symbol for g in graph.build().grids if g.body.symbol.startswith("fleet_d4_")}
    assert used == syms


def test_d4_verdict_and_trace_criterion(monkeypatch):
    from examples.dsv2lite.inplace_session import ARM_ENV, ARM_FLAGS, d4_verdict, paired
    assert ARM_FLAGS["d4"] == ["-DETX_DSV2_D4=1"] and ARM_ENV["d4"] == {"ETX_DSV2_D4": "1"}
    assert d4_verdict(paired([40.0, 35.0, 42.0, 38.0, 41.0]))["keep"]
    assert not d4_verdict(paired([40.0, 35.0, -1.0, 38.0, 41.0]))["keep"]          # one pair slower
    assert not d4_verdict(paired([28.0, 29.0, 31.0, 27.0, 29.5]))["keep"]          # median below 30 us
    from examples.dsv2lite import ledger as ld
    T = ld.T
    fake = [T(0, 5, "router", 10.0, 19.0, None, None), T(0, 5, "router", 10.0, 20.0, None, None),
            T(0, 5, "shgu", 10.5, 40.0, None, None), T(0, 6, "router", 90.0, 99.0, None, None),
            T(0, 6, "shgu", 100.0, 120.0, None, None)]
    monkeypatch.setattr(ld, "load_etx", lambda plan, trace, pro: fake)
    r = d4.trace_lead("plan.json", "trace.bin")
    assert r["layers"] == 2 and r["before_routing"] == 1 and r["leads_us"] == {"5": 9.5, "6": -1.0}
