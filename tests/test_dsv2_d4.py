"""D4 (docs/deepseek/DSV2-D1-D5.md): shared experts during routing, routed slices on the shared XCDs, and the build-time placement
check. The negative tests prove the refusal fires: in the rule, in the realised graph, inside graph.build() and through
the real `etx compile` command line."""
from __future__ import annotations

import dataclasses
from decimal import Decimal
import json
import re
from pathlib import Path

import pytest

from etx.session.experiment import decide
from examples.dsv2lite.arms import ARMS, LEVERS_ALL
from examples.dsv2lite import d4_placement as d4

ROOT = Path(__file__).resolve().parents[1]


def decision(name, gains):
    arm = ARMS[name]
    rounds = [{arm.rule.reference: {"device_ms": "3.500"},
               name: {"device_ms": str(Decimal("3.500") - Decimal(g) / 1000)}} for g in gains]
    return decide(arm.rule, rounds, name)


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


# The `etx compile` refusal command is executed in tests/e2e/test_api_dsv2_d4.py.


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
    assert ARMS["d4"].flags == ("-DETX_DSV2_D4=1",) and ARMS["d4"].env == {"ETX_DSV2_D4": "1"}
    assert decision("d4", [40, 35, 42, 38, 41])["passed"]
    assert not decision("d4", [40, 35, -1, 38, 41])["passed"]          # one pair slower
    assert not decision("d4", [28, 29, 31, 27, "29.5"])["passed"]          # median below 30 us
    from etx.models.dsv2lite import ledger as ld
    T = ld.T
    fake = [T(0, 5, "router", 10.0, 19.0, None, None), T(0, 5, "router", 10.0, 20.0, None, None),
            T(0, 5, "shgu", 10.5, 40.0, None, None), T(0, 6, "router", 90.0, 99.0, None, None),
            T(0, 6, "shgu", 100.0, 120.0, None, None)]
    monkeypatch.setattr(ld, "load_etx", lambda plan, trace, pro, merge_d4=True: fake if not merge_d4 else [])
    r = d4.trace_lead("plan.json", "trace.bin")
    assert r["layers"] == 2 and r["before_routing"] == 1 and r["leads_us"] == {"5": 9.5, "6": -1.0}


def test_d4s_session_builds_the_three_arms_before_timing(monkeypatch):
    from etx.experiments.jobs import load_session
    monkeypatch.delenv("ETX_SESS_MAX_SECONDS", raising=False)
    jobs, limit = load_session(ROOT / "examples/sessions/ds_d4s.py")
    by = {j.name: j for j in jobs}
    assert limit == 1800
    assert set(by) == {"etx_env", "source", "preflight", "build_default", "build_alld4r0", "build_alld4s",
                       "pairs_alld4s"}
    assert by["source"].reads == ["etx_env"] and by["preflight"].writes == ["inputs.json"]   # inputs: already on the box
    for arm in ("default", "alld4r0", "alld4s"):
        assert by[f"build_{arm}"].reads == ["source.json"]
        assert by[f"build_{arm}"].cmd.endswith(f"inplace_session build {arm}")
    assert [j.name for j in jobs if j.lane == "gpu"] == ["pairs_alld4s"]
    assert by["pairs_alld4s"].gpu == 0
    assert by["pairs_alld4s"].reads == ["inputs.json", "b_default/run", "b_alld4r0/run", "b_alld4s/run"]
    assert by["pairs_alld4s"].cmd.endswith("inplace_session pairs alld4s")
    assert ARMS["alld4s"].levers == ARMS["alld4r0"].levers == LEVERS_ALL
    assert ARMS["alld4s"].flags == ARMS["alld4r0"].flags == ("-DETX_DSV2_D4=1",)
    assert ARMS["alld4s"].env == {"ETX_DSV2_D4": "1", "ETX_DSV2_D4_ROWS": "32"}
    assert ARMS["alld4r0"].env == {"ETX_DSV2_D4": "1", "ETX_DSV2_D4_ROWS": "0"}


def _slice_session(tmp_path, monkeypatch):
    from etx.models.dsv2lite.session import Session
    for name in ("ETX_SESS_LOGDIR", "ETX_DSV2_FLEET", "ETX_DSV2_DATA"):
        monkeypatch.setenv(name, str(tmp_path / name))
    return Session()


def _full_slice_log(device_ms):
    device_ms = Decimal(device_ms)
    return ("run 0 (warm-up): 32/32 tokens match\n"
            "consecutive layers within the gate: 27 of 27\n"
            f"per-token latency (embed -> argmax on device): median {device_ms:.6f} ms\n"
            f"step wall (launch to end) {device_ms + Decimal('0.1'):.6f} ms/token\n"
            "tokens: 32/32 match HF greedy\n")


@pytest.mark.parametrize("gains,keep", [
    ([40, 35, 42, 38, 41], True),
    ([10, 9, 11, 8, 10], False),             # faster than default by >100 us, but <30 us against alld4r0
    ([40, 35, -1, 38, 41], False),           # every pair must improve
    ([40, 20, 30, 35, 40], False),           # median must exceed twice the full-range spread
])
def test_d4s_pairs_decides_against_alld4r0_and_reverses_order(tmp_path, monkeypatch, capsys, gains, keep):
    s = _slice_session(tmp_path, monkeypatch)
    seen = []

    def run(name, command, **kwargs):
        _, round_name, arm = name.split("_", 2)
        seen.append(arm)
        r = int(round_name[1:])
        device = {"default": "3.650", "alld4r0": "3.500", "alld4s": Decimal("3.500") - Decimal(gains[r]) / 1000}[arm]
        return _full_slice_log(device)

    monkeypatch.setattr(s, "run", run)
    s.pairs("alld4s")
    assert seen == ["default", "alld4r0", "alld4s", "alld4s", "alld4r0", "default"] * 2 + \
                   ["default", "alld4r0", "alld4s"]
    report = json.loads((s.logs / "pairs_alld4s.json").read_text())
    assert len(report["rounds"]) == 5
    assert all(v["tokens"] == 32 and v["layers"] == 27 for r in report["rounds"] for v in r.values())
    for basis in ("device_ms", "wall_ms"):
        assert set(report[basis]) == {"default", "alld4r0"}
        assert report[basis]["alld4r0"]["gains_us"] == pytest.approx(gains)
    assert report["decision"]["passed"] is keep
    assert report["device_ms"]["default"]["median_us"] > 100
    assert f"VERDICT arm=alld4s {'KEEP' if keep else 'NO-GO'}" in capsys.readouterr().out


@pytest.mark.parametrize("arm", ["default", "alld4r0", "alld4s"])
@pytest.mark.parametrize("old,new", [("27 of 27", "26 of 27"), ("tokens: 32/32", "tokens: 31/32"),
                                    ("32/32 tokens match", "31/32 tokens match")])
def test_d4s_pairs_refuses_any_arm_with_a_failed_gate(tmp_path, monkeypatch, arm, old, new):
    s = _slice_session(tmp_path, monkeypatch)

    def run(name, command, **kwargs):
        log = _full_slice_log("3.5")
        return log.replace(old, new) if name.endswith(f"_{arm}") else log

    monkeypatch.setattr(s, "run", run)
    with pytest.raises(ValueError):
        s.pairs("alld4s")
    assert not (s.logs / "pairs_alld4s.json").exists()


@pytest.mark.parametrize("gains,keep", [
    (["30.0"] * 5, True),                    # inclusive minimum gain
    (["29.999"] * 5, False),
    (["30.0", "30.0", "30.0", "30.0", "45.0"], False),  # exclusive twice-spread bound: median = 2 x spread
    (["30.0", "30.0", "30.0", "30.0", "44.999"], True),
])
def test_d4s_numeric_keep_boundaries(gains, keep):
    assert decision("alld4s", gains)["passed"] is keep


@pytest.mark.parametrize("fail_at", [1, 8, 15])
@pytest.mark.parametrize("failure", [RuntimeError("deadline; NOT COMPLETE"), KeyboardInterrupt()])
def test_d4s_pairs_does_not_publish_an_interrupted_run(tmp_path, monkeypatch, capsys, fail_at, failure):
    s = _slice_session(tmp_path, monkeypatch)
    calls = 0

    def run(name, command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise failure
        return _full_slice_log("3.5")

    monkeypatch.setattr(s, "run", run)
    with pytest.raises(type(failure)):
        s.pairs("alld4s")
    assert calls == fail_at
    assert not (s.logs / "pairs_alld4s.json").exists()
    assert "VERDICT" not in capsys.readouterr().out


def test_d4s_pairs_rejects_a_truncated_last_invocation(tmp_path, monkeypatch, capsys):
    s = _slice_session(tmp_path, monkeypatch)
    calls = 0

    def run(name, command, **kwargs):
        nonlocal calls
        calls += 1
        log = _full_slice_log("3.5")
        return log.rsplit("tokens:", 1)[0] if calls == 15 else log

    monkeypatch.setattr(s, "run", run)
    with pytest.raises(ValueError, match="32/32 free-greedy gate"):
        s.pairs("alld4s")
    assert calls == 15
    assert not (s.logs / "pairs_alld4s.json").exists()
    assert "VERDICT" not in capsys.readouterr().out


def test_d4s_pairs_refuses_reusing_a_completed_report(tmp_path, monkeypatch, capsys):
    s = _slice_session(tmp_path, monkeypatch)
    prior = s.logs / "pairs_alld4s.json"
    original = '{"arm": "alld4s", "keep_vs_alld4r0": {"keep": true}}\n'
    prior.write_text(original)

    def run(*args, **kwargs):
        pytest.fail("an existing result must be refused before launching a new invocation")

    monkeypatch.setattr(s, "run", run)
    with pytest.raises(RuntimeError, match="fresh ETX_SESS_LOGDIR"):
        s.pairs("alld4s")
    assert prior.read_text() == original
    assert "VERDICT" not in capsys.readouterr().out


def test_d4s_compiled_queues_preserve_routing_in_worker_lds(monkeypatch):
    """Caching ids/ws from shgu into sdown is valid only if nothing runs between them on that workgroup."""
    from etx.passes import PassOptions, compile_graph
    from examples.dsv2lite import graph
    monkeypatch.setenv("ETX_DSV2_D4", "1")
    monkeypatch.setenv("ETX_DSV2_D4_ROWS", "32")
    monkeypatch.setenv("ETX_DSV2_COH", "0")
    g = graph.build(layers=3, requests=1, routing_cached=True)
    plan = compile_graph(g, "gfx942", {}, {}, PassOptions(relay="on"))
    tasks = {t.id: t for t in plan.tasks}
    assert all(t.mode == "static" for t in tasks.values())
    count = 0
    for queue in plan.static_queues.values():
        for i, task_id in enumerate(queue):
            gu = tasks[task_id]
            if gu.grid.startswith("shgu_"):
                down = tasks[queue[i + 1]]
                assert down.grid == gu.grid.replace("shgu_", "sdown_")
                assert down.coord == gu.coord
                assert down.worker == gu.worker and down.domain == gu.domain
                assert gu.domain == 6 + gu.coord[0]
                count += 1
    shared_layers = sum(grid.name.startswith("shgu_") for grid in g.grids)
    assert count == shared_layers * 2 * 37 and shared_layers > 0


@pytest.mark.parametrize("changes", [{"requests": 2}, {"routing_cached": False}])
def test_d4s_refuses_to_run_without_single_request_cached_routing(monkeypatch, changes):
    from examples.dsv2lite import graph
    monkeypatch.setenv("ETX_DSV2_D4", "1")
    options = dict(layers=3, requests=1, routing_cached=True)
    options.update(changes)
    with pytest.raises(ValueError, match="one request with cached routing only"):
        graph.build(**options)


def test_d4s_wave_and_subgroup_ownership_cover_every_row_once():
    """Addressing proof: flattened slices stay on their shared XCD; four-lane groups read only their worker's part."""
    from collections import Counter
    for x in range(2):
        gu = Counter()
        dn = Counter()
        for worker in range(37):
            for wave in range(4):
                for n in range(worker + wave * 37, 3 * 32, 4 * 37):
                    gu[(x * 3 + n // 32, n % 32)] += 1
            for subgroup in range(256 // 4):
                for n in range(worker + subgroup * 37, 2048, (256 // 4) * 37):
                    assert n % 37 == worker  # shared GEMV's part[n] lives in this worker's LDS
                    dn[n] += 1
        assert gu == Counter({(slot, row): 1 for slot in range(x * 3, x * 3 + 3) for row in range(32)})
        assert dn == Counter({row: 1 for row in range(2048)})


def test_d4s_short_reduction_matches_fleet_including_cancellation_and_signed_zero():
    """Numerical proof of the changed reduction, independent of GPU scheduling or timing.

    Inputs are the four fp32 fma8 lane accumulators (the common fma8 body is unchanged). Compare the actual old
    half-wave butterfly, including its 28 inactive lanes, against 16 four-lane groups packed into a wave. Exercise
    three independent slice sums, the shared-first addition order, and the final BF16 epilogue bit for bit.
    """
    import numpy as np
    rng = np.random.default_rng(405)
    v = rng.normal(size=(4096, 3, 4)).astype(np.float32)
    v *= np.exp2(rng.integers(-100, 100, size=v.shape)).astype(np.float32)
    v[0] = np.float32(-0.0)
    v[1] = np.float32([0.0, -0.0, 0.0, -0.0])
    v[2] = np.float32([2**50, -2**50, 1, -1])
    v[3] = np.float32([1, -1, -1, 1])
    old = np.zeros((4096, 3, 32), np.float32)
    old[..., :4] = v
    for off in (16, 8, 4, 2, 1):
        old = old + old[..., np.arange(32) ^ off]
    # Adjacent groups hold unrelated rows: using offsets 4/8/16 here would cross rows and fail.
    new = (v + np.float32(0)).transpose(1, 0, 2).reshape(3, -1, 64)
    for off in (2, 1):
        new = new + new[..., np.arange(64) ^ off]
    got = new.reshape(3, 4096, 4).transpose(1, 0, 2)[..., 0]
    want = old[..., 0]
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))

    def bf16(a):
        u = np.asarray(a, np.float32).view(np.uint32)
        return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)

    shared = rng.normal(size=4096).astype(np.float32)
    ref, opt = shared.copy(), shared.copy()
    for j in range(3):
        ref = ref + want[:, j]
        opt = opt + got[:, j]
    # Fleet's last expert_down_chunk has scale=1 and two BF16 roundings; the new store removes the redundant one.
    assert np.array_equal(bf16(np.float32(1) * bf16(ref)).view(np.uint32), bf16(opt).view(np.uint32))
    # This fixture distinguishes the forbidden concatenated/reassociated reduction from shared-first accumulation.
    assert np.float32(np.float32(np.float32(2**25) + np.float32(-(2**25))) + np.float32(1)) != \
           np.float32(np.float32(2**25) + np.float32(np.float32(-(2**25)) + np.float32(1)))
