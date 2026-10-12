"""Local-only protocol gates: truncated/mismatching runs cannot authorize an arm."""
from array import array
import json
from pathlib import Path

import pytest

from etx.codegen import emit_plan_json
from etx.passes import compile_graph, verify_plan
from etx.tools.gpu_jobs import load_session
from examples.dsv2lite.graph import build
from examples.dsv2lite.inplace_session import full_gain, full_result, paired_screen, phase_span


@pytest.mark.parametrize("mode,phase,last", [
    ("capture", "qkv_5", "down_4"), ("capture", "router_5", "oproj_5"),
    ("replay", "qkv_5", "iso0x0_r0_qkv_5"), ("replay", "router_5", "iso0x0_r0_router_5"),
])
def test_native_capture_and_single_phase_replay(monkeypatch, mode, phase, last):
    monkeypatch.setenv("ETX_DSV2_SNAPSHOT", f"{mode}:0:{phase}")
    g = build(); plan = compile_graph(g, "gfx942", {}, {})
    assert verify_plan(plan) == []
    assert g.grids[-1].name == last
    if mode == "replay":
        assert len(plan.tasks) == 296 and len(g.grids) == 1
        assert not g.grids[0].in_edges
    else:
        assert phase not in [x.name for x in g.grids]
        assert "argmax" not in [x.name for x in g.grids]


@pytest.mark.parametrize("snapshot,extra", [
    ("replay:0:qkv_6", {}), ("replay:1:qkv_5", {}),
    ("replay:0:qkv_5", {"ETX_DSV2_ISOLATE": "0:qkv_5"}),
    ("replay:0:qkv_5", {"ETX_DSV2_REQUESTS": "2"}),
])
def test_snapshot_rejects_unsupported_or_mutating_graph(monkeypatch, snapshot, extra):
    monkeypatch.setenv("ETX_DSV2_SNAPSHOT", snapshot)
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        build()


def test_screen_uses_paired_weighted_net_not_independent_medians():
    result = paired_screen([1, 2, 3, 4, 5], [4, 3, 2, 1, 0])
    assert result["gains_us"] == [131, 132, 133, 134, 135]
    assert result["go"] and result["hundred_us_route"]
    assert not paired_screen([1] * 5)["go"]
    assert not paired_screen([2, 2, 2, 2, -1])["go"]
    assert not paired_screen([2, 2, 2, 2, 4])["go"]
    for values in ([2] * 4, [float("nan")] * 5):
        with pytest.raises(ValueError):
            paired_screen(values)


def good_log(fleet=False):
    warm = "7.000 ms total, " if fleet else ""
    wall = ("wall time of the launches: 96.000 ms for 32 tokens = 3.000 ms/token (333 tok/s), 1 launch(es)"
            if fleet else "step wall (launch to end) 3.000 ms/token")
    return (f"run 0 (warm-up): {warm}32/32 tokens match\n"
            "  consecutive layers within the gate: 27 of 27\n"
            "per-token latency (embed -> argmax on device): median 2.900 ms mean 2.9 ms; " + wall + "\n"
            "tokens: 32/32 match HF greedy\n")


@pytest.mark.parametrize("fleet", [False, True])
def test_both_real_full_log_formats_require_all_gates(fleet):
    assert full_result(good_log(fleet))["wall_ms"] == 3.0
    for old, new in [("27 of 27", "26 of 27"), ("tokens: 32/32", "tokens: 31/32"),
                     ("32/32 tokens match", "31/32 tokens match"), ("median 2.900", "median 0.000")]:
        with pytest.raises(ValueError):
            full_result(good_log(fleet).replace(old, new))


def test_full_checkpoint_is_separate_from_screen():
    assert full_gain([.075] * 5)["pass_70us"]
    assert not full_gain([.065] * 5)["pass_70us"]
    assert not full_gain([.1, .1, .1, .1, -.001])["pass_70us"]


def test_phase_trace_rejects_partial_samples_but_accepts_zero_relative_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("ETX_DSV2_SNAPSHOT", "replay:0:qkv_5")
    plan = compile_graph(build(), "gfx942", {}, {})
    p = tmp_path / "plan.json"; p.write_text(emit_plan_json(plan))
    raw = tmp_path / "trace.bin"
    rows = []
    for task in plan.tasks:
        coords = list(task.coord) + [0] * (4 - len(task.coord))
        rows += [0, *coords, 100, 100, 2100, task.worker]
    array("q", rows).tofile(raw.open("wb"))
    assert phase_span(p, raw, "qkv_5") == 20
    array("q", rows[:-9]).tofile(raw.open("wb"))
    with pytest.raises((ValueError, SystemExit)):
        phase_span(p, raw, "qkv_5")


def test_prepared_session_pins_gpu_and_gates_every_candidate():
    jobs, limit = load_session(Path("examples/sessions/ds_inplace.py"))
    assert limit == 7200
    by_name = {job.name: job for job in jobs}
    assert all(j.gpu == 0 and j.quiet_s <= 90 for j in jobs if j.lane == "gpu")
    for arm in ("Q", "gamma", "tree", "both"):
        job = by_name[f"full_{arm}"]
        assert job.gate == (f"screen_{arm}", rf"^SCREEN arm={arm} GO=1(?: |$)")
        assert job.after == ["screen_both"]
    assert set(by_name["trace"].after_any) == {f"full_{x}" for x in ("Q", "gamma", "tree", "both")}


def test_inputs_are_prepared_on_the_box_before_the_first_timed_run():
    """31 GB cannot be staged from the laptop (docs/DATA.md:59-64): prep jobs make the inputs, preflight re-checks
    them, builds only wait for the source check, and the first GPU timing waits for both."""
    jobs, _ = load_session(Path("examples/sessions/ds_inplace.py"))
    by_name = {job.name: job for job in jobs}

    def upstream(name, seen=None):
        seen = set() if seen is None else seen
        for dep in by_name[name].after:
            if dep not in seen:
                seen.add(dep)
                upstream(dep, seen)
        return seen

    assert {"prep_ref", "prep_pack", "prep_sums", "source"} <= upstream("preflight")
    assert by_name["prep_ref"].lane == "gpu" and "source" in by_name["prep_ref"].after
    for arm in ("default", "fleet", "stamps", "Q", "gamma", "tree", "both"):
        assert by_name[f"build_{arm}"].after == ["source"]
    assert {"preflight", "build_default", "build_fleet"} <= set(by_name["baseline"].after)
    # The runner forwards only ETX_SESS_MAX_SECONDS: every job sets the path defaults itself.
    assert all("ETX_DSV2_DATA:=$HOME/dsv2-data" in j.cmd and "ETX_DSV2_FLEET:=$HOME/fleet-3fb67a6" in j.cmd
               for j in jobs)
    # Inputs come from the pinned export, never from a fleet checkout.
    assert all("fleet-mi300x" not in j.cmd for j in jobs)


@pytest.mark.parametrize("identity,ok", [
    ("commit " + "a" * 40 + "\nbranch b\ndirty 0\n", True),
    ("commit " + "a" * 40 + "\nbranch b\ndirty 1\n", False),
    ("branch b\ndirty 0\n", False),
    (None, False),
])
def test_source_identity_must_name_a_clean_commit(tmp_path, monkeypatch, identity, ok):
    from examples.dsv2lite.inplace_session import Session
    monkeypatch.setenv("ETX_SESS_LOGDIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ETX_DSV2_DATA", str(tmp_path / "data"))
    session = Session()
    session.root = tmp_path
    if identity is not None:
        (tmp_path / "SOURCE_IDENTITY").write_text(identity)
    if ok:
        assert session.source_identity().startswith("commit ")
    else:
        with pytest.raises(RuntimeError, match="SOURCE_IDENTITY"):
            session.source_identity()
