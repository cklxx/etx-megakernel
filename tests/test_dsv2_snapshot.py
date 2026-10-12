"""CPU gates for the snapshot harness (ETX_DSV2_SNAPSHOT): capture builds the native prefix, replay builds one phase."""
import pytest

from etx.passes import compile_graph, verify_plan
from examples.dsv2lite.graph import build


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
