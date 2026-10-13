"""K1 (docs/DSV2-D1-D5.md): ETX_DSV2_KVA=shared computes kv_a once (1/8 per XCD) behind a global qkv event.
Default byte-identical; shared: the event, the edges, the symbols, the compiled scope; refusals; the registry arms."""
import hashlib
import os
from pathlib import Path

import pytest

from etx.codegen import emit_kernel, emit_plan_json
from etx.ir.types import Scope
from etx.passes import PassOptions, compile_graph
from examples.dsv2lite import arms
from examples.dsv2lite import graph as dsv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "09011c207b1e92febf29a2c2efc2d944ef9ad7ac8a0803633579c8444d821f8a"   # tests/test_dsv2_d5.py
DEFAULT_KERNEL = "751293fad5f5c98ec6dbf9eeca3c73319b1d4ded4b6ca73996a65757c0ffae5d"


@pytest.fixture(autouse=True)
def clean_graph_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("ETX_DSV2_") or key in ("ETX_PREFETCH_ISOLATE_POLLER", "ETX_COHERENCE"):
            monkeypatch.delenv(key)


def _plan(layers=27):
    return compile_graph(dsv.build(layers=layers), "gfx942", {}, {}, PassOptions(inline_tiles=True, relay="on"))


def test_default_and_explicit_replicated_are_byte_identical(monkeypatch):
    for value in (None, "replicated"):
        if value is not None:
            monkeypatch.setenv("ETX_DSV2_KVA", value)
        plan = _plan()
        assert hashlib.sha256(emit_plan_json(plan).encode()).hexdigest() == DEFAULT_PLAN
        source = emit_kernel(plan)
        assert hashlib.sha256(source.encode()).hexdigest() == DEFAULT_KERNEL
        assert "_kvs" not in source


def test_shared_makes_one_global_qkv_event_read_by_every_attention_task(monkeypatch):
    monkeypatch.setenv("ETX_DSV2_KVA", "shared")
    g = dsv.build(layers=3)
    grids = {x.name: x for x in g.grids}
    for L in (0, 1, 2):
        e = g.events[f"E_qkv_{L}"]
        assert e.shape == (1,) and e.wait_count == dsv.XCDS * dsv.W
        assert grids[f"qkv_{L}"].body.symbol.endswith("_kvs") and grids[f"attn_{L}"].body.symbol == "fleet_attention_kvs"
        assert grids[f"qkv_{L}"].out_edges[f"E_qkv_{L}"].text == "xw->(0)"
        assert grids[f"attn_{L}"].in_edges[f"E_qkv_{L}"].text == "hc->(0)"
        assert grids[f"qabs_{L}"].in_edges[f"E_qkv_{L}"].text == "hs->(0)"
        assert grids[f"qabs_{L}"].body.symbol == "fleet_q_absorb"      # q_absorb reads q only: same tile
    assert {x.body.symbol for x in g.grids if x.name.startswith("qkv_")} == {"fleet_qkv_fused_kvs", "fleet_qkv_fused_fold_kvs"}


def test_the_compiler_gives_the_shared_qkv_event_device_scope(monkeypatch):
    """The cross-XCD hand-off's safety (last-arriver write-back + sc1 loads) needs DEVICE scope, not DOMAIN."""
    plan = _plan(layers=3)
    assert plan.events["E_qkv_1"].scope == Scope.DOMAIN                     # today: XCD-local
    monkeypatch.setenv("ETX_DSV2_KVA", "shared")
    plan = _plan(layers=3)
    assert plan.events["E_qkv_1"].scope == Scope.DEVICE
    assert plan.events["E_qkv_1"].cross_domain_edges > 0


def test_the_kept_recipe_with_shared_kva_compiles(monkeypatch):
    for k, v in {**arms.BASE_ENV, **arms.ARMS["alld4r0l1kvs"].env}.items():
        monkeypatch.setenv(k, v)
    plan = _plan(layers=27)
    assert all(plan.events[f"E_qkv_{L}"].scope == Scope.DEVICE for L in range(27))


@pytest.mark.parametrize("env, match", [
    ({"ETX_DSV2_KVA": "half"}, "replicated | shared"),
    ({"ETX_DSV2_KVA": "shared", "ETX_DSV2_D5": "1"}, "without ETX_DSV2_D5"),
    ({"ETX_DSV2_KVA": "shared", "ETX_DSV2_REQUESTS": "2"}, "one request"),
])
def test_unsupported_combinations_are_refused(monkeypatch, env, match):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match=match):
        dsv.build(layers=2)


def test_the_k1_and_ablation_arms_differ_from_the_kept_arm_in_exactly_one_thing():
    kept = arms.ARMS["alld4r0l1"]
    env = lambda a: arms.build_env(arms.ARMS[a])
    diff = lambda a: {k: v for k, v in env(a).items() if env("alld4r0l1").get(k) != v}
    assert diff("alld4r0l1kvs") == {"ETX_DSV2_KVA": "shared", "ETX_CFLAGS": env("alld4r0l1")["ETX_CFLAGS"] + " -DETX_DSV2_KVA_SHARED=1"}
    assert set(arms.ARMS["alld4r0l1noQ"].levers) == set(kept.levers) - {"gemv_no_terminal_all"}
    assert set(arms.ARMS["alld4r0l1noA"].levers) == set(kept.levers) - {"argmax_tree"}
    for name in ("alld4r0l1noQ", "alld4r0l1noA"):
        assert set(diff(name)) == {"ETX_FLEET_PATCH"} and arms.ARMS[name].flags == kept.flags
    for name in ("alld4r0l1kvs", "alld4r0l1noQ", "alld4r0l1noA"):
        a = arms.ARMS[name]
        assert a.order == ("alld4r0l1", name) and a.references == ("alld4r0l1",) and a.rule.reference == "alld4r0l1"
    k1 = arms.ARMS["alld4r0l1kvs"].rule
    assert (k1.min_median_us, k1.spread_factor, k1.strict, k1.labels) == (30, 2, True, ("KEEP", "DROP"))


def test_the_shared_entry_points_exist_only_behind_their_define():
    """The default translation unit must not change (always-present entry points moved the kept ISA in 78 lines)."""
    shim = (ROOT / "examples/dsv2lite/fleet_shim.hip").read_text()
    guarded = shim.split("#if defined(ETX_DSV2_KVA_SHARED) && ETX_DSV2_KVA_SHARED", 1)
    assert len(guarded) == 2 and "_kvs(" not in guarded[0]
    assert all(s in guarded[1].split("#endif", 1)[0] for s in ("fleet_qkv_fused_kvs(", "fleet_qkv_fused_fold_kvs(", "fleet_attention_kvs("))
