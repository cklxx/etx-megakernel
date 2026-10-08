"""Items 4 and 5 of docs/PLAN-generic-decoder-2026-10-08.md section 7, against docs/SEC7-interface-fix.md.

Item 4: a shape outside every DSL variant selects the imported kernel, and the planner does not stop.
Item 5: a variant above the budget does not change the selection of the other phases.

WAITS FOR GROK (items 1 and 2): `S` is the real etx.select when it exists, else tests/select_stub.py. The variants
here are test variants that carry the host.hip rules of today (the dsl_gemv rule is host.hip:358). They go to
`plan()` through `variants=`, so the same tests check the real planner's rules when etx.select lands. The tests on
the real REGISTRY and the three model configs are item 3 (lucky).
"""
import json
from pathlib import Path

import pytest

from etx.codegen import manifest
from tests.select_stub import load

S, STUB = load()
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "examples/llm/configs"
GFX = "gfx942"
B1024 = S.Budget(1024, 128, 65536)
B512 = S.Budget(512, 256, 65536)


def _v(export, phase, kind, shape, vgpr=128, lds=0, cost=1.0, threads=0):
    return S.Variant(export=export, phase=phase, kind=kind, backend=(GFX, "sm_90") if kind == "dsl" else (GFX,),
                     dtype="bf16", shape=shape, resources=S.Resources(vgpr=vgpr, lds=lds, threads=threads),
                     cost=(lambda s, b, c=cost: c))


def _dsl_gemv(s):                       # host.hip:358
    return s["K"] % 256 == 0 and s["K"] <= 4096


def _variants():
    out = []
    for p in ("qkv", "o_proj", "gate_up", "down", "lm_head", "router"):
        out.append(_v("dsl_gemv", p, "dsl", _dsl_gemv, vgpr=128, lds=21120, cost=1.0))
        out.append(_v("wvsplitk_y2", p, "imported", lambda s: 0 < s["K"] <= 16384, vgpr=64, lds=32768, cost=2.0))
    out.append(_v("dsl_topk", "topk", "dsl", lambda s: s["E"] <= 128, vgpr=40, cost=1.0))        # host.hip:382
    out.append(_v("topk_softmax", "topk", "imported", lambda s: True, vgpr=32, cost=2.0))
    out.append(_v("dsl_moe_up", "moe_up", "dsl", lambda s: s["MI"] % 32 == 0 and s["H"] <= 4096 and s["H"] % 256 == 0,
                  vgpr=128, lds=21120, cost=1.0))                                                  # host.hip:466
    out.append(_v("fused_moe_up", "moe_up", "imported", lambda s: True, vgpr=128, lds=32768, cost=2.0))
    out.append(_v("dsl_moe_down", "moe_down", "dsl", lambda s: s["H"] % 64 == 0 and s["MI"] <= 4096 and s["MI"] % 256 == 0,
                  vgpr=128, lds=21120, cost=1.0))                                                  # host.hip:467
    out.append(_v("fused_moe_down", "moe_down", "imported", lambda s: True, vgpr=128, lds=32768, cost=2.0))
    return out


def _choice(p):
    return {phase: (v.export, v.kind) for phase, v in p.phases.items()}


# ---- item 4: fallback


def test_dense_shape_outside_every_dsl_variant_selects_the_imported_kernel():
    """Qwen3-8B: down has K = INTER = 12288 > 4096. host.hip stops the run today; the planner selects wvSplitK."""
    p = S.plan(CFG / "qwen3-8b.json", GFX, B1024, variants=_variants())
    c = _choice(p)
    assert c["down"] == ("wvsplitk_y2", "imported")
    assert all(c[ph] == ("dsl_gemv", "dsl") for ph in ("qkv", "o_proj", "gate_up", "lm_head")), c


def test_moe_shape_outside_every_dsl_variant_selects_the_imported_kernel():
    """Mixtral-8x7B: moe_down has K = MI = 14336 > 4096 (no chunked DSL variant in this list)."""
    c = _choice(S.plan(CFG / "mixtral-8x7b.json", GFX, B1024, variants=_variants()))
    assert c["moe_down"] == ("fused_moe_down", "imported")
    assert c["moe_up"] == ("dsl_moe_up", "dsl") and c["topk"] == ("dsl_topk", "dsl")


def test_topk_above_128_experts_selects_the_imported_kernel(tmp_path):
    """host.hip:382 stops dsl_topk at E > 128. A 256-expert copy of Qwen3-30B-A3B falls back for topk only."""
    cfg = json.loads((CFG / "qwen3-30b-a3b.json").read_text())
    cfg["num_experts"] = 256
    (tmp_path / "e256.json").write_text(json.dumps(cfg))
    base = _choice(S.plan(CFG / "qwen3-30b-a3b.json", GFX, B1024, variants=_variants()))
    wide = _choice(S.plan(tmp_path / "e256.json", GFX, B1024, variants=_variants()))
    assert base["topk"] == ("dsl_topk", "dsl") and wide["topk"] == ("topk_softmax", "imported")
    assert {k: v for k, v in wide.items() if k != "topk"} == {k: v for k, v in base.items() if k != "topk"}


def test_a_variant_of_another_backend_is_not_a_fallback():
    """An imported kernel that does not list the machine is not legal there (sm_90 has no imported kernels here)."""
    vs = [v for v in _variants() if v.phase != "down"] + [
        _v("dsl_gemv", "down", "dsl", _dsl_gemv), _v("wvsplitk_y2", "down", "imported", lambda s: True)]
    with pytest.raises(S.NoLegalVariant, match="down"):
        S.plan(CFG / "qwen3-8b.json", "sm_90", B1024, variants=vs)


def test_a_phase_without_any_legal_variant_stops_the_planner_with_its_name():
    vs = [v for v in _variants() if not (v.phase == "down" and v.kind == "imported")]
    with pytest.raises(S.NoLegalVariant, match="down"):
        S.plan(CFG / "qwen3-8b.json", GFX, B1024, variants=vs)


# ---- item 5: budget


@pytest.mark.parametrize("cfg", ["qwen3-8b.json", "qwen3-30b-a3b.json", "mixtral-8x7b.json"])
def test_a_variant_above_every_budget_changes_nothing(cfg):
    """A cheap variant that no level can hold (VGPR 512, then LDS 96 KB) does not change any phase or the level."""
    levels = (B1024, B512)
    base = S.plan(CFG / cfg, GFX, levels, variants=_variants())
    for res in ({"vgpr": 512}, {"lds": 96 * 1024}, {"threads": 256}):
        for phase in base.phases:
            extra = _v("dsl_cheap", phase, "dsl", lambda s: True, cost=0.01, **res)
            p = S.plan(CFG / cfg, GFX, levels, variants=_variants() + [extra])
            assert _choice(p) == _choice(base) and p.budget == base.budget, (phase, res)


def test_a_variant_above_one_level_does_not_change_the_other_phases_at_any_level():
    """o_proj gets a cheap variant that needs 200 VGPRs. At 1024 threads it is illegal: nothing changes. At 512
    threads it is legal: o_proj changes and no other phase changes (the constraint is a maximum, not a sum)."""
    cfg = CFG / "qwen3-30b-a3b.json"
    extra = _v("dsl_gemv_wide", "o_proj", "dsl", lambda s: True, vgpr=200, cost=0.1)
    for level in (B1024, B512):
        base = _choice(S.plan(cfg, GFX, level, variants=_variants()))
        with_extra = _choice(S.plan(cfg, GFX, level, variants=_variants() + [extra]))
        changed = {k for k in base if base[k] != with_extra[k]}
        assert changed == (set() if level is B1024 else {"o_proj"}), (level, changed)


def test_the_resource_constraint_is_a_maximum_not_a_sum():
    """Two phases each take 40 KB of LDS at a 64 KB level. Their sum is above the limit; each is legal."""
    cfg = CFG / "qwen3-30b-a3b.json"
    vs = _variants() + [_v("dsl_lds_a", "o_proj", "dsl", lambda s: True, lds=40960, cost=0.1),
                        _v("dsl_lds_b", "router", "dsl", lambda s: True, lds=40960, cost=0.1)]
    c = _choice(S.plan(cfg, GFX, B1024, variants=vs))
    assert c["o_proj"] == ("dsl_lds_a", "dsl") and c["router"] == ("dsl_lds_b", "dsl")


def test_the_planner_does_not_read_the_environment(monkeypatch):
    cfg = CFG / "qwen3-30b-a3b.json"
    base = _choice(S.plan(cfg, GFX, B1024, variants=_variants()))
    for k, v in {"ETX_VL_DSL_GEMV": "0", "ETX_VL_DSL_TOPK": "0", "ETX_VL_IMPORTED": "1", "ETX_VL_DSL_FOLD": "0"}.items():
        monkeypatch.setenv(k, v)
    assert _choice(S.plan(cfg, GFX, B1024, variants=_variants())) == base


# ---- the bridge to item 6


def test_plan_manifest_is_the_format_that_the_manifest_publishes():
    p = S.plan(CFG / "qwen3-8b.json", GFX, B1024, variants=_variants())
    sel = p.manifest()
    assert sel["budget"] == {"threads": 1024, "vgpr": 128, "lds": 65536}
    assert sel["phases"]["down"] == ("wvsplitk_y2", "imported")
    block = "\n".join(manifest.selection_block(sel))
    assert '"down"' in block and "#define ETX_SEL_N_PHASES 5" in block

