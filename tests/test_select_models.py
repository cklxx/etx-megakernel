"""Item 3 of docs/core/PLAN-generic-decoder-2026-10-08.md section 7: the selection for Qwen3-8B, Qwen3-30B-A3B and
Mixtral-8x7B, from the model config only, on the real REGISTRY (docs/core/SEC7-interface-fix.md).

The tests do not name the phase keys of the registry. They find a phase by its shape. They check each selected
variant against the host.hip rules of today, written here a second time (HOST_RULES). Thus a shape predicate in the
registry that is wider than the host rule fails here, and does not stop the host at run time.

The tests skip until etx.select exists (the stub in tests/select_stub.py has an empty REGISTRY).
"""
import os
from pathlib import Path

import pytest

from tests.select_stub import load

S, STUB = load()
pytestmark = pytest.mark.skipif(STUB, reason="etx.select does not exist yet (items 1 and 2)")

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "examples/llm/configs"
MODELS = {"qwen3-8b": CFG / "qwen3-8b.json", "qwen3-30b-a3b": CFG / "qwen3-30b-a3b.json",
          "mixtral-8x7b": CFG / "mixtral-8x7b.json"}
GFX = "gfx942"


def _dims(name):
    from examples.llm.model import load_dims
    return load_dims(str(MODELS[name]))


def _gemv_ok(s, d, b):                       # host.hip:358 and :393
    return s["K"] % 256 == 0 and s["K"] <= 4096


# export -> the condition under which host.hip does not stop (s: phase shape, d: Dims, b: Budget).
# dsl_moe_*: the ETX_EMIT_DSL_BIGK bound (65536, host.hip:432 and :463). Without BIGK the bound is 4096.
HOST_RULES = {
    "dsl_gemv": _gemv_ok,
    "dsl_gemv_fold": lambda s, d, b: _gemv_ok(s, d, b) and d.E > 0,
    "dsl_attnprep": lambda s, d, b: d.HD == 128 and d.NH % d.NKV == 0 and d.G + 2 <= b.threads // 64 and d.qk_norm,
    "dsl_topk": lambda s, d, b: d.E <= 128,
    "dsl_moe_up": lambda s, d, b: d.MI % 32 == 0 and d.H % 256 == 0 and d.H <= 65536,
    "dsl_moe_down": lambda s, d, b: d.H % 64 == 0 and d.MI % 256 == 0 and d.MI <= 65536,
    "dsl_moe_up_fold": lambda s, d, b: d.MI % 256 == 0 and d.H % 256 == 0 and max(d.H, d.MI) <= 65536 and d.E <= 128,
    "dsl_moe_down_fold": lambda s, d, b: d.MI % 256 == 0 and d.H % 256 == 0 and max(d.H, d.MI) <= 65536 and d.E <= 128,
}


def _legal(v, shape, budget, machine=GFX):   # contract section 5, rule 1
    r = v.resources
    return (machine in v.backend and bool(v.shape(shape)) and r.vgpr <= budget.vgpr and r.lds <= budget.lds
            and r.threads in (0, budget.threads))


@pytest.fixture(scope="module", params=sorted(MODELS))
def case(request):
    name = request.param
    p = S.plan(MODELS[name], GFX, None)
    return name, _dims(name), S.phase_shapes(MODELS[name]), p


def test_plan_covers_every_phase(case):
    name, d, shapes, p = case
    assert set(p.phases) == set(shapes), name
    assert set(p.manifest()["phases"]) == set(shapes)
    assert p.budget in S.DEFAULT_BUDGETS


def test_every_selected_variant_is_legal_and_cheapest(case):
    name, d, shapes, p = case
    total = 0.0
    for ph, v in p.phases.items():
        s = shapes[ph]
        assert v.phase == ph and _legal(v, s, p.budget), (name, ph, v.export)
        legal = [w for w in S.REGISTRY if w.phase == ph and _legal(w, s, p.budget)]
        best = min(w.cost(s, p.budget) for w in legal)
        assert v.cost(s, p.budget) == best, (name, ph, v.export)
        if v.kind != "dsl":                   # rule 2: a tie goes to dsl
            assert all(w.kind != "dsl" for w in legal if w.cost(s, p.budget) == best), (name, ph)
        total += v.cost(s, p.budget)
    assert p.cost == pytest.approx(total)


def test_every_selected_dsl_variant_passes_the_host_rule(case):
    name, d, shapes, p = case
    for ph, v in p.phases.items():
        if v.kind == "dsl":
            rule = HOST_RULES.get(v.export)
            if rule is not None:
                assert rule(shapes[ph], d, p.budget), (name, ph, v.export)


def test_registry_predicates_are_not_wider_than_the_host_rules(case):
    """Every dsl variant that the registry calls legal for a phase of the model passes the host rule."""
    name, d, shapes, p = case
    for ph, s in shapes.items():
        for v in S.REGISTRY:
            if v.phase == ph and v.kind == "dsl" and v.export in HOST_RULES and _legal(v, s, p.budget):
                assert HOST_RULES[v.export](s, d, p.budget), (name, ph, v.export)


def test_dense_model_has_no_expert_phase_and_down_is_not_dsl_gemv():
    d = _dims("qwen3-8b")
    p = S.plan(MODELS["qwen3-8b"], GFX, None)
    assert not d.moe
    assert not any(v.export.startswith(("dsl_moe", "dsl_topk")) for v in p.phases.values())
    shapes = S.phase_shapes(MODELS["qwen3-8b"])
    down = [ph for ph, s in shapes.items() if s.get("K") == d.INTER == 12288 and s.get("M") == d.H]
    assert down, "no phase with the down-projection shape (M = H, K = INTER)"
    for ph in down:                           # K = 12288 > 4096: host.hip:358 stops on dsl_gemv
        assert p.phases[ph].export not in ("dsl_gemv", "dsl_gemv_fold"), ph


@pytest.mark.parametrize("name", ["qwen3-30b-a3b", "mixtral-8x7b"])
def test_moe_models_have_router_and_experts(name):
    d = _dims(name)
    shapes = S.phase_shapes(MODELS[name])
    assert d.moe and d.E == {"qwen3-30b-a3b": 128, "mixtral-8x7b": 8}[name]
    assert d.TOPK == {"qwen3-30b-a3b": 8, "mixtral-8x7b": 2}[name]
    assert any(s.get("M") == d.E and s.get("K") == d.H for s in shapes.values()), "no router GEMV phase (M = E)"
    S.plan(MODELS[name], GFX, None)           # does not raise


def test_mixtral_does_not_select_attnprep():
    """Mixtral has no q/k norm (host.hip:379), so dsl_attnprep is never legal for it."""
    d = _dims("mixtral-8x7b")
    p = S.plan(MODELS["mixtral-8x7b"], GFX, None)
    assert not d.qk_norm and d.MI == 14336
    assert all(v.export != "dsl_attnprep" for v in p.phases.values())


def test_config_path_and_dims_give_the_same_plan():
    for name, path in MODELS.items():
        assert S.plan(path, GFX, None).manifest() == S.plan(_dims(name), GFX, None).manifest(), name


def test_planner_does_not_read_the_environment(monkeypatch):
    """Rule 6: ETX_* switches do not change the selection."""
    before = {n: S.plan(p, GFX, None).manifest() for n, p in MODELS.items()}
    for k in [k for k in os.environ if k.startswith("ETX_")]:
        monkeypatch.delenv(k)
    for k in ("ETX_VL_DSL_GEMV", "ETX_VL_DSL_MOE", "ETX_VL_DSL_TOPK", "ETX_VL_DSL_ATTNPREP", "ETX_VL_DSL_FOLD"):
        monkeypatch.setenv(k, "0")
    assert {n: S.plan(p, GFX, None).manifest() for n, p in MODELS.items()} == before


def test_sm_90_selection_uses_only_sm_90_variants():
    for name, path in MODELS.items():
        try:
            p = S.plan(path, "sm_90", None)
        except S.NoLegalVariant:
            continue                          # the registry may have no sm_90 variant for a phase yet
        assert all("sm_90" in v.backend for v in p.phases.values()), name
