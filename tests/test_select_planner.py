"""Smoke tests of etx.select (section 7 items 1 and 2): the interface the item 3-6 tests build on."""
import os

import pytest

from etx.select import (BUDGET_LEVELS, Budget, NoLegalVariant, REGISTRY, Resources, Variant, phase_shapes,
                        plan, variants_for)
from etx.select.variants import GEMV_LDS

CFG = "examples/llm/configs/{}.json"
MOE_MODELS = ("qwen3-30b-a3b", "mixtral-8x7b", "qwen3-8b")   # qwen3-8b: model.py follows the fallback


def _keyed(rows):
    """fix's phase-key rule (docs/SEC7-interface-fix.md section 1): <role>@0 only when layer 0 differs."""
    by_role: dict[str, dict[int, str]] = {}
    for role, layer, export in rows:
        by_role.setdefault(role, {})[layer] = export
    out = {}
    for role, layers in by_role.items():
        rest = {e for l, e in layers.items() if l != 0}
        if 0 in layers and rest and layers[0] not in rest:
            out[f"{role}@0"] = layers[0]
            out[role] = rest.pop()
        else:
            out[role] = next(iter(layers.values()))
    return out


@pytest.mark.parametrize("name", MOE_MODELS)
def test_default_plan_equals_model_py_default_build(name):
    """The planner reproduces the exports that examples/vllm_llm/model.py lays out in its default build. model.py
    reads ETX_VL_* at import time and other tests import it, so it runs in a fresh interpreter with a clean env."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ETX_VL_", "ETX_LLM_"))}
    env["PYTHONPATH"] = str(root)
    code = ("import json, sys; from examples.llm.model import load_dims; from examples.vllm_llm import model as M; "
            "print(json.dumps([(i['role'], i['layer'], i['export']) for i in M.instances(load_dims(sys.argv[1]))]))")
    out = subprocess.run([sys.executable, "-c", code, CFG.format(name)], cwd=root, env=env, check=True,
                         capture_output=True, text=True).stdout
    want = _keyed(tuple(r) for r in json.loads(out))
    got = {k: v.export for k, v in plan(CFG.format(name)).phases.items()}
    assert got == want


def test_fallback_selects_imported_when_no_dsl_variant_fits():
    """Qwen3-8B down: K = 12288 > 4096 (host.hip:358), so dsl_gemv is not legal and wvSplitK runs it."""
    p = plan(CFG.format("qwen3-8b"))
    assert p.phases["down"].export == "wvsplitk_y2" and p.phases["down"].kind == "imported"
    assert p.reasons["down"].startswith("fallback")
    assert p.phases["gate_up"].kind == "dsl"


def test_variant_above_budget_changes_nothing():
    base = plan(CFG.format("qwen3-30b-a3b"))
    big = Variant("dsl_gemv_huge", "o_proj", "dsl", ("gfx942",), "bf16", lambda s: True,
                  Resources(vgpr=256, lds=0, threads=1024), lambda s, b: 0.0)
    p = plan(CFG.format("qwen3-30b-a3b"), variants=[*REGISTRY, big])
    assert p.manifest() == base.manifest()
    assert p.resources() == base.resources()


def test_budget_levels_and_threads():
    """At 512 threads the 1024-thread DSL tiles are not legal; wvSplitK (64 x 16) is not either, so qkv has nothing."""
    with pytest.raises(NoLegalVariant, match="qkv"):
        plan(CFG.format("qwen3-30b-a3b"), budget=Budget(512, 256, 65536))
    p = plan(CFG.format("qwen3-30b-a3b"), budget=BUDGET_LEVELS)
    assert p.budget == BUDGET_LEVELS[0]


def test_manifest_format_and_resources():
    p = plan(CFG.format("qwen3-30b-a3b"))
    m = p.manifest()
    assert m["budget"] == {"threads": 1024, "vgpr": 128, "lds": 65536}
    assert m["phases"]["qkv@0"] == ("dsl_gemv", "dsl") and m["phases"]["attn"] == ("attn_mfma16_g8", "imported")
    assert p.resources() == {"vgpr": 128, "lds": GEMV_LDS, "threads": 1024}
    assert "attn_mfma16_g8" in p.exports("imported") and "dsl_gemv" not in p.exports("imported")


def test_declarations_are_well_formed():
    from etx.ops import gemv
    assert GEMV_LDS == gemv.LDS_BYTES or gemv.DSL_NB != 1
    for v in REGISTRY:
        assert v.kind == ("dsl" if v.export.startswith("dsl_") else
                          "generic" if v.export.startswith("llm_") else "imported"), v
        assert v.backend == ("gfx942",)
    for name in ("qwen3-30b-a3b", "qwen3-8b", "mixtral-8x7b", "phi-3.5-moe", "qwen2.5-1.5b"):
        for ph in phase_shapes(CFG.format(name)):
            assert variants_for(ph), (name, ph)        # REGISTRY plans every phase: none is skipped
    assert not any(v.legal("sm_90", {}, BUDGET_LEVELS[0]) for v in variants_for("o_proj"))


def test_planner_ignores_the_environment(monkeypatch):
    a = plan(CFG.format("qwen3-30b-a3b")).manifest()
    for k in ("ETX_VL_DSL_FOLD", "ETX_VL_IMPORTED", "ETX_VL_FOLD_TOPK", "ETX_VL_DSL_GEMV"):
        monkeypatch.setenv(k, "0")
    assert plan(CFG.format("qwen3-30b-a3b")).manifest() == a


def test_plan_accepts_a_config_dict():
    import json
    from pathlib import Path
    cfg = json.loads(Path(CFG.format("mixtral-8x7b")).read_text())
    assert plan(cfg).manifest() == plan(CFG.format("mixtral-8x7b")).manifest()


@pytest.mark.parametrize("hd", [64, 96, 192])
def test_unsupported_head_size_has_a_generic_plan_but_no_vllm_host_abi(hd):
    from dataclasses import replace
    from examples.llm.model import load_dims
    from examples.vllm_llm import model as M
    d = replace(load_dims(CFG.format("qwen3-8b")), HD=hd, qk_norm=False)
    p = plan(d, "gfx942", None)
    assert (p.phases["attn"].export, p.phases["attn"].kind) == ("llm_attn", "generic")
    with pytest.raises(SystemExit, match="phase attn: only the generic tile llm_attn is legal; the vLLM host cannot launch it"):
        M.instances(d)


def test_generic_is_only_selected_without_a_legal_specialized_variant():
    from dataclasses import replace
    generic = next(v for v in REGISTRY if v.kind == "generic")
    cheap = replace(generic, cost=lambda s, b: 0.0)
    p = plan(CFG.format("qwen3-8b"), variants=[v for v in REGISTRY if v is not generic] + [cheap])
    assert p.phases["attn"].kind == "imported"


@pytest.mark.parametrize("hd,g,legal", [(128, 5, True), (192, 16, True), (130, 5, False),
                                       (200, 5, False), (128, 17, False)])
def test_generic_attention_shape_bounds(hd, g, legal):
    generic = next(v for v in REGISTRY if v.kind == "generic")
    assert generic.shape({"HD": hd, "G": g}) is legal


@pytest.mark.parametrize("g", range(1, 17))
def test_day0_attention_imports(g):
    from dataclasses import replace
    from examples.llm.model import load_dims
    d = load_dims(CFG.format("qwen3-8b"))
    selected = plan(replace(d, NH=d.NKV * g)).phases["attn"]
    assert selected.export == f"attn_mfma{4 if g <= 4 else 16}_g{g}"
    assert selected.kind == "imported"


@pytest.mark.parametrize("qk_norm", [False, True])
def test_g17_attention_has_no_legal_variant(qk_norm):
    from dataclasses import replace
    from examples.llm.model import load_dims
    d = load_dims(CFG.format("qwen3-8b"))
    d = replace(d, NH=d.NKV * 17, qk_norm=qk_norm)
    with pytest.raises(NoLegalVariant, match="phase 'attn'"):
        plan(d, "gfx942", None)


@pytest.mark.parametrize("hd", [128, 256])
def test_attention_head_size_without_qk_norm(hd):
    from dataclasses import replace
    from examples.llm.model import load_dims
    d = replace(load_dims(CFG.format("qwen3-8b")), HD=hd, qk_norm=False)
    if hd == 256:
        with pytest.raises(NoLegalVariant, match="phase 'attn'"):
            plan(d, "gfx942", None)
    else:
        selected = plan(d, "gfx942", None).phases["attn"]
        assert (selected.export, selected.kind) == ("attn_mfma4_g4", "imported")


@pytest.mark.parametrize("k,export", [(16384, "wvsplitk_y2"), (16392, "wvsplitk_l40_y2"),
                                     (20480, "wvsplitk_l40_y2"), (20488, None), (17412, None)])
def test_day0_down_lds_bounds(k, export):
    from dataclasses import replace
    from examples.llm.model import load_dims
    d = replace(load_dims(CFG.format("qwen3-8b")), INTER=k)
    if export is None:
        with pytest.raises(NoLegalVariant, match="down"):
            plan(d)
    else:
        assert plan(d).phases["down"].export == export


@pytest.mark.parametrize("export,k", [("wvsplitk_y2", 16384), ("wvsplitk_l40_y2", 16392)])
def test_y2_requires_even_rows(export, k):
    variant = next(v for v in variants_for("down") if v.export == export)
    assert variant.shape(dict(fold=0, K=k, M=4096))
    assert not variant.shape(dict(fold=0, K=k, M=4097))


@pytest.mark.parametrize("name,count,refusal", [
    ("mistral-7b-v0.3", 354, None), ("qwen3-14b", 402, None), ("qwen2.5-7b", 310, None),
    ("phi-3.5-mini", None, "phase attn: only the generic tile llm_attn"),
    ("olmo-2-13b", None, "olmo2 full-width q/k norm"),
    ("qwen1.5-moe-a2.7b", None, "shared experts"),
])
def test_unseen_day0_configs(name, count, refusal):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ETX_VL_", "ETX_LLM_", "ETX_DSL_"))}
    env["PYTHONPATH"] = str(root)
    code = ("import sys; from examples.llm.model import load_dims; "
            "from examples.vllm_llm import model as M; "
            "print(len(M.instances(load_dims(sys.argv[1]))))")
    result = subprocess.run([sys.executable, "-c", code,
                             str(root / "tests/fixtures/unseen" / name / "config.json")],
                            cwd=root, env=env, capture_output=True, text=True)
    if refusal:
        assert result.returncode != 0
        assert refusal in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert int(result.stdout.strip()) == count
