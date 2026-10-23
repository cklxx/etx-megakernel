"""CPU plan parity with the pre-migration decoder entry points.

Synthetic import descriptors exercise planning only: these tests make no claim
about imported kernel bytes or model output correctness on a GPU.
"""
from dataclasses import replace
import importlib
import json
from pathlib import Path

import pytest

from etx.codegen import emit_plan_json
from etx.compiler.frontend.decoder import DecoderConfig, build
from etx.passes import compile_graph, verify_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    import os
    for key in list(os.environ):
        if key.startswith(("ETX_", "DSL_")):
            monkeypatch.delenv(key)
    yield
    # Other tests use module-level options; do not leave our last variant cached.
    from examples.llm import model as lm
    from examples.vllm_llm import model as vm
    with monkeypatch.context() as reset:
        for key in list(os.environ):
            if key.startswith(("ETX_", "DSL_")):
                reset.delenv(key)
        importlib.reload(lm)
        importlib.reload(vm)


def plan_json(graph):
    plan = compile_graph(graph, "gfx942", {}, {})
    assert verify_plan(plan) == []
    return emit_plan_json(plan)


def no_legacy_build(*args, **kwargs):
    pytest.fail("frontend delegated graph construction to the legacy builder")


def test_qwen3_8b_native_full_plan(monkeypatch):
    monkeypatch.setenv("ETX_LLM_CONFIG", str(ROOT / "examples/llm/configs/qwen3-8b.json"))
    from examples.llm import model as lm
    importlib.reload(lm)
    config = DecoderConfig.from_example("llm")
    assert config.dims.L == 36
    expected = plan_json(lm.build())
    monkeypatch.setattr(lm, "build", no_legacy_build)
    assert plan_json(build(config)) == expected


def imported_config(tmp_path, monkeypatch, overrides):
    from test_vllm_moe import IMPORTS
    from etx.ops import attnprep, gemv, topk
    descriptors = {e: {"slots": s, "arg_bytes": a} for e, (s, a) in IMPORTS.items()}
    for mod in (attnprep, gemv, topk):
        descriptors.update(mod.import_entries())
    # Default head-affine attention uses one block per task, reduce slots = G.
    descriptors["attn_mfma16_g8"]["slots"] = 1
    if overrides.get("ETX_VL_DSL_ATTN") == "0":
        descriptors["attn_mfma16_g8"]["slots"] = 4
    imp = tmp_path / "imports.json"
    imp.write_text(json.dumps(descriptors))
    for key, value in {
        "ETX_LLM_CONFIG": str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"),
        "ETX_VL_IMPORTS": str(imp),
        "ETX_VL_PLAN": str(tmp_path / "vl_plan.txt"),
        "ETX_VL_CHAINS": str(tmp_path / "chain_tiles.hip"),
        **overrides,
    }.items():
        monkeypatch.setenv(key, value)
    from examples.llm import model as lm
    from examples.vllm_llm import model as vm
    importlib.reload(lm)
    importlib.reload(vm)
    return vm, DecoderConfig.from_example("vllm")


@pytest.mark.parametrize("overrides", [
    {},  # Acceptance case: the full 48-layer Qwen3-30B-A3B, batch one.
    {"ETX_LLM_LAYERS": "2", "ETX_VL_IMPORTED": "1", "ETX_VL_DSL_ATTN": "0", "ETX_VL_XCD": "0"},
    {"ETX_LLM_LAYERS": "2", "ETX_VL_IMPORTED": "1", "ETX_VL_DSL_ATTN": "0", "ETX_VL_XCD": "1", "ETX_VL_CHAIN": "1"},
    {"ETX_LLM_LAYERS": "2", "ETX_VL_REQUESTS": "2", "ETX_VL_BATCH": "1", "ETX_VL_NB": "2"},
])
def test_qwen3_moe_imported_plan(tmp_path, monkeypatch, overrides):
    vm, config = imported_config(tmp_path, monkeypatch, overrides)
    if not overrides:
        assert config.dims.L == 48 and config.settings["REQUESTS"] == 1
    expected = plan_json(vm.build())
    plan_path = tmp_path / "vl_plan.txt"
    expected_rows = plan_path.read_bytes()
    expected_validation = Path(str(plan_path) + ".imports-check.json").read_bytes()
    chain_path = tmp_path / "chain_tiles.hip"
    expected_chain = chain_path.read_bytes() if chain_path.exists() else None
    monkeypatch.setattr(vm, "build", no_legacy_build)
    assert plan_json(build(config)) == expected
    assert plan_path.read_bytes() == expected_rows
    assert Path(str(plan_path) + ".imports-check.json").read_bytes() == expected_validation
    if expected_chain is not None:
        assert chain_path.read_bytes() == expected_chain
    # A resolved config survives subsequent environment changes and batching
    # annotations without silently changing the graph.
    monkeypatch.setenv("ETX_VL_UC", "1")
    monkeypatch.setenv("ETX_CFLAGS", "-DETX_DSL_LOCAL_FOLD")
    monkeypatch.setenv("ETX_VL_PRODUCER_SCORE", "1")
    assert plan_json(build(config)) == expected


def test_explicit_native_config_needs_no_example_builder(monkeypatch):
    from examples.llm import model as lm
    config = DecoderConfig.from_example()
    config = replace(config, dims=replace(config.dims, L=1), context=64, chunk=32)
    monkeypatch.setattr(lm, "build", no_legacy_build)
    graph = build(config)
    assert graph.name == "llm_qwen3-8b"
    assert graph.grid("attn_0").grid == (config.dims.NKV, 2)
    assert verify_plan(compile_graph(graph, "gfx942", {}, {})) == []


@pytest.mark.parametrize("backend", ["unknown", ""])
def test_unknown_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="unsupported decoder backend"):
        DecoderConfig.from_example(backend)
    with pytest.raises(ValueError, match="unsupported decoder backend"):
        build(DecoderConfig(backend, None))


def test_invalid_native_context_is_rejected():
    config = DecoderConfig.from_example()
    with pytest.raises(ValueError, match="multiple of chunk"):
        build(replace(config, context=33))
