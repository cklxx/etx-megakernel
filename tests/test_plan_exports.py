"""Build import closure and launch settings for dense fallbacks and A/B builds."""
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_helper(model, env=None, *args):
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("ETX_", "DSL_"))}
    result = subprocess.run(
        [sys.executable, "-m", "examples.vllm_llm.plan_exports",
         f"examples/llm/configs/{model}.json", *args],
        cwd=ROOT, env={**clean, **(env or {})}, text=True, capture_output=True, check=True)
    return result.stdout


@pytest.mark.parametrize("model", ["qwen3-14b", "qwen2.5-7b"])
def test_dense_imports_include_both_projection_tiles_and_residual_norm(model):
    names = set(run_helper(model, None, "--kind", "imported").splitlines())
    assert {"wvsplitk_y2", "wvsplitk_l40_y2", "fused_add_rms_norm"} <= names
    assert not any(n.startswith("dsl_") for n in names)


def test_unused_moe_activation_is_not_imported():
    assert run_helper("qwen3-30b-a3b", None, "--kind", "imported").splitlines() == [
        "attn_mfma16_g8", "attn_reduce", "rms_norm_2d"]


@pytest.mark.parametrize("model", ["qwen3-8b", "qwen3-30b-a3b", "mixtral-8x7b"])
@pytest.mark.parametrize("override", [
    {}, {"ETX_VL_IMPORTED": "1"}, {"ETX_VL_DSL_GEMV": "0"},
    {"ETX_VL_DSL_TOPK": "0"}, {"ETX_VL_DSL_FOLD": "0"},
    {"ETX_VL_DSL_ATTN": "0", "ETX_VL_QKNR": "grid"},
])
def test_export_closure_matches_compile_instances(model, override):
    records = run_helper(model, override).splitlines()
    assert records == sorted(set(records))
    names = {line.split("\t")[0] for line in records}
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("ETX_", "DSL_"))}
    expected = subprocess.check_output([
        sys.executable, "-c",
        "import json,sys; from examples.llm.model import load_dims; "
        "from examples.vllm_llm.model import instances; "
        "print(json.dumps(sorted({p['export'] for p in instances(load_dims(sys.argv[1]))})))",
        f"examples/llm/configs/{model}.json"], cwd=ROOT, env={**clean, **override}, text=True)
    assert names == set(json.loads(expected))


def test_settings_filter_caller_slots_and_use_planned_attention_grid():
    settings = run_helper("qwen3-14b", {"ETX_VL_SLOTS": "absent=8,attn_reduce=1"}, "--shell")
    assert "SLOTS=attn_mfma16_g5=1,attn_reduce=5\n" in settings
    assert "absent" not in settings
    assert "reshape_and_cache" not in settings
    assert "rms_norm_2d=640" in settings


@pytest.mark.parametrize("model", ["qwen3-14b", "qwen2.5-7b", "qwen3-30b-a3b"])
def test_dsl_emitters_match_final_plan(model):
    settings = run_helper(model, None, "--shell")
    value = next(line.split("=", 1)[1] for line in settings.splitlines()
                 if line.startswith("DSLOPS="))
    ops = set(filter(None, shlex.split(value)[0].split(",")))
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("ETX_", "DSL_"))}
    exports = set(json.loads(subprocess.check_output([
        sys.executable, "-c",
        "import json,sys; from examples.llm.model import load_dims; "
        "from examples.vllm_llm.model import instances; "
        "print(json.dumps(sorted({p['export'] for p in instances(load_dims(sys.argv[1]))})))",
        f"examples/llm/configs/{model}.json"], cwd=ROOT, env=clean, text=True)))
    assert ("gemv" in ops) == bool(exports & {"dsl_gemv", "dsl_gemv_fold"})
    assert ("topk" in ops) == ("dsl_topk" in exports)
    assert ("attnprep" in ops) == ("dsl_attnprep" in exports)
