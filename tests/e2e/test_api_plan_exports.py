"""Historical plan-export CLI workflows through the external process boundary.

This migrates test execution, not the CLI implementation or model defaults.
"""
import json
import shlex
from pathlib import Path
import sys

import pytest

from etx.external import ProcessRequest, execute
from etx.testing import plan_export_instances

ROOT = Path(__file__).resolve().parents[2]


def run_helper(tmp_path, model, env=None, *args):
    clean = {'PATH': '/usr/bin:/bin'}  # No ambient ETX/DSL settings or secrets in receipts.
    output = tmp_path / 'plan-exports'
    result = execute(ProcessRequest(
        (sys.executable, '-u', '-m', 'examples.vllm_llm.plan_exports',
         f'examples/llm/configs/{model}.json', *args),
        str(ROOT), 30, environment={**clean, **(env or {})}), output=output)
    text = (output / 'output.log').read_text()
    assert result.status == 'complete' and result.returncode == 0, text
    assert json.loads((output / 'request.json').read_text())['environment'] == {**clean, **(env or {})}
    return text


@pytest.mark.parametrize("model", ["qwen3-14b", "qwen2.5-7b"])
def test_dense_imports_include_both_projection_tiles_and_residual_norm(tmp_path, model):
    names = set(run_helper(tmp_path, model, None, "--kind", "imported").splitlines())
    assert {"wvsplitk_y2", "wvsplitk_l40_y2", "fused_add_rms_norm"} <= names
    assert not any(n.startswith("dsl_") for n in names)


def test_unused_moe_activation_is_not_imported(tmp_path):
    assert run_helper(tmp_path, "qwen3-30b-a3b", None, "--kind", "imported").splitlines() == [
        "attn_mfma16_g8", "attn_reduce", "rms_norm_2d"]


@pytest.mark.parametrize("model", ["qwen3-8b", "qwen3-30b-a3b", "mixtral-8x7b"])
@pytest.mark.parametrize("override", [
    {}, {"ETX_VL_IMPORTED": "1"}, {"ETX_VL_DSL_GEMV": "0"},
    {"ETX_VL_DSL_TOPK": "0"}, {"ETX_VL_DSL_FOLD": "0"},
    {"ETX_VL_DSL_ATTN": "0", "ETX_VL_QKNR": "grid"},
])
def test_export_closure_matches_compile_instances(tmp_path, model, override):
    records = run_helper(tmp_path, model, override).splitlines()
    assert records == sorted(set(records))
    names = {line.split("\t")[0] for line in records}
    expected = plan_export_instances(model, output=tmp_path / 'compile-instances',
                                    environment=override)
    assert names == set(expected)


def test_settings_filter_caller_slots_and_use_planned_attention_grid(tmp_path):
    settings = run_helper(tmp_path, "qwen3-14b", {"ETX_VL_SLOTS": "absent=8,attn_reduce=1"}, "--shell")
    assert "SLOTS=attn_mfma16_g5=1,attn_reduce=5\n" in settings
    assert "absent" not in settings
    assert "reshape_and_cache" not in settings
    assert "rms_norm_2d=640" in settings


@pytest.mark.parametrize("model", ["qwen3-14b", "qwen2.5-7b", "qwen3-30b-a3b"])
def test_dsl_emitters_match_final_plan(tmp_path, model):
    settings = run_helper(tmp_path, model, None, "--shell")
    value = next(line.split("=", 1)[1] for line in settings.splitlines()
                 if line.startswith("DSLOPS="))
    ops = set(filter(None, shlex.split(value)[0].split(",")))
    exports = set(plan_export_instances(model, output=tmp_path / 'compile-instances'))
    assert ("gemv" in ops) == bool(exports & {"dsl_gemv", "dsl_gemv_fold"})
    assert ("topk" in ops) == ("dsl_topk" in exports)
    assert ("attnprep" in ops) == ("dsl_attnprep" in exports)
