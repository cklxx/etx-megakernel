"""The original-replay branch is valid only for replicated x-form imports."""
import sys

import pytest

from test_moe_mapping import ROOT, _build


@pytest.fixture
def isolated_model():
    # _build reloads this shared module and build() also derives DSL_ATTNHEAD.
    # Restore the whole namespace, including cached/derived values, not just env.
    path = sys.path[:]
    from examples.vllm_llm import model
    saved = vars(model).copy()
    try:
        yield
    finally:
        vars(model).clear()
        vars(model).update(saved)
        sys.path[:] = path


def test_replicated_imports_are_single_task_without_precopy(tmp_path, monkeypatch, isolated_model):
    _build(tmp_path, monkeypatch, str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"), "head")
    rows = [line.split() for line in (tmp_path / "vl_plan.txt").read_text().splitlines()[2:]]
    replicated = [r for r in rows if not r[1].startswith("dsl_") and int(r[12]) > 1]
    assert replicated, "fixture must exercise the replicated original-launch branch"
    for r in replicated:
        assert int(r[8]) == 1 and int(r[14]) == -1, r
    assert any(r[1] == "rms_norm_2d" and r[2:4] == ["0", "in_norm"] for r in replicated)


def test_check_source_contract():
    host = (ROOT / "examples/vllm_llm/host.hip").read_text()
    assert "--check needs a plan without per-XCD replicas" not in host
    assert 'a == "--check-pos"' in host
    assert 'CHECK-IMPORTED checked=%d planned=%d replicas=%d position=%d' in host
