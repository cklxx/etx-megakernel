"""ETX_EMIT_DSL_BIGK: the MoE tiles' chunked K > 4096 path exists only when the plan says a shape needs it
(round 43: as a resident runtime branch it cost Qwen +0.12 ms), and the host guard switches on the same macro."""
import os
import re

from etx.ops import gemv


def test_bigk_is_preprocessor_gated():
    src = gemv.emit_tiles()
    # the template and both body branches are inside #if ETX_EMIT_DSL_BIGK ... #endif
    assert src.count("#if ETX_EMIT_DSL_BIGK") >= 3
    for blk in re.split(r"#if ETX_EMIT_DSL_BIGK", src)[1:]:
        assert "#endif" in blk
    # with the macro off, the small-K path's LDS bases are the original runtime-K expressions
    assert "float* g = xs + d.K; float* part = g + 2 * d.rpt;" in src
    assert "int* lids = (int*)(xs + d.K + 16 * DSL_RCH)" in src
    # and the chunked path accumulates through dsl_gemv_bigk
    assert "dsl_gemv_bigk" in src


def test_bigk_derivation(monkeypatch):
    # the plan publishes dsl_bigk exactly when a MoE K exceeds the staged window
    from etx.codegen.manifest import FEATURES
    assert FEATURES["dsl_bigk"][0] == "ETX_EMIT_DSL_BIGK" and FEATURES["dsl_bigk"][1] == "value"


def test_host_guard_switches_on_the_same_macro():
    hip = open(os.path.join(os.path.dirname(__file__), "..", "examples", "vllm_llm", "host.hip")).read()
    # both the relaxed and the strict guards exist, selected by the manifest macro -- never by a -D
    assert hip.count("#if ETX_EMIT_DSL_BIGK") == 2
    assert 'die("dsl_moe_up: MI % 32 or H > 4096' in hip          # the strict branch survives
    assert 'die("dsl_moe_up: MI % 32 or H % 256 (K > 4096 is chunked' in hip
