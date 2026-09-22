import json
import shutil
import subprocess
from pathlib import Path

import pytest

from etx.codegen import emit_kernel, emit_lowering_header, emit_plan_json
from etx.frontends import emit_tile_decls
from etx.machine import list_archs, load_machine
from etx.passes import compile_graph
from examples import gemm_rs, moe_layer, splitk_sum

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("arch", list_archs())
def test_lowering_header_has_every_scope(arch):
    h = emit_lowering_header(load_machine(arch))
    for s in ("DOMAIN", "DEVICE", "SYSTEM"):
        for p in ("RELEASE", "ACQUIRE", "POLL", "ARRIVE"):
            assert f"#define ETX_{p}_{s}" in h
    assert "ETX_BACKOFF" in h and "ETX_DOMAIN_ID" in h


def test_kernel_text_dispatches_every_grid():
    b = moe_layer.bindings()
    plan = compile_graph(moe_layer.build(), "gfx942", b, moe_layer.runtime(b))
    src = emit_kernel(plan)
    for g in plan.graph.grids:
        assert f"case {plan.type_ids[g.name]}:" in src and g.body.symbol in src
    assert "etx_wait_DEVICE" in src or "etx_wait_DOMAIN" in src
    assert "etx_push_consumers" in src          # dynamic segments push
    assert "prologue recomputes norm" in src    # Pass 3 visible in the kernel
    decls = emit_tile_decls(plan.graph)
    assert "etx_tile_group_gemm_up" in decls
    data = json.loads(emit_plan_json(plan))
    assert data["arch"] == "gfx942" and len(data["descs"]) == len(plan.tasks)
    assert any(e["runtime_init"] for e in data["events"])


def test_multi_device_emits_one_kernel_per_device():
    b = gemm_rs.bindings()
    plan = compile_graph(gemm_rs.build(2), "gfx90a", b, {})
    k0, k1 = emit_kernel(plan, 0), emit_kernel(plan, 1)
    assert "etx_tile_reduce_scatter" in k0 and "etx_tile_reduce_scatter" in k1
    assert "etx_wait_SYSTEM" in k0


@pytest.mark.skipif(shutil.which("hipcc") is None, reason="hipcc not installed")
def test_generated_hip_compiles(tmp_path):
    b = splitk_sum.bindings()
    plan = compile_graph(splitk_sum.build(), "gfx942", b, {})
    (tmp_path / "etx_lowering.h").write_text(emit_lowering_header(plan.machine))
    src = tmp_path / "k.hip"
    src.write_text(emit_kernel(plan))
    inc = ROOT / "etx" / "runtime" / "include"
    subprocess.run(["hipcc", "--offload-arch=gfx942", "-fgpu-rdc", "-fsyntax-only", f"-I{inc}", f"-I{tmp_path}", str(src)], check=True)
