"""Offline occupancy checks; no CUDA runtime or GPU is required."""
from types import SimpleNamespace

import pytest

from etx.machine import load_machine
from etx.passes import compile_graph
from etx.passes.p1_tiling import _wg_per_cu
from examples import gemm_rs, moe_layer, splitk_sum


@pytest.mark.parametrize("arch", ["sm_90", "sm_100"])
@pytest.mark.parametrize("threads,regs,blocks", [
    # CUDA occupancy calculator, H100: 65536 registers/SM, 256 registers
    # allocated per warp, 64 warps/SM, no user shared memory.
    # Blackwell has the same limits for these cases.
    (256, 64, 4), (256, 128, 2), (256, 255, 1),
    (1024, 64, 1), (1024, 128, 0), (1024, 255, 0),
])
def test_cuda_calculator_blocks_per_sm(arch, threads, regs, blocks):
    plan = SimpleNamespace(machine=load_machine(arch))
    assert _wg_per_cu(plan, regs, 0, 0, threads) == blocks


@pytest.mark.parametrize("arch", ["sm_90", "sm_100"])
def test_register_granule_and_shared_memory_reservation(arch):
    plan = SimpleNamespace(machine=load_machine(arch))
    assert _wg_per_cu(plan, 65, 0, 0, 256) == 3
    assert _wg_per_cu(plan, 64, 1, 0, 256) == 3
    # Two blocks fit at 114 KiB only without the 1 KiB per-block reserve.
    assert _wg_per_cu(plan, 32, 0, 114 * 1024, 256) == 1
    assert _wg_per_cu(plan, 32, 0, 228 * 1024, 256) == 0


@pytest.mark.parametrize("arch", ["sm_90", "sm_100"])
def test_per_thread_limit_is_separate_from_register_file(arch):
    graph = splitk_sum.build()
    graph.grids[0].resource.vgpr = 256
    with pytest.raises(ValueError, match="registers exceed 255"):
        compile_graph(graph, arch, splitk_sum.bindings())


@pytest.mark.parametrize("module,expected", [
    (moe_layer, 2), (splitk_sum, 2), (gemm_rs, 1),
])
def test_gfx942_residency_unchanged(module, expected):
    bindings = module.bindings()
    runtime = module.runtime(bindings) if hasattr(module, "runtime") else {}
    plan = compile_graph(module.build(), "gfx942", bindings, runtime)
    assert plan.wg_per_cu == expected


@pytest.mark.parametrize("arch", ["sm_90", "sm_100"])
def test_launch_bounds_use_target_residency(arch):
    from etx.codegen import emit_kernel

    graph = splitk_sum.build()
    for grid in graph.grids:
        grid.resource.vgpr = 64
        grid.resource.agpr = 0
        grid.resource.threads = 256
        grid.resource.lds_bytes = 0
        grid.resource.prefetch_bytes = 0
    plan = compile_graph(graph, arch, splitk_sum.bindings())
    assert plan.wg_per_cu == 4
    assert "#define ETX_MIN_WG_PER_CU 4" in emit_kernel(plan)


@pytest.mark.parametrize("arch", ["sm_90", "sm_100"])
def test_infeasible_block_is_rejected(arch):
    graph = splitk_sum.build()
    graph.grids[0].resource.vgpr = 128
    graph.grids[0].resource.threads = 1024
    with pytest.raises(ValueError, match="cannot fit one workgroup"):
        compile_graph(graph, arch, splitk_sum.bindings())
