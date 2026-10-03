from dataclasses import replace
import importlib
from pathlib import Path
import shutil
import pytest

from examples.llm.model import load_dims
from etx.importer.launch_spec import launch_for, verify_capture, CAPTURE_ROOT, validate_wvsplitk

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,up_grid,down_grid,bn", [
    ("qwen3-30b-a3b", 192, 256, 64), ("mixtral-8x7b", 3584, 512, 16),
])
def test_captured_launches(name, up_grid, down_grid, bn):
    d = load_dims(str(ROOT / f"examples/llm/configs/{name}.json"))
    for role, grid in (("up", up_grid), ("down", down_grid)):
        spec = launch_for(d, role)
        assert spec.grid == (grid, 1, 1) and spec.block_n == bn
        spec.validate(grid=spec.grid, n=spec.n, k=spec.k, top_k=d.TOPK, block_m=spec.block_m)
        with pytest.raises(ValueError, match="conflicts"):
            spec.validate(grid=(grid // 4, 1, 1), n=spec.n, k=spec.k, top_k=d.TOPK, block_m=spec.block_m)
        with pytest.raises(ValueError, match="recapture"):
            launch_for(replace(d, MI=d.MI + 32), role)


def test_capture_change_is_rejected(tmp_path):
    export = "fused_moe_up_e8"
    shutil.copytree(CAPTURE_ROOT / export, tmp_path / export)
    (tmp_path / export / "fused_moe_kernel.ttir").write_text("wrong capture")
    with pytest.raises(ValueError, match="fingerprint"):
        verify_capture(export, tmp_path)


def test_historical_cucount_hole_fails():
    kwargs = dict(grid=(256, 1, 1), rows=32000, ytile=2, wave_groups=16)
    validate_wvsplitk(**kwargs, cu_count=256)
    with pytest.raises(ValueError, match="CuCount"):
        validate_wvsplitk(**kwargs, cu_count=304)


@pytest.mark.parametrize("name", ["qwen3-30b-a3b", "mixtral-8x7b"])
def test_model_uses_the_same_specification(monkeypatch, name):
    monkeypatch.setenv("ETX_VL_IMPORTED", "1")
    monkeypatch.setenv("ETX_VL_DSL_GEMV", "0")
    import examples.vllm_llm.model as vm
    vm = importlib.reload(vm)
    d = load_dims(str(ROOT / f"examples/llm/configs/{name}.json"))
    rows = vm.instances(d)
    for role in ("up", "down"):
        spec = launch_for(d, role)
        matches = [r for r in rows if r["role"] == f"moe_{role}"]
        assert len(matches) == d.L
        assert all(r["grid"] == spec.grid and r["export"] == spec.export for r in matches)


@pytest.mark.parametrize('export', ['fused_moe_up', 'fused_moe_down', 'fused_moe_up_e8', 'fused_moe_down_e8'])
def test_actual_captured_ir_imports_without_gpu(export):
    from etx.importer.triton_import import import_triton
    spec = verify_capture(export)
    source, info, meta, notes = import_triton(CAPTURE_ROOT / export, 'fused_moe_kernel', f'etx_imp_{export}', 1024, 1)
    assert info.block == (spec['num_warps'] * 64, 1, 1)
    assert info.lds_bytes_per_slot == spec['shared_bytes']
    assert info.slots == 1 and info.arg_bytes > 0
    assert f'@etx_imp_{export}' in source


def test_conflicting_spec_constants_are_rejected(monkeypatch):
    from etx.importer import launch_spec
    records = launch_spec.registry()
    records['fused_moe_up_e8']['constants']['BLOCK_SIZE_N'] = 64
    monkeypatch.setattr(launch_spec, 'registry', lambda: records)
    with pytest.raises(ValueError, match='block constants'):
        launch_spec.verify_capture('fused_moe_up_e8')


def test_import_does_not_inherit_final_device_libraries(monkeypatch, tmp_path):
    from etx.importer import vllm_kernels
    from types import SimpleNamespace
    monkeypatch.setenv('ETX_DEVLIBS', '/final-link-only')
    seen = {}
    def run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stderr='')
    monkeypatch.setattr(vllm_kernels.subprocess, 'run', run)
    vllm_kernels.compile_ir(tmp_path/'input.hip', tmp_path/'output.ll', tmp_path, vllm_kernels.SOURCES[0], 'local')
    assert 'ETX_DEVLIBS' not in seen['env']


@pytest.mark.parametrize('override', [
    {'grid': (128, 2, 1)}, {'grid': (128, 1, 2)},
    {'cu_count': 256.0}, {'ytile': True}, {'ytile': 2.0}, {'wave_groups': 1.5},
])
def test_wvsplitk_rejects_unsupported_grid_and_noninteger_abi(override):
    args = dict(grid=(256, 1, 1), cu_count=256, rows=8192, ytile=2, wave_groups=16)
    with pytest.raises(ValueError):
        validate_wvsplitk(**(args | override))
