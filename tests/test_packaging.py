"""Installed resource conformance and external-only CPU workload smoke checks."""
from etx.testing import installed_package_probe


def test_wheel_contains_importer_shim_headers(tmp_path):
    result = installed_package_probe(tmp_path / 'package')
    contents = set(result['wheel_files'])
    expected = {
        "etx/importer/hip_ir_local.sh",
        "etx/importer/shim/etx_cuda2hip.h",
        "etx/importer/shim/cuda_runtime.h",
        "etx/importer/shim/ATen/cuda/CUDAContext.h",
        "etx/importer/shim_nofp8/quantization/w8a8/fp8/common.cuh",
        "etx/build/toolchain.py",
        "etx/operators/tiles/gemm_rs.hip",
        "etx/operators/hosts/gemm_rs.hip",
        "etx/operators/hosts/splitk.hip",
        "etx/operators/hosts/moe.hip",
        "etx/operators/tiles/splitk.hip",
        "etx/operators/tiles/moe.hip",
        "etx/experiments/specs.json",
        "etx/testing/data/sealed/original-host.hip",
        "etx/testing/data/sealed/original-host.json",
    }
    missing = expected - contents
    assert not missing, f"wheel is missing importer files: {sorted(missing)}"
