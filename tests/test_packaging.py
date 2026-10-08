"""Build a wheel and verify that the importer ships the shim headers it includes."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wheel_contains_importer_shim_headers(tmp_path):
    project = tmp_path / "project"
    ignored = shutil.ignore_patterns(
        ".git", ".venv", "build", "dist", "etx.egg-info", "__pycache__", ".pytest_cache"
    )
    shutil.copytree(ROOT, project, ignore=ignored)
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheel_dir), str(project)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        contents = set(wheel.namelist())
    expected = {
        "etx/importer/hip_ir_local.sh",
        "etx/importer/shim/etx_cuda2hip.h",
        "etx/importer/shim/cuda_runtime.h",
        "etx/importer/shim/ATen/cuda/CUDAContext.h",
        "etx/importer/shim_nofp8/quantization/w8a8/fp8/common.cuh",
    }
    missing = expected - contents
    assert not missing, f"wheel is missing importer files: {sorted(missing)}"
