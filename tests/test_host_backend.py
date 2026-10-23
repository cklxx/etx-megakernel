"""CPU-only host backend checks. Compilation never loads a GPU runtime.

HIPCC / NVCC select compilers; ETX_HOST_HIP_FLAGS / ETX_HOST_CUDA_FLAGS
supply local SDK include/device-library flags for split SDK installations.
"""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "etx/runtime/host"


def run(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_vendor_calls_are_confined_to_backend():
    headers = [ROOT / "etx/runtime/include/etx/host.h", *BACKENDS.glob("*.h")]
    for path in headers:
        if path.name != "backend_hip.h":
            assert not re.search(r"\bhip[A-Z]\w*", path.read_text()), path
        if path.name == "host.h":
            assert not re.search(r"\bcuda[A-Z]\w*|#include\s*[<\"](?:hip/|cuda)",
                                 path.read_text())


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    out = tmp_path_factory.mktemp("host_backend")
    run([sys.executable, "-m", "etx", "compile", "examples/splitk_sum.py",
         "--arch", "gfx942", "--out", str(out)])
    return out


@pytest.mark.parametrize("backend", ["hip", "cuda"])
def test_real_host_compilation(backend, plan):
    name = "hipcc" if backend == "hip" else "nvcc"
    compiler = os.environ.get(name.upper()) or shutil.which(name)
    if not compiler:
        pytest.skip(f"{name} not installed; set {name.upper()} to enable SDK check")
    source = plan / ("host_" + backend + (".hip" if backend == "hip" else ".cu"))
    source.write_text(r'''
#include "plan_data.h"
#include "etx/host.h"
void compile_host(etx_host& h) {
  h.setup(); h.run(); h.run_unfused(); h.snapshot_events(h.ev_strided);
  h.report_phases(etx_type_names, ETX_N_TYPES);
  namespace b = etx::host_backend;
  void* p = nullptr;
  ETX_CHECK(b::malloc_fine_grained(&p, 64)); ETX_CHECK(b::free(p));
  ETX_CHECK(b::malloc_host_coherent(&p, 64)); ETX_CHECK(b::free_host(p));
  int can = 0; ETX_CHECK(b::device_can_access_peer(&can, 0, 1));
  ETX_CHECK(b::enable_peer_access(1)); ETX_CHECK(b::disable_peer_access(1));
  b::graph g = nullptr; b::graph_exec exec = nullptr;
  ETX_CHECK(b::graph_destroy(g)); ETX_CHECK(b::graph_exec_destroy(exec));
  ETX_CHECK(b::event_destroy(h.ev_start)); ETX_CHECK(b::stream_destroy(h.stream));
}
''')
    flags = shlex.split(os.environ.get(f"ETX_HOST_{backend.upper()}_FLAGS", ""))
    arch = ["--offload-arch=gfx942"] if backend == "hip" else []
    run([compiler, "-O3", "-std=c++17", *arch, *flags, "-DETX_HOUR_PROFILE",
         "-I" + str(ROOT / "etx/runtime/include"), "-I" + str(plan),
         "-c", str(source), "-o", str(source.with_suffix(".o"))])

    # Adapter symbols would change the host object ABI and add call overhead.
    nm = shutil.which("nm")
    if nm:
        symbols = subprocess.check_output(
            [nm, "-C", str(source.with_suffix(".o"))], text=True, timeout=30)
        assert "etx::host_backend::" not in symbols
