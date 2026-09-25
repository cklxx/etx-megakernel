"""IR importer: vLLM's wvSplitK (bf16, N=1, gfx942 config) sliced, compiled to IR, imported, linked into a
stand-in megakernel and compiled for gfx942 with the local LLVM. Skipped when the toolchain or the vLLM /
ROCm header checkouts are missing (see etx/importer/hip_ir_local.sh)."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from etx.importer.ir_import import import_kernel
from etx.importer.slice_device import slice_device

ROOT = Path(__file__).resolve().parents[1]
LLVM = Path(os.environ.get("ETX_LLVM", "/opt/homebrew/opt/llvm/bin"))
VLLM = Path(os.environ.get("ETX_VLLM", Path.home() / "code/vllm"))
HDRS = Path(os.environ.get("ETX_ROCM_HEADERS", Path.home() / "code/rocm-headers"))
need = pytest.mark.skipif(not ((LLVM / "clang++").exists() and (VLLM / "csrc/rocm/skinny_gemms.cu").exists() and (HDRS / "inc/hip").exists()),
                          reason="local LLVM / vLLM / ROCm headers not available")

INST = ("\ntemplate __global__ void wvSplitK_hf_sml_<__hip_bfloat16, 64, 2, 16, 8, 2, 1>(const int, const int, const int, const int, "
        "const int, const int, const __hip_bfloat16*, const __hip_bfloat16* __restrict__, const __hip_bfloat16* __restrict__, "
        "__hip_bfloat16*, const int, const int);\n")

MK = """#define ETX_IMPORT_LDS 24576
#include "etx/import.h"
extern "C" __device__ void etx_imp_wv(const void* args, int blk, int tid, int slot);
extern "C" __global__ void __launch_bounds__(1024, 1) t_mk(const void* args, int ntasks) {
  etx_import_init();
  __syncthreads();
  for (int t = blockIdx.x; t < ntasks; t += gridDim.x) { etx_imp_wv(args, t, threadIdx.x, 0); __syncthreads(); }
}
"""


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_slice_keeps_kernels_drops_host():
    src = '#include <torch/all.h>\n__global__ void k(float* x) { x[0] = 1; }\nat::Tensor f(at::Tensor a) {\n#if A\n  { return a;\n#else\n  { return a;\n#endif\n  }\n}\nint g() { return 1; }\n'
    out = slice_device(src)
    assert "__global__ void k" in out and "int g()" in out
    assert "at::Tensor f" not in out and "torch/all.h>" not in out.replace("dropped] #include <torch/all.h>", "")


@need
def test_wvsplitk_import_links_and_inlines(tmp_path):
    sl = tmp_path / "skinny.hip"
    sl.write_text(slice_device((VLLM / "csrc/rocm/skinny_gemms.cu").read_text(), r"wvSplitKQ|fp8|Fp8|FP8") + INST)
    ll = tmp_path / "skinny.ll"
    run(["bash", str(ROOT / "etx/importer/hip_ir_local.sh"), str(sl), str(ll), "-I", str(ROOT / "etx/importer/shim_nofp8"),
         "-I", str(VLLM / "csrc"), "-I", str(VLLM / "csrc/rocm")])
    out, info = import_kernel(ll.read_text(), "wvSplitK_hf_sml_", "etx_imp_wv", (64, 16, 1), 1024, lds_cap=24576)
    assert len(info.params) == 12 and info.slots == 1 and info.lds_bytes == 24576 and info.arg_bytes == 80
    imp = tmp_path / "imp.ll"; imp.write_text(out)
    run([str(LLVM / "llvm-as"), str(imp), "-o", str(tmp_path / "imp.bc")])
    (tmp_path / "mk.hip").write_text(MK)
    run(["bash", str(ROOT / "etx/importer/hip_ir_local.sh"), str(tmp_path / "mk.hip"), str(tmp_path / "mk.ll"), "-I", str(ROOT / "etx/runtime/include"),
         "-Xclang", "-mlink-builtin-bitcode", "-Xclang", str(tmp_path / "imp.bc")])
    mk = (tmp_path / "mk.ll").read_text()
    assert "@etx_imp_wv" not in re.sub(r"^declare[^\n]*\n", "", mk, flags=re.M).split("define", 1)[1] or "call void @etx_imp_wv" not in mk
    run([str(LLVM / "llc"), "-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx942", "-O3", str(tmp_path / "mk.ll"), "-o", str(tmp_path / "mk.s")])
    asm = (tmp_path / "mk.s").read_text()
    assert int(re.search(r"\.vgpr_spill_count:\s+(\d+)", asm).group(1)) == 0
    assert int(re.search(r"\.group_segment_fixed_size:\s+(\d+)", asm).group(1)) >= 24576
    assert "v_mfma" in asm
