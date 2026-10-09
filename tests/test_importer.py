"""IR importer: vLLM's wvSplitK (bf16, N=1, gfx942 config) sliced, compiled to IR, imported, linked into a
stand-in megakernel and compiled for gfx942 with the local LLVM. Skipped when the toolchain or the vLLM /
ROCm header checkouts are missing (see etx/importer/hip_ir_local.sh)."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from etx.importer.ir_import import ImportError_, import_kernel
from etx.importer.slice_device import slice_device

ROOT = Path(__file__).resolve().parents[1]
LLVM = Path(os.environ.get("ETX_LLVM", "/opt/homebrew/opt/llvm/bin"))
VLLM = Path(os.environ.get("ETX_VLLM", ROOT / "build/external/vllm"))
HDRS = Path(os.environ.get("ETX_ROCM_HEADERS", ROOT / "build/external/rocm-headers"))
ASSEMBLER = LLVM / ("llvm-as" if (LLVM / "llvm-as").exists() else "llvm-link")
need = pytest.mark.skipif(not ((LLVM / "clang++").exists() and (VLLM / "csrc/rocm/skinny_gemms.cu").exists() and (HDRS / "inc/hip").exists()),
                          reason="local toolchain unavailable; set ETX_LLVM, ETX_VLLM and ETX_ROCM_HEADERS")

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
def test_wvsplitk_import_links_and_inlines(tmp_path, monkeypatch):
    monkeypatch.delenv("ETX_DEVLIBS", raising=False)  # import first; final-link settings must not lower IDs early
    sl = tmp_path / "skinny.hip"
    sl.write_text(slice_device((VLLM / "csrc/rocm/skinny_gemms.cu").read_text(), r"wvSplitKQ|fp8|Fp8|FP8") + INST)
    ll = tmp_path / "skinny.ll"
    run(["bash", str(ROOT / "etx/importer/hip_ir_local.sh"), str(sl), str(ll), "-I", str(ROOT / "etx/importer/shim_nofp8"),
         "-I", str(VLLM / "csrc"), "-I", str(VLLM / "csrc/rocm")])
    out, info = import_kernel(ll.read_text(), "wvSplitK_hf_sml_", "etx_imp_wv", (64, 16, 1), 1024, lds_cap=24576)
    assert len(info.params) == 12 and info.slots == 1 and info.lds_bytes == 24576 and info.arg_bytes == 80
    imp = tmp_path / "imp.ll"; imp.write_text(out)
    run([str(ASSEMBLER), str(imp), "-o", str(tmp_path / "imp.bc")])
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


SYN = """%"struct.T" = type { [16 x float] }
@k.s = internal addrspace(3) global %"struct.T" undef, align 16
@k.v = internal addrspace(3) global float undef, align 4

define amdgpu_kernel void @k(ptr addrspace(1) noundef %o, i32 noundef %n) #0 {
  %t = tail call i64 @__ockl_get_local_id(i32 noundef 0)
  %b = tail call i64 @__ockl_get_group_id(i32 noundef 0)
  %t32 = trunc i64 %t to i32
  %p = getelementptr inbounds float, ptr addrspace(3) @k.s, i32 %t32
  store float 1.0, ptr addrspace(3) %p, align 4
  tail call void @llvm.amdgcn.s.barrier()
  %x = load float, ptr addrspace(3) @k.v, align 4
  %q = getelementptr inbounds float, ptr addrspace(1) %o, i64 %b
  store float %x, ptr addrspace(1) %q, align 4
  ret void
}
declare i64 @__ockl_get_local_id(i32)
declare i64 @__ockl_get_group_id(i32)
declare void @llvm.amdgcn.s.barrier()
attributes #0 = { "amdgpu-flat-work-group-size"="1,64" "amdgpu-no-workgroup-id-x" }
"""


def test_import_slots_and_quoted_struct_lds():
    out, info = import_kernel(SYN, "k", "etx_imp_k", (16, 1, 1), 256)
    assert info.slots == 4 and info.slot_threads == 64 and info.slot_barrier
    assert info.lds_bytes_per_slot == 80 and info.lds_bytes == 320          # the quoted struct (64 B) + the float, 16-aligned
    assert "@k.s" not in out and "@k.v" not in out                            # both moved into the arena
    assert out.count("define internal void @etx_imp_k.body.s") == 4
    assert "@etx_imp_slot_barrier(i32 %etx.slot, i32 1)" in out
    assert "icmp ult i32 %tid, 16" in out                                     # lanes past the 16-thread block skip the body
    assert "call i64 @__ockl_get" not in out                                  # every id read rewritten


def test_static_lds_over_cap_is_rejected_instead_of_truncated():
    with pytest.raises(ImportError_, match=r"static LDS needs 80 B per slot, exceeding the 64 B limit"):
        import_kernel(SYN, "k", "etx_imp_k", (16, 1, 1), 256, lds_cap=64)


def test_slice_enters_namespace_holding_host_launchers():
    # topk_softmax_kernels.cu: kernels and their torch launchers share `namespace vllm { namespace moe {`
    src = ('namespace vllm {\nnamespace moe {\n__global__ void k(float* x) { x[0] = 1; }\n'
           'void launch(torch::stable::Tensor& t) { k<<<1, 1>>>(nullptr); }\n} // namespace moe\n} // namespace vllm\n')
    out = slice_device(src)
    assert "__global__ void k" in out and "torch::stable" not in out
    body = re.sub(r"//[^\n]*", "", out)
    assert body.count("{") == body.count("}")


def test_slice_keeps_device_only_libtorch_stable_headers():
    out = slice_device('#include "libtorch_stable/quantization/vectorization.cuh"\n#include "libtorch_stable/torch_utils.h"\n')
    assert out.startswith('#include "libtorch_stable/quantization/vectorization.cuh"') and "dropped] #include \"libtorch_stable/torch_utils.h" in out


def test_triton_downgrade_float_literals_and_attrs():
    from etx.importer.triton_import import _fp_bits, downgrade_ir, dynamic_lds_to_static
    # llvm-dis 23 prints these bit patterns as the literals on the left (checked with the local LLVM 23)
    assert _fp_bits("bfloat", "+", "nan", 0x3F) == "0xR7FFF"
    assert _fp_bits("bfloat", "+", "qnan", 0) == "0xR7FC0"
    assert _fp_bits("bfloat", "+", "snan", 1) == "0xR7F81"
    assert _fp_bits("bfloat", "-", "qnan", 0) == "0xRFFC0"
    assert _fp_bits("half", "+", "qnan", 0) == "0xH7E00"
    assert _fp_bits("float", "+", "qnan", 0) == "0x7FF8000000000000"
    assert _fp_bits("bfloat", "+", "inf", 0) == "0xR7F80"
    t, notes = downgrade_ir("select <2 x i1> %c, <2 x bfloat> splat (bfloat +nan(0x3F)), <2 x bfloat> %v\n"
                            "attributes #2 = { convergent nocreateundeforpoison nounwind }\n")
    assert "bfloat 0xR7FFF" in t and "nocreateundeforpoison" not in t and len(notes) == 2
    lds = dynamic_lds_to_static("@global_smem = external addrspace(3) global [0 x i8], align 16\n", 20480)
    assert lds.startswith("@global_smem = internal addrspace(3) global [20480 x i8] undef, align 16")


@pytest.mark.skipif(not ASSEMBLER.exists(), reason="local LLVM not available")
def test_triton_downgrade_hex_literals_parse_back(tmp_path):
    # the hex literal must name the same value LLVM 23 printed: round-trip through the local LLVM 23
    from etx.importer.triton_import import downgrade_ir
    src = "define <2 x bfloat> @f() {\n  ret <2 x bfloat> <bfloat +nan(0x3F), bfloat -qnan>\n}\n"
    t, _ = downgrade_ir(src)
    (tmp_path / "a.ll").write_text(t)
    run([str(ASSEMBLER), str(tmp_path / "a.ll"), "-o", str(tmp_path / "a.bc")])
    back = run([str(LLVM / "llvm-dis"), str(tmp_path / "a.bc"), "-o", "-"])
    # LLVM 22 prints hex bits; LLVM 23 prints named NaNs. Both must preserve
    # the payload and sign, rather than merely parsing successfully.
    assert ("bfloat +nan(0x3F), bfloat -qnan" in back or
            "bfloat 0xR7FFF, bfloat 0xRFFC0" in back)
