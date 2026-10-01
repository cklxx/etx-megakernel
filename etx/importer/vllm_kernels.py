"""The vLLM kernels ETX imports, and the driver that turns them into linkable bitcode.

Each entry names a vLLM source file, the specialisations vLLM's own host code launches for our models
(bf16, batch 1, gfx942 with 304 CUs; derived from the host dispatch code, cited per entry) and the block
shape of that launch. The driver slices the file (etx.importer.slice_device), appends a host function that
takes the address of every specialisation (so the device compile emits exactly those kernels), compiles it
to IR (hip_ir_local.sh here, the VM's hipcc there: same flags), imports each kernel (etx.importer.ir_import)
and assembles one bitcode file per export plus a JSON with the argument layouts.

  python -m etx.importer.vllm_kernels --vllm ~/code/vllm --out build/imp [--threads 1024] [--compiler local|hipcc]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from etx.importer.ir_import import import_kernel
from etx.importer.slice_device import slice_device

HERE = Path(__file__).resolve().parent
BF = "__hip_bfloat16"
CB = "c10::BFloat16"


@dataclass
class Kernel:
    export: str                   # etx_imp_<export>
    match: str                    # substring of the mangled name, unique among the file's specialisations
    spec: str                     # the specialisation, as written in the (void)&... keep line
    block: tuple[int, int, int]   # the launch's block shape (from vLLM's host code)
    lds_cap: int | None = None
    note: str = ""
    inline: bool = True           # False: an out-of-line call (own register allocation), see ir_import


@dataclass
class Source:
    path: str                     # relative to the vLLM checkout
    kernels: list[Kernel]
    drop: str | None = None       # extra definitions to drop (slice_device --drop)
    includes: list[str] = field(default_factory=list)   # extra -I (relative to etx/importer)
    defines: list[str] = field(default_factory=list)


SOURCES = [
    Source("csrc/rocm/skinny_gemms.cu", drop=r"wvSplitKQ|fp8|Fp8|FP8", includes=["shim_nofp8"], kernels=[
        # wvSplitK(): gfx9 -> WVSPLIT_TILE_CFG(THRDS=64, WVPRGRP=16, sYT, N); sYT = ceil(M / (CuCount*4));
        # N == 1 -> YTILE 2, UNRL 2 unless sYT <= 1 (YTILE 1, UNRL 4); Kbp*N <= LDS/2 -> the _sml_ variant.
        Kernel("wvsplitk_y2", "wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi2ELi16ELi8ELi2ELi1E",
               f"wvSplitK_hf_sml_<{BF}, 64, 2, 16, 8, 2, 1>", (64, 16, 1), lds_cap=32 * 1024,
               note="M > 1216 (every projection and lm_head); LDS holds the K-vector, K <= 16384"),
        Kernel("wvsplitk_y1", "wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi1E",
               f"wvSplitK_hf_sml_<{BF}, 64, 1, 16, 8, 4, 1>", (64, 16, 1), lds_cap=32 * 1024,
               note="M <= 1216 (the MoE router, 128 rows)"),
    ]),
    Source("csrc/rocm/attention.cu", defines=["ENABLE_FP8"], kernels=[
        # paged_attention_custom_launcher: grid (num_seqs, max_num_partitions, num_kv_heads), block 256;
        # gqa <= 4 -> mfma4, else mfma16; then the reduction, grid (num_heads, num_seqs), block head_size.
        Kernel("attn_mfma4_g4", "QKV_mfma4_kernelI14__hip_bfloat16S0_LN4vllm18Fp8KVCacheDataTypeE0ES0_Li16ELi128ELi256ELb0ELi4E",
               f"paged_attention_ll4mi_QKV_mfma4_kernel<{BF}, {BF}, vllm::Fp8KVCacheDataType::kAuto, {BF}, 16, 128, 256, false, 4>", (256, 1, 1),
               note="Qwen3-8B (32/8 heads)"),
        Kernel("attn_mfma16_g6", "QKV_mfma16_kernelI14__hip_bfloat16S0_LN4vllm18Fp8KVCacheDataTypeE0ES0_Li16ELi128ELi256ELb0ELi6E",
               f"paged_attention_ll4mi_QKV_mfma16_kernel<{BF}, {BF}, vllm::Fp8KVCacheDataType::kAuto, {BF}, 16, 128, 256, false, 6, MFMAType::F16>", (256, 1, 1),
               note="Qwen2.5-1.5B (12/2 heads)"),
        Kernel("attn_mfma16_g8", "QKV_mfma16_kernelI14__hip_bfloat16S0_LN4vllm18Fp8KVCacheDataTypeE0ES0_Li16ELi128ELi256ELb0ELi8E",
               f"paged_attention_ll4mi_QKV_mfma16_kernel<{BF}, {BF}, vllm::Fp8KVCacheDataType::kAuto, {BF}, 16, 128, 256, false, 8, MFMAType::F16>", (256, 1, 1),
               note="Qwen3-30B-A3B (32/4 heads)"),
        Kernel("attn_reduce", "reduce_kernelI14__hip_bfloat16S0_Li128ELi128ELi256ELi1E",
               f"paged_attention_ll4mi_reduce_kernel<{BF}, {BF}, 128, 128, 256, 1>", (128, 1, 1),
               note="npar_loops = ceil(partitions / 64) = 1 up to 16K tokens"),
    ]),
    Source("csrc/libtorch_stable/layernorm_kernels.cu", kernels=[
        # rms_norm: grid num_tokens, block min(hidden / vec, 1024), vec = gcd(8, hidden); 2-D for a vector,
        # 3-D for q/k norm over [tokens, heads, head_dim].
        Kernel("rms_norm_2d", "rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1E", f"vllm::rms_norm_kernel<{CB}, 8, 2, true>", (0, 1, 1),
               note="block = hidden/8 (512 for 4096, 192 for 1536, 256 for 2048): set per model"),
        Kernel("rms_norm_3d", "rms_norm_kernelIN3c108BFloat16ELi8ELi3ELb1E", f"vllm::rms_norm_kernel<{CB}, 8, 3, true>", (16, 1, 1),
               note="q/k norm: head_dim 128 / 8 = 16 threads per head"),
        # fused_add_rms_norm: grid num_tokens, block min(hidden, 1024), width 8 when aligned
        Kernel("fused_add_rms_norm", "fused_add_rms_norm_kernelIN3c108BFloat16ELi8ELb1E", f"vllm::fused_add_rms_norm_kernel<{CB}, 8, true>", (1024, 1, 1)),
    ]),
    Source("csrc/libtorch_stable/pos_encoding_kernels.cu", kernels=[
        # rotary_embedding: grid num_tokens, block min(num_heads * rot_dim / 2, 512); neox style for Qwen
        Kernel("rope_neox", "rotary_embedding_kernelIN3c108BFloat16ES2_Lb1E", f"vllm::rotary_embedding_kernel<{CB}, {CB}, true>", (512, 1, 1),
               note="block min(NH*64, 512): 512 for every model here"),
    ]),
    Source("csrc/libtorch_stable/fused_qknorm_rope_kernel.cu", kernels=[
        # fused_qk_norm_rope (vLLM's QKNormRoPEFusionPass, compilation_config enable_qk_norm_rope_fusion): batch 1 ->
        # token_heads_per_warp 1 (auto-select only raises it on SM 9.0) -> launchFusedQKNormRope: block 256 (8 warps of
        # 32 lanes), grid ceil((NH + NKV) / 8), one warp per head; bf16 qkv and cos/sin cache, neox -> interleave false
        Kernel("qknorm_rope", "fusedQKNormRopeKernelIN3c108BFloat16E",
               f"tensorrt_llm::kernels::fusedQKNormRopeKernel<{CB}, {CB}, 128, false>", (256, 1, 1),
               note="head_dim 128 (Qwen3); grid 5 for 32 + 4 heads"),
    ]),
    Source("csrc/libtorch_stable/cache_kernels.cu", defines=["ENABLE_FP8"], kernels=[
        # reshape_and_cache: grid num_tokens, block min(num_kv_heads * head_size / x, 512), x = 16 / 2
        Kernel("reshape_and_cache", "reshape_and_cache_kernelI14__hip_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0E",
               f"vllm::reshape_and_cache_kernel<{BF}, {BF}, vllm::Fp8KVCacheDataType::kAuto>", (0, 1, 1),
               note="block = NKV * 16: 128 (8B), 32 (1.5B), 64 (30B-A3B): set per model"),
    ]),
    Source("csrc/libtorch_stable/activation_kernels.cu", kernels=[
        # silu_and_mul: grid num_tokens, block min(d / vec, 1024), vec = 16 / 2 (d % 8 == 0)
        Kernel("silu_and_mul", "act_and_mul_kernelIN3c108BFloat16E",
               f"vllm::act_and_mul_kernel<{CB}, typename vllm::PackedTypeConverter<{CB}>::Type, vllm::silu_kernel<{CB}>, "
               f"vllm::packed_silu_kernel<typename vllm::PackedTypeConverter<{CB}>::Type>, true, true, false, false>", (1024, 1, 1),
               note="dense: 1 row, block 1024; MoE (fused_experts): 8 rows of 768, block 768/8 = 96: set per model"),
    ]),
    # Qwen3-30B-A3B MoE at batch 1 (fused_moe/fused_moe.py, TritonExperts; docs/RESEARCH-moe-import.md): router
    # (wvsplitk_y1), top-k, two Triton fused_moe_kernel launches (etx.importer.triton_import), silu, moe_sum
    Source("csrc/libtorch_stable/moe/topk_softmax_kernels.cu", kernels=[
        # topkGatingLauncherHelper: BYTES_PER_LDG = min(16, 2 * 128) = 16, VPT = 128 / (64 / (256 / 16)) ... = 8,
        # ROWS_PER_WARP = 64 / 16 = 4; block (64, WARPS_PER_TB = 4); grid ceil(ceil(rows / 4) / 4) = 1 at batch 1
        Kernel("topk_softmax_e128", "topkGatingILi8ELi128ELi4ELi16ELi64Ei14__hip_bfloat16LNS0_11ScoringFuncE0E",
               f"vllm::moe::topkGating<8, 128, 4, 16, 64, int, {BF}, vllm::moe::ScoringFunc::SCORING_SOFTMAX>", (64, 4, 1),
               note="128 experts, softmax scoring, int32 ids (Qwen3-30B-A3B)"),
        # topkGating is instantiated per expert count by the launcher switch; BYTES = min(16, 2E), ELTS = BYTES/2,
        # VPT = max(1, E/(ELTS*64))*ELTS -- 8 for both E=8 and E=128. Slicing one expert count and running another
        # computes garbage deterministically (Mixtral top-2-of-8 through the e128 slice, 2026-10-02): shape-keyed.
        Kernel("topk_softmax_e8", "topkGatingILi8ELi8ELi4ELi16ELi64Ei14__hip_bfloat16LNS0_11ScoringFuncE0E",
               f"vllm::moe::topkGating<8, 8, 4, 16, 64, int, {BF}, vllm::moe::ScoringFunc::SCORING_SOFTMAX>", (64, 4, 1),
               note="8 experts (Mixtral-8x7B)"),
    ]),
    Source("csrc/libtorch_stable/moe/moe_align_sum_kernels.cu", kernels=[
        # moe_sum, vectorised path (hidden contiguous, 16 B aligned): block 256, grid min(ceil(tokens * d / 8 / 256), 65535)
        Kernel("moe_sum_k8", "moe_sum_vec_kernelIN3c108BFloat16EiLi8ELb0E",
               f"vllm::moe::moe_sum_vec_kernel<{CB}, int32_t, 8, false>", (256, 1, 1),
               note="top-8, not pad-aware (no expert map)"),
        # moe_sum_vec_kernel bakes TOPK into the template: the k8 slice at top-2 sums six rows nobody wrote
        # (the Mixtral failure root cause). Shape-keyed like the topk entries.
        Kernel("moe_sum_k2", "moe_sum_vec_kernelIN3c108BFloat16EiLi2ELb0E",
               f"vllm::moe::moe_sum_vec_kernel<{CB}, int32_t, 2, false>", (256, 1, 1),
               note="top-2 (Mixtral-8x7B), not pad-aware"),
    ]),
]

# Triton kernels, taken from the Triton cache vLLM wrote on the VM (vLLM compiles them at run time, so the exact kernel
# is the cached one; examples/vllm_llm/triton/<export>/CACHE_DIR names the cache entry). fused_moe_kernel at batch 1
# (naive block assignment, default config BLOCK_SIZE_M/N/K = 16/64/128, 4 warps, 20 KB LDS): num_valid_tokens = 8
# is not a multiple of 16, which selects the entries without that divisibility specialisation.
TRITON_DIR = HERE.parents[1] / "examples/vllm_llm/triton"
TRITON = {
    "fused_moe_up": ("fused_moe_kernel", "experts gate_up: top_k = 8, MUL_ROUTED_WEIGHT = False; grid 8 x 1536/64 = 192"),
    "fused_moe_down": ("fused_moe_kernel", "experts down: top_k = 1, MUL_ROUTED_WEIGHT = True; grid 8 x 2048/64 = 256"),
}


def _rocm_inc() -> list[str]:
    """ETX_ROCM_HEADERS: a ROCm include tree (hip/, hipcub/, rocprim/) for boxes whose ROCm install ships the
    compiler and device libraries but no HIP headers -- rsync ~/code/rocm-headers/inc there and point this at it.
    Empty when the toolchain finds its own headers, which is the normal case."""
    d = os.environ.get("ETX_ROCM_HEADERS", "")
    return ["-isystem", d] if d else []


def _flags(vllm: Path, s: Source) -> list[str]:
    th = os.environ.get("ETX_TORCH_HEADERS", str(Path.home() / "rocm-headers/pytorch"))
    inc = []
    for i in s.includes:
        inc += ["-I", str(HERE / i)]
    inc += ["-I", str(vllm / "csrc"), "-I", str(vllm / Path(s.path).parent), "-I", str(vllm / "csrc/libtorch_stable")]
    return (["-D__HIP_PLATFORM_AMD__", "-DUSE_ROCM", "-DNDEBUG", "-isystem", th] + _rocm_inc()
            + ["-include", str(HERE / "shim/etx_cuda2hip.h"), "-I", str(HERE / "shim")]
            + [f"-D{d}" for d in s.defines] + inc)


def compile_orig(out: Path, vllm: Path, exports: set[str] | None) -> list[Path]:
    """The original kernels, compiled as vLLM compiles them (host + device), each behind an accessor
    etx_orig_<export>() returning the kernel's host handle: the reference for host.hip --check."""
    objs, table = [], ['#include <hip/hip_runtime.h>', '#include <cstring>']
    names = []
    for s in SOURCES:
        ks = [k for k in s.kernels if exports is None or k.export in exports]
        if not ks:
            continue
        base = Path(s.path).stem
        src = out / f"orig_{base}.hip"
        src.write_text(f'#include "{base}.hip"\n' + "".join(f'extern "C" const void* etx_orig_{k.export}() {{ return (const void*)&{k.spec}; }}\n' for k in ks))
        obj = out / f"orig_{base}.o"
        cmd = ["hipcc", "-x", "hip", "-c", "--offload-arch=gfx942", "-std=c++17", "-O3", "-I", str(out)] + _flags(vllm, s) + [str(src), "-o", str(obj)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"compile orig {s.path} failed:\n{r.stderr[-4000:]}")
        objs.append(obj)
        names += [k.export for k in ks]
    # Triton originals: vLLM's own code object from its cache, launched with hipModuleLaunchKernel by host.hip
    tri = [e for e in TRITON if exports is None or e in exports]
    table += ['extern "C" bool etx_orig_triton(const char* n, const char** hsaco, const char** fn, int* shared) {']
    for e in tri:
        meta = json.loads((TRITON_DIR / e / f"{TRITON[e][0]}.json").read_text())
        table.append(f'  if (!strcmp(n, "{e}")) {{ *hsaco = "{TRITON_DIR / e / (TRITON[e][0] + ".hsaco")}"; *fn = "{TRITON[e][0]}"; '
                     f'*shared = {int(meta["shared"])}; return true; }}')
    table += ["  return false;", "}"]
    table += [f'extern "C" const void* etx_orig_{e}();' for e in names]
    table += ['extern "C" const void* etx_orig_find(const char* n) {'] + [f'  if (!strcmp(n, "{e}")) return etx_orig_{e}();' for e in names] + ["  return nullptr;", "}"]
    (out / "orig_table.hip").write_text("\n".join(table) + "\n")
    return objs


def compile_ir(src_hip: Path, out_ll: Path, vllm: Path, s: Source, compiler: str) -> None:
    inc = []
    for i in s.includes:
        inc += ["-I", str(HERE / i)]
    inc += ["-I", str(vllm / "csrc"), "-I", str(vllm / Path(s.path).parent), "-I", str(vllm / "csrc/libtorch_stable")]
    defs = [f"-D{d}" for d in s.defines]
    if compiler == "local":
        cmd = ["bash", str(HERE / "hip_ir_local.sh"), str(src_hip), str(out_ll)] + defs + inc
    else:   # the VM: ROCm's own hipcc, device only, before the device libraries
        th = os.environ.get("ETX_TORCH_HEADERS", str(Path.home() / "rocm-headers/pytorch"))     # torch/headeronly (sparse pytorch checkout)
        cmd = ["hipcc", "-x", "hip", "--cuda-device-only", "--offload-arch=gfx942", "-nogpulib", "-std=c++17", "-O3", "-S", "-emit-llvm",
               "-D__HIP_PLATFORM_AMD__", "-DUSE_ROCM", "-DNDEBUG", "-isystem", th] + _rocm_inc() + [
               "-include", str(HERE / "shim/etx_cuda2hip.h"), "-I", str(HERE / "shim")] + defs + inc + [str(src_hip), "-o", str(out_ll)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"compile {s.path} failed:\n{r.stderr[-4000:]}")


def run(vllm: Path, out: Path, threads: int, compiler: str, blocks: dict[str, tuple[int, int, int]], only: set[str] | None = None,
        slots: dict[str, int] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    from etx.importer.bundle import llvm_tool
    llvm_as = llvm_tool("llvm-as")
    if not Path(llvm_as).exists():            # ROCm ships llvm-link but not llvm-as; llvm-link reads textual IR too
        llvm_as = llvm_tool("llvm-link")
    report = {}
    for s in SOURCES:
        ks = [k for k in s.kernels if only is None or k.export in only]
        if not ks:
            continue
        base = Path(s.path).stem
        text = slice_device((vllm / s.path).read_text(), s.drop)
        text += "\n// etx import: the specialisations vLLM launches for our models\nvoid etx_keep_" + base + "() {\n"
        text += "".join(f"  (void)&{k.spec};\n" for k in ks) + "}\n"
        hip = out / f"{base}.hip"; hip.write_text(text)
        ll = out / f"{base}.ll"
        compile_ir(hip, ll, vllm, s, compiler)
        mod = ll.read_text()
        for k in ks:
            blk = blocks.get(k.export, k.block)
            if blk[0] == 0:
                raise SystemExit(f"{k.export}: the block shape depends on the model ({k.note}); pass it in blocks")
            txt, info = import_kernel(mod, k.match, f"etx_imp_{k.export}", blk, threads, k.lds_cap, k.inline, (slots or {}).get(k.export))
            (out / f"{k.export}.ll").write_text(txt)
            r = subprocess.run([llvm_as, str(out / f"{k.export}.ll"), "-o", str(out / f"{k.export}.bc")], capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"llvm-as {k.export}: {r.stderr[-2000:]}")
            report[k.export] = info.to_json() | {"source": s.path, "spec": k.spec, "note": k.note}
    from etx.importer.triton_import import import_triton
    for e, (kname, note) in TRITON.items():
        if only is not None and e not in only:
            continue
        txt, info, meta, notes = import_triton(TRITON_DIR / e, kname, f"etx_imp_{e}", threads, (slots or {}).get(e))
        (out / f"{e}.ll").write_text(txt)
        r = subprocess.run([llvm_as, str(out / f"{e}.ll"), "-o", str(out / f"{e}.bc")], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"llvm-as {e}: {r.stderr[-2000:]}")
        report[e] = info.to_json() | {"source": f"triton:{(TRITON_DIR / e / 'CACHE_DIR').read_text().strip()}", "spec": kname,
                                      "note": note, "downgrade": notes}
    old = json.loads((out / "imports.json").read_text()) if only and (out / "imports.json").exists() else {}
    (out / "imports.json").write_text(json.dumps(old | report, indent=1))     # --only updates, it does not drop the rest
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=str(Path.home() / "code/vllm"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1024)
    ap.add_argument("--compiler", choices=["local", "hipcc"], default="local")
    ap.add_argument("--block", action="append", default=[], help="export=bx,by,bz for model-dependent launches")
    ap.add_argument("--only", default=None, help="comma-separated exports")
    ap.add_argument("--orig", action="store_true", help="also compile the original kernels (hipcc) for host.hip --check")
    ap.add_argument("--slots", default="", help="export=N,... : at most N of the kernel's blocks per megakernel workgroup")
    a = ap.parse_args()
    blocks = {}
    for b in a.block:
        e, v = b.split("=")
        t = tuple(int(x) for x in v.split(","))
        blocks[e] = t + (1,) * (3 - len(t))
    only = set(a.only.split(",")) if a.only else None
    slots = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.slots.split(",") if kv}
    rep = run(Path(a.vllm).expanduser(), Path(a.out), a.threads, a.compiler, blocks, only, slots)
    if a.orig:
        for o in compile_orig(Path(a.out), Path(a.vllm).expanduser(), only):
            print("orig", o)
    for e, i in rep.items():
        print(f"{e:22s} {i['block']} slots={i['slots']} args={i['arg_bytes']}B lds={i['lds_bytes']}B barriers={i['barriers']}")


if __name__ == "__main__":
    main()
