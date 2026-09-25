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
               f"vllm::packed_silu_kernel<typename vllm::PackedTypeConverter<{CB}>::Type>, true, true, false, false>", (1024, 1, 1)),
    ]),
]


def compile_ir(src_hip: Path, out_ll: Path, vllm: Path, s: Source, compiler: str) -> None:
    inc = []
    for i in s.includes:
        inc += ["-I", str(HERE / i)]
    inc += ["-I", str(vllm / "csrc"), "-I", str(vllm / Path(s.path).parent), "-I", str(vllm / "csrc/libtorch_stable")]
    defs = [f"-D{d}" for d in s.defines]
    if compiler == "local":
        cmd = ["bash", str(HERE / "hip_ir_local.sh"), str(src_hip), str(out_ll)] + defs + inc
    else:   # the VM: ROCm's own hipcc, device only, before the device libraries
        cmd = ["hipcc", "-x", "hip", "--cuda-device-only", "--offload-arch=gfx942", "-nogpulib", "-std=c++17", "-O3", "-S", "-emit-llvm",
               "-DUSE_ROCM", "-DNDEBUG", "-include", str(HERE / "shim/etx_cuda2hip.h"), "-I", str(HERE / "shim")] + defs + inc + [str(src_hip), "-o", str(out_ll)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"compile {s.path} failed:\n{r.stderr[-4000:]}")


def run(vllm: Path, out: Path, threads: int, compiler: str, blocks: dict[str, tuple[int, int, int]], only: set[str] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    llvm = Path(os.environ.get("ETX_LLVM", "/opt/homebrew/opt/llvm/bin"))
    llvm_as = str(llvm / "llvm-as") if compiler == "local" else "llvm-as"
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
            txt, info = import_kernel(mod, k.match, f"etx_imp_{k.export}", blk, threads, k.lds_cap, k.inline)
            (out / f"{k.export}.ll").write_text(txt)
            r = subprocess.run([llvm_as, str(out / f"{k.export}.ll"), "-o", str(out / f"{k.export}.bc")], capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"llvm-as {k.export}: {r.stderr[-2000:]}")
            report[k.export] = info.to_json() | {"source": s.path, "spec": k.spec, "note": k.note}
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
    a = ap.parse_args()
    blocks = {}
    for b in a.block:
        e, v = b.split("=")
        t = tuple(int(x) for x in v.split(","))
        blocks[e] = t + (1,) * (3 - len(t))
    rep = run(Path(a.vllm).expanduser(), Path(a.out), a.threads, a.compiler, blocks, set(a.only.split(",")) if a.only else None)
    for e, i in rep.items():
        print(f"{e:22s} {i['block']} slots={i['slots']} args={i['arg_bytes']}B lds={i['lds_bytes']}B barriers={i['barriers']}")


if __name__ == "__main__":
    main()
