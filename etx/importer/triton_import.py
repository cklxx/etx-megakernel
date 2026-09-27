"""Import a Triton kernel vLLM ran (from the Triton cache it wrote) into an ETX megakernel.

Input: `<cache>/<hash>/<name>.llir` (the LLVM IR Triton handed to its backend, after its own passes) and
`<name>.json` (num_warps, the dynamic LDS size in `shared`, the scratch sizes). The IR is Triton's LLVM (23 for the
ROCm 7.2 vLLM image); the megakernel is compiled with ROCm's clang (LLVM 22), so `downgrade_ir` first rewrites the few
constructs LLVM 22 cannot parse (each rewrite is exact and listed below). Then the kernel goes through the same
import as a HIP kernel (etx.importer.ir_import) with two Triton-specific steps:

  * dynamic LDS: Triton addresses all shared memory through `@global_smem = external addrspace(3) global [0 x i8]`,
    sized at launch; it becomes a static array of the json's `shared` bytes, which ir_import then moves into the arena
  * trailing parameters: Triton appends the global- and profile-scratch pointers to every kernel; they stay in the
    argument block (the host passes null: both scratch sizes are 0 for the kernels imported, checked here)

  python -m etx.importer.triton_import <cache>/<hash> --name fused_moe_kernel --export etx_imp_fused_moe_up \
      --threads 1024 [--slots 1] -o out.ll --info out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from etx.importer.ir_import import ImportError_, import_kernel

# IEEE layouts: (exponent bits, mantissa bits, LLVM 22 hex prefix)
_FP = {"half": (5, 10, "0xH"), "bfloat": (8, 7, "0xR"), "float": (8, 23, None), "double": (11, 52, None)}


def _fp_bits(ty: str, sign: str, kind: str, payload: int) -> str:
    """LLVM 23 special-value literal -> the LLVM 22 hex literal of the same bits."""
    e, m, pre = _FP[ty]
    exp = (1 << e) - 1
    if kind == "inf":
        mant = 0
    elif kind == "qnan":
        mant = (1 << (m - 1)) | payload
    elif kind == "nan":                         # quiet NaN with a payload (llvm-dis prints 0xR7FFF as +nan(0x3F))
        mant = (1 << (m - 1)) | payload
    elif kind == "snan":
        mant = payload
    else:
        raise ImportError_(f"float literal kind {kind}")
    bits = ((1 if sign == "-" else 0) << (e + m)) | (exp << m) | mant
    if pre:
        return f"{pre}{bits:0{(1 + e + m) // 4}X}"
    if ty == "float":                           # float constants are printed as the double of the same value
        dbits = ((1 if sign == "-" else 0) << 63) | (0x7FF << 52) | (mant << 29)
        return f"0x{dbits:016X}"
    return f"0x{bits:016X}"


# attributes LLVM 22 does not know; each only licenses optimisations, so dropping it keeps the semantics
_DROP_ATTRS = ("nocreateundeforpoison",)
_FPLIT = re.compile(r"\b(half|bfloat|float|double) ([+-])(inf|qnan|snan|nan)(?:\((0x[0-9A-Fa-f]+)\))?")


def strip_debug(text: str) -> str:
    """Debug info (LLVM 23 adds DIDerivedType fields LLVM 22 rejects) does not change code: drop it with an LLVM >= 23
    `opt` (ETX_LLVM23, default the local Homebrew LLVM), which reads the IR as written."""
    opt = Path(os.environ.get("ETX_LLVM23", "/opt/homebrew/opt/llvm/bin")) / "opt"
    if not opt.exists():
        raise ImportError_(f"{opt} not found: set ETX_LLVM23 to a bin directory of LLVM >= 23 (Triton's IR version)")
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.ll", Path(d) / "out.ll"
        src.write_text(text)
        r = subprocess.run([str(opt), "-S", "--strip-debug", str(src), "-o", str(dst)], capture_output=True, text=True)
        if r.returncode:
            raise ImportError_(f"opt --strip-debug: {r.stderr[-2000:]}")
        return dst.read_text()


def downgrade_ir(text: str) -> tuple[str, list[str]]:
    """Rewrite LLVM 23 textual IR for LLVM 22's parser. Returns the text and a list of what was rewritten."""
    done = []
    if "!dbg" in text:
        text = strip_debug(text)
        done.append("debug info stripped")
    n = 0

    def fp(m):
        nonlocal n
        n += 1
        return f"{m.group(1)} " + _fp_bits(m.group(1), m.group(2), m.group(3), int(m.group(4) or "0", 16))
    text = _FPLIT.sub(fp, text)
    if n:
        done.append(f"{n} special float literal(s) -> hex")
    for a in _DROP_ATTRS:
        k = len(re.findall(rf"\b{a}\b ?", text))
        if k:
            text = re.sub(rf"\b{a}\b ?", "", text)
            done.append(f"attribute {a} dropped ({k})")
    return text, done


def dynamic_lds_to_static(text: str, shared: int) -> str:
    """Triton's `@global_smem = external addrspace(3) global [0 x i8]` -> a static array of `shared` bytes."""
    pat = re.compile(r"@global_smem = external addrspace\(3\) global \[0 x i8\](, align (\d+))?")
    if not pat.search(text):
        if shared:
            raise ImportError_(f"json says {shared} B of shared memory but the IR has no @global_smem")
        return text
    return pat.sub(lambda m: f"@global_smem = internal addrspace(3) global [{shared} x i8] undef, align {m.group(2) or 16}", text)


def import_triton(cache_dir: Path, name: str, export: str, threads: int, max_slots: int | None = None):
    meta = json.loads((cache_dir / f"{name}.json").read_text())
    for k in ("global_scratch_size", "profile_scratch_size"):
        if meta.get(k, 0):
            raise ImportError_(f"{name}: {k} = {meta[k]} (the importer passes null scratch pointers)")
    text, notes = downgrade_ir((cache_dir / f"{name}.llir").read_text())
    text = dynamic_lds_to_static(text, int(meta["shared"]))
    block = (64 * int(meta["num_warps"]), 1, 1)
    out, info = import_kernel(text, name, export, block, threads, None, True, max_slots)
    return out, info, meta, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--threads", type=int, default=1024)
    ap.add_argument("--slots", type=int, default=None)
    ap.add_argument("-o", required=True)
    ap.add_argument("--info", required=True)
    a = ap.parse_args()
    out, info, meta, notes = import_triton(Path(a.cache_dir), a.name, a.export, a.threads, a.slots)
    Path(a.o).write_text(out)
    Path(a.info).write_text(json.dumps(info.to_json() | {"triton": {k: meta[k] for k in ("num_warps", "shared", "hash")}, "downgrade": notes}, indent=1))
    print(f"{a.export}: block {info.block} slots={info.slots} args={info.arg_bytes}B lds={info.lds_bytes}B; {'; '.join(notes) or 'no rewrites'}")


if __name__ == "__main__":
    main()
