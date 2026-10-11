#!/usr/bin/env python3
"""Static ISA census of the six DSL tile bodies of the kept KEY32-on build (CPU only).

  .venv/bin/python docs/master-table-2026-10-11/isa.py > docs/master-table-2026-10-11/isa.txt

Compiles build/qwen300-rocm-on/dsl_tiles.hip (sha256 checked against the certificate's) inside one wrapper
kernel per tile with the local LLVM 23 and the ROCm 7.2.4 headers, the build's flags and -gline-tables-only,
then counts instructions per class and per dsl_tiles.hip source line from the .loc directives. The local
compiler is not the VM's (CLAUDE.md section 4): counts are an estimate [E] of the timed ISA, not the timed ISA.
"""
from __future__ import annotations

import collections
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "build/qwen300-rocm-on"
SRC = B / "dsl_tiles.hip"
DEVLIBS = Path.home() / "code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode"
FLAGS = ["-DETX_DSL_COH", "-DETX_DSL_NT", "-DETX_DSL_VECNORM", "-DETX_DSL_WSUM", "-DETX_DSL_SCALAR",
         "-DETX_DSL_GU1", "-DETX_DSL_RBATCH", "-DETX_TOPK_KEY32"]
TILES = [("k_gemv_fold", "etx_tile_dsl_gemv_fold"), ("k_gemv", "etx_tile_dsl_gemv"),
         ("k_attnprep", "etx_tile_dsl_attnprep"), ("k_topk_rep", "etx_tile_rep_dsl_topk"),
         ("k_moe_up_x_fold", "etx_tile_x_dsl_moe_up_fold"), ("k_moe_down_x_fold", "etx_tile_x_dsl_moe_down_fold")]
WRAP = """#define ETX_IMPORT_LDS 21120
#define ETX_IMPORT_SLOTS 8
#include "etx_lowering.h"
#include "etx/abi.h"
#include "etx/primitives.h"
#include "dsl_tiles.hip"
#define K(name, sym) extern "C" __global__ __launch_bounds__(1024,1) void name(const etx_ctx* c) { sym(c); \
ETX_RELEASE_DOMAIN(); __syncthreads(); }
""" + "".join(f"K({k}, {s})\n" for k, s in TILES)
CLASSES = [("mfma", r"^v_mfma"), ("vmem_load", r"^(global|buffer|flat)_load"),
           ("vmem_store", r"^(global|buffer|flat)_store"), ("vmem_atomic", r"^(global|buffer|flat)_atomic"),
           ("smem", r"^s_(load|buffer_load|store)"), ("lds", r"^ds_"), ("waitcnt", r"^s_waitcnt"),
           ("barrier", r"^s_barrier"), ("cache_ctl", r"^(buffer_wbl2|buffer_inv|s_dcache)"),
           ("branch", r"^s_(cbranch|branch|setpc|swappc)"), ("dpp", r"_dpp\b|row_|quad_perm|row_bcast"),
           ("valu", r"^v_"), ("salu", r"^s_")]


def compile_s(work: Path) -> Path:
    llvm = os.environ.get("ETX_LLVM", "/opt/homebrew/opt/llvm/bin")
    (work / "wrap.hip").write_text(WRAP)
    ll, s = work / "wrap.ll", work / "wrap.s"
    inc = ["-I", str(B), "-I", str(ROOT / "etx/runtime/include"), "-I", str(ROOT / "etx/runtime")]
    env = dict(os.environ, ETX_DEVLIBS=os.environ.get("ETX_DEVLIBS", str(DEVLIBS)))
    subprocess.run(["bash", str(ROOT / "etx/importer/hip_ir_local.sh"), str(work / "wrap.hip"), str(ll),
                    "-gline-tables-only", *FLAGS, *inc], check=True, cwd=ROOT, env=env)
    subprocess.run([f"{llvm}/llc", "-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx942", "-O3", str(ll), "-o", str(s)],
                   check=True)
    return s


def classify(op: str, line: str) -> str:
    for name, pat in CLASSES:
        if re.search(pat, op if name != "dpp" else line):
            if name == "dpp" and not op.startswith("v_"):
                continue
            return name
    return "other"


def main() -> None:
    print(f"dsl_tiles.hip sha256 {hashlib.sha256(SRC.read_bytes()).hexdigest()}")
    print(f"flags {' '.join(FLAGS)}; local LLVM 23 + ROCm 7.2.4 headers; counts are static [E]\n")
    cache: dict[str, list[str]] = {}

    def text(path: str, no: int) -> str:
        if not no:
            return "<line 0: compiler-generated, no source line>"
        if path not in cache:
            cache[path] = Path(path).read_text().splitlines() if Path(path).exists() else []
        src = cache[path]
        return src[no - 1].strip()[:84] if no <= len(src) else "?"
    with tempfile.TemporaryDirectory() as td:
        s = compile_s(Path(td))
        asm = s.read_text().splitlines()
    files: dict[str, str] = {}
    func, cur, kname = None, None, ""
    per = collections.defaultdict(collections.Counter)
    lines = collections.defaultdict(collections.Counter)
    meta: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for ln in asm:
        m = re.match(r'\s*\.file\s+(\d+)\s+"([^"]*)"\s+"([^"]+)"', ln)
        if m:
            d = Path(m.group(2))
            files[m.group(1)] = str((d if d.is_absolute() else ROOT / d) / m.group(3))
            continue
        m = re.match(r"^(k_\w+):", ln)
        if m:
            func, cur = m.group(1), None
            continue
        m = re.match(r"\s*\.loc\s+(\d+)\s+(\d+)", ln)
        if m and func:
            cur = (files.get(m.group(1), "?"), int(m.group(2)))
            continue
        m = re.match(r"\s+\.amdhsa_kernel\s+(k_\w+)", ln)
        if m:
            kname = m.group(1)
        m = re.match(r"; (NumVgprs|NumSgprs|ScratchSize|LDSByteSize|Occupancy): (\d+)", ln)
        if m:
            meta[kname][m.group(1)] = m.group(2)
        m2 = re.match(r"^\t([sv]_\w+|global_\w+|buffer_\w+|flat_\w+|ds_\w+)\b", ln)
        if func and m2:
            op = m2.group(1)
            c = classify(op, ln)
            per[func][c] += 1
            per[func]["total"] += 1
            if cur:
                lines[func][cur] += 1
        if ln.startswith("\t.size\tk_"):
            func = None
    cols = ["total", "valu", "salu", "mfma", "dpp", "vmem_load", "vmem_store", "vmem_atomic", "smem", "lds",
            "waitcnt", "barrier", "cache_ctl", "branch"]
    print("STATIC INSTRUCTION CLASSES per tile wrapper (includes the inlined ETX_RELEASE_DOMAIN + final barrier)")
    print(f"{'kernel':20s} " + " ".join(f"{c[:8]:>8s}" for c in cols))
    for k, _ in TILES:
        print(f"{k:20s} " + " ".join(f"{per[k][c]:8d}" for c in cols))
    print("\nRESOURCES (local compiler)")
    for k, _ in TILES:
        print(f"{k:20s} " + " ".join(f"{a}={b}" for a, b in meta[k].items()))
    print("\nTOP SOURCE LINES per tile by static instruction count (file:line count source); static, not executed")
    for k, _ in TILES:
        byfile = collections.Counter()
        for (f, _), n in lines[k].items():
            byfile[Path(f).name] += n
        print(f"-- {k}: by file " + ", ".join(f"{f} {n}" for f, n in byfile.most_common(6)))
        for (f, no), n in sorted(lines[k].items(), key=lambda kv: -kv[1])[:16]:
            print(f"  {Path(f).name + ':' + str(no):>22s} {n:4d}  {text(f, no)}")


if __name__ == "__main__":
    main()
