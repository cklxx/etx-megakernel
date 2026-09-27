"""Build-time patches for a COPY of fleet-mi300x's sources (the fleet repository itself is never modified).

  python -m examples.dsv2lite.patch_fleet <fleet src dir copy> <patch> [<patch> ...]

gemv_noinline: fleet's gemv_rows dispatches on K at run time between five GEMV paths, so every call site instantiates
  all five and the register allocator sizes the union: 262 registers per lane (VGPR + AGPR) where each path alone needs
  74-118 (measured with the local gfx942 compile, 2026-09-27). Two workgroups per CU (the two-request interleave) need
  <= 256, so the paths become separate (noinline) functions: each call pays one function call, and the megakernel's
  heaviest tile becomes attention (240).
"""
from __future__ import annotations

import sys
from pathlib import Path

PATCHES = {
    "gemv_noinline": ("kernels/gemv.h", [
        ("template <int RPI, int D>\n__device__ __forceinline__ void gemv_rows_pipelined(",
         "template <int RPI, int D>\n__device__ __noinline__ void gemv_rows_pipelined(   // etx patch gemv_noinline"),
        ("template <int RPI>   // rows per wave per group (even); each half-wave takes RPI/2\n__device__ __forceinline__ void gemv_rows_pipelined_half(",
         "template <int RPI>   // rows per wave per group (even); each half-wave takes RPI/2\n__device__ __noinline__ void gemv_rows_pipelined_half(   // etx patch"),
        ("template <int RPI, int D>\n__device__ __forceinline__ void gemv_rows_impl(",
         "template <int RPI, int D>\n__device__ __noinline__ void gemv_rows_impl(   // etx patch gemv_noinline"),
    ]),
}


def apply(src: Path, name: str) -> None:
    rel, subs = PATCHES[name]
    f = src / rel
    text = f.read_text()
    for old, new in subs:
        if text.count(old) != 1:
            raise SystemExit(f"patch {name}: expected exactly one match in {f} for:\n{old}")
        text = text.replace(old, new)
    f.write_text(text)
    print(f"patched {f} ({name}, {len(subs)} edits)")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    for name in sys.argv[2:]:
        apply(src, name)


if __name__ == "__main__":
    main()
