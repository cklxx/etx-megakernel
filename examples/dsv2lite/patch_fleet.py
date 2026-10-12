"""Build-time patches for a COPY of fleet-mi300x's sources (the fleet repository itself is never modified).

  python -m examples.dsv2lite.patch_fleet <fleet src dir copy> <patch> [<patch> ...]

gemv_noinline: fleet's gemv_rows dispatches on K at run time between five GEMV paths, so every call site instantiates
  all five and the register allocator sizes the union: 262 registers per lane (VGPR + AGPR) where each path alone needs
  74-118 (measured with the local gfx942 compile, 2026-09-27). Two workgroups per CU (the two-request interleave) need
  <= 256, so the paths become separate (noinline) functions: each call pays one function call, and the megakernel's
  heaviest tile becomes attention (240).

Every other patch is a lever: one module examples/dsv2lite/levers/<name>.py with `apply(src: Path)`, each on its own
branch. A lever edits the copy with _once / _function (exact, single-match edits) and keeps every original helper
byte for byte, so a default build is unchanged and its tests can diff against the pinned source.
"""
from __future__ import annotations

import importlib
import pkgutil
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


def _once(text: str, old: str, new: str, name: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"patch {name}: expected exactly one match for:\n{old}")
    return text.replace(old, new)


def _function(text: str, signature: str) -> str:
    """Extract a pinned C++ helper, including its balanced macro bodies."""
    if text.count(signature) != 1:
        raise SystemExit(f"expected one helper signature: {signature}")
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise SystemExit(f"unbalanced helper: {signature}")
    return text[start:end]


def levers() -> list[str]:
    """The lever patches present on this branch."""
    from examples.dsv2lite import levers as pkg
    return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__) if not m.name.startswith("_"))


def apply(src: Path, name: str) -> None:
    if name in PATCHES:
        rel, subs = PATCHES[name]
        f = src / rel
        text = f.read_text()
        for old, new in subs:
            text = _once(text, old, new, name)
        f.write_text(text)
        print(f"patched {f} ({name}, {len(subs)} edits)")
        return
    if name not in levers():
        raise SystemExit(f"unknown patch {name!r}; levers on this branch: {levers()}")
    importlib.import_module(f"examples.dsv2lite.levers.{name}").apply(src)
    print(f"patched {src} ({name})")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    for name in sys.argv[2:]:
        apply(src, name)


if __name__ == "__main__":
    main()
