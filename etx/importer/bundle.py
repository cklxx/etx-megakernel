"""Link a set of imported kernels into one bitcode file for the megakernel compile.

clang links the device libraries before the -mlink-builtin-bitcode files given on the command line, with
"only needed" semantics, so math functions an imported kernel calls (e.g. __ocml_rsqrt_f32 in vLLM's
rms_norm) would stay unresolved. This links the imports together and then pulls in exactly the library
functions they need, with the same oclc control libraries clang uses for gfx942 at default precision
(unsafe math off, finite-only off, wave64, ISA 9.4.2, ABI 600).

  python -m etx.importer.bundle --imports DIR --exports a,b,c --devlibs <amdgcn/bitcode dir> -o imports.bc
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

OCLC = ["oclc_unsafe_math_off.bc", "oclc_finite_only_off.bc", "oclc_wavefrontsize64_on.bc", "oclc_isa_version_942.bc",
        "oclc_abi_version_600.bc"]


def llvm_tool(name: str) -> str:
    for d in (os.environ.get("ETX_LLVM"), "/opt/rocm/llvm/bin", "/opt/rocm/lib/llvm/bin", "/opt/homebrew/opt/llvm/bin"):
        if d and (Path(d) / name).exists():
            return str(Path(d) / name)
    return name


def bundle(imp_dir: Path, exports: list[str], devlibs: Path, out: Path) -> None:
    link = llvm_tool("llvm-link")
    tmp = out.with_suffix(".imports.bc")
    r = subprocess.run([link] + [str(imp_dir / f"{e}.bc") for e in exports] + ["-o", str(tmp)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"llvm-link imports: {r.stderr[-2000:]}")
    libs = [str(devlibs / "ocml.bc"), str(devlibs / "ockl.bc")] + [str(devlibs / f) for f in OCLC]
    linked = out.with_suffix(".linked.bc")
    r = subprocess.run([link, str(tmp), "--only-needed"] + libs + ["-o", str(linked)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"llvm-link device libraries: {r.stderr[-2000:]}")
    # only the entry points stay visible: the library functions become internal to the bundle, so clang's
    # "only needed" builtin linking takes them along with their callers (a linkonce_odr copy was dropped)
    keep = ",".join(f"etx_imp_{e}" for e in exports) + ",etx_lds_arena,etx_imp_slot_barrier"
    r = subprocess.run([llvm_tool("opt"), "-passes=internalize,globaldce", f"-internalize-public-api-list={keep}", str(linked), "-o", str(out)],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"opt internalize: {r.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imports", required=True); ap.add_argument("--exports", required=True)
    ap.add_argument("--devlibs", required=True); ap.add_argument("-o", required=True)
    a = ap.parse_args()
    bundle(Path(a.imports), a.exports.split(","), Path(a.devlibs), Path(a.o))


if __name__ == "__main__":
    main()
