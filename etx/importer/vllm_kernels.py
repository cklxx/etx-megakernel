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
from dataclasses import asdict
from pathlib import Path

from etx.importer.ir_import import import_kernel
from etx.importer.slice_device import slice_device

HERE = Path(__file__).resolve().parent
from etx.importer.spec import SOURCES, TRITON, TRITON_DIR, OPERATORS, Source

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
    env = os.environ.copy()
    if compiler == "local":
        # The final megakernel links device libraries; imports must retain
        # their original ID calls. Do not inherit a matrix's final-link setting.
        env.pop("ETX_DEVLIBS", None)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
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
            report[k.export] = info.to_json() | {"source": s.path, "spec": k.spec, "note": k.note, "operator_spec": asdict(OPERATORS[k.export])}
    from etx.importer.triton_import import import_triton
    from etx.importer.launch_spec import verify_capture
    for e, (kname, note) in TRITON.items():
        if only is not None and e not in only:
            continue
        capture_spec = verify_capture(e, TRITON_DIR)
        txt, info, meta, notes = import_triton(TRITON_DIR / e, kname, f"etx_imp_{e}", threads, (slots or {}).get(e))
        (out / f"{e}.ll").write_text(txt)
        r = subprocess.run([llvm_as, str(out / f"{e}.ll"), "-o", str(out / f"{e}.bc")], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"llvm-as {e}: {r.stderr[-2000:]}")
        report[e] = info.to_json() | {"source": f"triton:{(TRITON_DIR / e / 'CACHE_DIR').read_text().strip()}", "spec": kname,
                                      "note": note, "downgrade": notes, "capture_spec": capture_spec, "operator_spec": asdict(OPERATORS[e])}
    from etx.importer.spec import validate_import
    for export, info in report.items():
        validate_import(export, info)
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
