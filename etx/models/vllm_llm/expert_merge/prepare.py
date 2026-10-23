"""Generate, compile and identify the isolated probe without touching model files."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from . import paths
FLAGS = ("-DETX_DSL_WSUM", "-DETX_DSL_SCALAR", "-DETX_DSL_GU1", "-DETX_DSL_RBATCH",
         "-DETX_DSL_NORM_HOIST", "-DETX_DSL_VECNORM", "-DETX_TOPK_KEY32")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def sources():
    files = list((paths.ROOT/"etx").rglob("*.py")) + list((paths.ROOT/"etx/runtime/include").rglob("*.h"))
    files += list(paths.HERE.glob("*.py")) + list(paths.HERE.glob("*.hip")) + list(paths.HERE.glob("*.h")) + list(paths.HERE.glob("*.sh"))
    files += [paths.ROOT/"etx/machine/arch/gfx942.yaml", paths.ROOT/"examples/vllm_llm/kept_build.py"]
    files += [paths.ROOT/"examples/sessions/qwen_expert_merge.py"]
    return {str(p.relative_to(paths.ROOT)): sha(p) for p in sorted(set(files))}


def clean_env():
    return {k:v for k,v in os.environ.items() if not k.startswith(("ETX_", "DSL_ABLATE_", "EEM_"))
            and k not in ("CFLAGS", "CXXFLAGS")}


def call(cmd, env=None):
    print("STAGE", shlex.join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)), cwd=paths.ROOT, env=env, check=True)


def generate(out, arm, diagnostic=False, head=False):
    if arm not in ("control", "candidate"):
        raise ValueError("unknown expert merge arm")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"identity.json").exists():
        raise ValueError("build exists; choose a new directory to retain evidence")
    identity = dict(schema=1, arm=arm, diagnostic=diagnostic, head=head, M=151936 if head else 5120,
                    flags=list(FLAGS), sources=sources(), imported_launches=0,
                    full_model_default="OFF", input_contract="selected-slot bf16; upstream top-k fixed")
    identity["id"] = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    config = f'''#pragma once
#define EEM_CANDIDATE {int(arm == 'candidate')}
#define EEM_DIAGNOSTIC {int(diagnostic)}
#define EEM_HEAD {int(head)}
#define EEM_M {identity['M']}
#define EEM_IDENTITY "{identity['id']}"
'''
    (out/"probe_config.h").write_text(config)
    env = clean_env()
    # Import the emitter in a clean process too. Its module constants read env.
    source = subprocess.check_output([sys.executable, "-c",
        "from etx.models.vllm_llm.expert_merge.source import emit; "
        f"print(emit({arm == 'candidate'}, {diagnostic}), end='')"], cwd=paths.ROOT, env=env, text=True)
    (out/"tiles.hip").write_text(source)
    env.update(EEM_ARM=arm,EEM_TILES=str(out/"tiles.hip"))
    call([sys.executable,"-m","etx","compile",paths.HERE/"graph.py","--arch","gfx942","--out",out,"--inline-tiles","--sim"],env)
    for file in ("plan.json","plan_data.h","megakernel_d0.hip","tiles.hip","probe_config.h"):
        identity.setdefault("generated",{})[file] = sha(out/file)
    (out/"identity.json").write_text(json.dumps(identity,indent=2,sort_keys=True)+"\n")
    print("GENERATED",identity["id"],out,flush=True)
    return identity


def verify(out):
    out = Path(out)
    identity = json.loads((out/"identity.json").read_text())
    frozen = {k:v for k,v in identity.items() if k not in ("id", "generated")}
    if hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest() != identity["id"]:
        raise ValueError("stale build manifest identity")
    if identity["sources"] != sources():
        raise ValueError("stale source identity; rebuild in a new directory")
    for file, digest in identity["generated"].items():
        if sha(out/file) != digest:
            raise ValueError("stale generated identity: " + file)
    return identity


def compile_local(out):
    out=Path(out).resolve(); verify(out)
    rocm=Path(os.environ.get("ETX_ROCM_HEADERS",str(Path.home()/"code/rocm-headers")))
    llvm=Path(os.environ.get("ETX_LLVM","/opt/homebrew/opt/llvm/bin"))
    libs=rocm/"devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode"
    env=clean_env(); env.update(ETX_DEVLIBS=str(libs),ETX_ROCM_HEADERS=str(rocm),ETX_LLVM=str(llvm))
    inc=["-Ietx/runtime/include","-I",str(out),"-I.",*FLAGS]
    call(["bash","etx/importer/hip_ir_local.sh",out/"megakernel_d0.hip",out/"mk.ll",*inc],env)
    call([llvm/"llc","-mtriple=amdgcn-amd-amdhsa","-mcpu=gfx942","-O3",out/"mk.ll","-o",out/"mk.s"],env)
    call([llvm/"clang++","-x","hip","--cuda-host-only","-fsyntax-only","-nogpuinc","-nogpulib",
          "--offload-arch=gfx942","-std=c++17","-isystem",rocm/"clang","-isystem",rocm/"inc",
          "-include","__clang_hip_runtime_wrapper.h","-isystem","etx/importer/shim/mac",
          "-D__HIP_PLATFORM_AMD__",*inc,paths.HERE/"host.hip"],env)
    print("LOCAL compile + assembly + host syntax OK; no link/load/execute",flush=True)


def compile_native(out):
    out=Path(out).resolve(); identity=verify(out)
    cmd=["hipcc","-O3","-std=c++17","--offload-arch=gfx942","-Ietx/runtime/include","-I",str(out),"-I.",
         *FLAGS,"-Rpass-analysis=kernel-resource-usage",out/"megakernel_d0.hip",paths.HERE/"host.hip","-o",out/"run"]
    call(cmd,clean_env())
    identity["binary"] = sha(out/"run")
    identity["command"] = list(map(str,cmd))
    identity["compiler"] = subprocess.check_output(["hipcc","--version"],text=True)
    (out/"build.json").write_text(json.dumps(identity,indent=2,sort_keys=True)+"\n")
    print("NATIVE compile/link OK; load/execute pending",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("generate","local","native","verify"))
    p.add_argument("out",type=Path)
    p.add_argument("--arm",choices=("control","candidate"),default="control")
    p.add_argument("--diagnostic",action="store_true"); p.add_argument("--head",action="store_true")
    a=p.parse_args()
    if a.action=="generate": generate(a.out,a.arm,a.diagnostic,a.head)
    elif a.action=="local": compile_local(a.out)
    elif a.action=="native": compile_native(a.out)
    else: print(json.dumps(verify(a.out),sort_keys=True))


if __name__=="__main__": main()


def __getattr__(name):
    # Historical module attributes (prepare.ROOT / prepare.HERE) follow the configured repository.
    if name in ("ROOT", "HERE"):
        return getattr(paths, name)
    raise AttributeError(name)
