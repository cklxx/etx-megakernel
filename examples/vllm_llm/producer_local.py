"""Run the complete CPU acceptance. Never invokes a GPU or a cloud API.

Outputs go into a new evidence directory. Build outputs stay under build/.
A failing required step prevents local-ready.json. No step is silently skipped.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from examples.vllm_llm import producer_session as ps
from examples.vllm_llm.kept_build import BASE_FLAGS,KEPT_ENV

ROOT=Path(__file__).resolve().parents[2]



def rebase_includes(source, destination):
    """Map only container include paths to the same files in the host mount.

    Keep the original ROCm source. Never change instructions or emit a stub.
    """
    text=Path(source).read_text();mapping={}
    def mapped(match):
        path=Path.home()/'code'/match[1]
        if not path.is_file():raise ValueError('missing host include: '+str(path))
        mapping[match[0]]=str(path)
        return '#include "'+str(path)+'"'
    native=re.sub(r'^#include "/code/([^"\n]+)"$',mapped,text,flags=re.M)
    if not mapping:raise ValueError('no container includes to map')
    with Path(destination).open('x') as f:f.write(native)
    return mapping


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    sources=ps.local_sources()
    results=[]
    receipt_lock=Lock()
    def run(name,cmd,timeout=1200,env=None):
        if sources!=ps.local_sources():raise RuntimeError('source changed during acceptance; rerun required')
        log=a.out/f'{name}.log';start=time.monotonic()
        print(f'LOCAL phase={name} evidence={log}',flush=True)
        with log.open('x') as f:
            f.write('# command '+json.dumps(list(map(str,cmd)))+'\n');f.flush()
            try:
                r=subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
                rc=r.returncode
            except subprocess.TimeoutExpired:rc=124
        with receipt_lock:
            results.append(dict(phase=name,command=list(map(str,cmd)),exit=rc,seconds=time.monotonic()-start,log=str(log)))
            (a.out/'commands.json').write_text(json.dumps(results,indent=2)+'\n')
        print(f'LOCAL phase={name} exit={rc}',flush=True)
        if rc:raise RuntimeError(f'{name} failed, exit {rc}: inspect {log}')
    clean=ps.clean_env();clean.pop('ETX_SESS_LOGDIR',None)
    r=Path.home()/'code/rocm-headers'
    llvm=Path('/opt/homebrew/opt/llvm/bin')
    devlibs=r/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode'
    env=dict(clean,ETX_PYTHON=sys.executable,ETX_ROCM_HEADERS=str(r),ETX_DEVLIBS=str(devlibs),ETX_LLVM=str(llvm))
    for script in ('examples/vllm_llm/build.sh','examples/vllm_llm/local_matrix.sh','examples/vllm_llm/rocm_build_check.sh','tools/session.sh'):
        run('shell-'+Path(script).stem,['bash','-n',script],env=env)
    run('focused',[sys.executable,'-m','pytest','-q','tests/test_producer.py','tests/test_host_import_check.py','tests/test_router_study.py','tests/test_gpu_jobs.py','tests/test_session_references.py','tests/test_test_paths.py'],env=env)
    def builds_and_simulation():
        run('simulation',[sys.executable,'-u','-m','examples.vllm_llm.producer_replay','--out',a.out/'simulation.json'],env=env)
        run('session-dry',[sys.executable,'-m','etx.tools.gpu_jobs','examples/sessions/qwen_producer.py','--dry-run'],env=env)
        run('matrix',['bash','examples/vllm_llm/local_matrix.sh',ROOT/f'build/producer-{a.out.name}-matrix'],timeout=1800,env=env)
        for arm in ps.ARMS:
            b=ROOT/f'build/producer-{a.out.name}-{arm}'
            overrides=dict(KEPT_ENV,ETX_VL_PRODUCER_SCORE=str(int(arm=='producer')),
                           ETX_CFLAGS=BASE_FLAGS+' -DETX_DSL_VECNORM -DETX_TOPK_KEY32')
            run(f'rocm-{arm}',['bash','examples/vllm_llm/rocm_build_check.sh','qwen3-30b-a3b','default',b],env=dict(env,**overrides))
            proof=ps.audit_plan(b,arm=='producer');(a.out/f'plan-{arm}.json').write_text(json.dumps(proof,indent=2)+'\n')
            mapping=rebase_includes(b/'megakernel_d0.hip',b/'native_megakernel.hip')
            (a.out/f'include-map-{arm}.json').write_text(json.dumps(mapping,indent=2)+'\n')
            flags=(BASE_FLAGS+' -DETX_DSL_VECNORM -DETX_TOPK_KEY32').split()
            run(f'llvm-ir-{arm}',['bash','etx/importer/hip_ir_local.sh',b/'native_megakernel.hip',b/'local.ll',
                                 '-Ietx/runtime/include','-I',b,'-I.',*flags,
                                 '-Xclang','-mlink-builtin-bitcode','-Xclang',b/'imports.bc'],env=env)
            run(f'llvm-asm-{arm}',[llvm/'llc','-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',b/'local.ll','-o',b/'local.s'],env=env)
            host=[llvm/'clang++','-x','hip','--cuda-host-only','-nogpuinc','-nogpulib','--offload-arch=gfx942','-std=c++17',
                  '-isystem',r/'clang','-isystem',r/'inc','-include','__clang_hip_runtime_wrapper.h',
                  '-isystem','etx/importer/shim/mac','-D__HIP_PLATFORM_AMD__','-DETX_VL_CHECK',
                  '-Ietx/runtime/include','-I',b,'-I.',*flags]
            run(f'host-object-{arm}',[*host,'-c','examples/vllm_llm/host.hip','-o',b/'host-local.o'],env=env)
            for name in ('resource.log','host_dev.log','compile.log','vl_plan.txt','plan.json'):
                shutil.copyfile(b/name,a.out/f'{arm}-{name}')
    # The full suite and compiler outputs have separate evidence/build paths.
    # Wait for both branches, including on failure. Never leave a check running.
    with ThreadPoolExecutor(max_workers=2) as pool:
        checks=[pool.submit(lambda: run('full-suite',[sys.executable,'-m','pytest','-q','-ra'],timeout=1800,env=env)), pool.submit(builds_and_simulation)]
        for check in checks:check.result()
    if sources!=ps.local_sources():raise RuntimeError('source changed during acceptance; rerun required')
    ready=dict(ready=True,sources=sources,commands=results,
               limitations=['GPU executable link/load/execution pending: macOS has no Linux HIP runtime',
                            'real-model capture/replay pending; no retained Qwen router tensors found',
                            'GPU performance and memory visibility unknown'],gpu_spend_usd=0)
    with (ps.ART/'local-ready.json').open('x') as f:json.dump(ready,f,indent=2);f.write('\n')
    print('LOCAL COMPLETE; GPU not run; local-ready.json written',flush=True)

if __name__=='__main__':main()
