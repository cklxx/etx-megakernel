"""Build the kept control and full-dispatch C6 resource envelope, without a GPU.

Run in a native ROCm environment with --compiler rocm, or on macOS with local.
The envelope cannot execute. It has one deliberately unbound resource argument.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from . import paths
from .build import clean_env, sha, hybrid_host
from etx.models.declarations import declaration as _declaration
KEPT_ENV = _declaration(paths.ROOT, "vllm_llm.kept_build").KEPT_ENV
BASE_FLAGS = _declaration(paths.ROOT, "vllm_llm.kept_build").BASE_FLAGS
source_hashes = _declaration(paths.ROOT, "vllm_llm.kept_build").source_hashes
from etx.models.vllm_llm.bf16_300.build import resources


def prepare(out, compiler='local'):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    env=clean_env(); env.update(KEPT_ENV)
    local=compiler=='local'
    headers=Path(env.get('ETX_ROCM_HEADERS',str(Path.home()/'code/rocm-headers')))
    llvm=Path(env.get('ETX_LLVM','/opt/homebrew/opt/llvm/bin' if local else '/opt/rocm/llvm/bin'))
    libs=Path(env.get('ETX_DEVLIBS',str(headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')))
    if not local and 'ETX_DEVLIBS' not in env:
        candidates=list(Path('/opt').glob('rocm*/**/ocml.bc'))
        if candidates: libs=candidates[0].parent
    env.update(ETX_LLVM=str(llvm),ETX_ROCM_HEADERS=str(headers),ETX_DEVLIBS=str(libs))
    config=paths.ROOT/'examples/llm/configs/qwen3-30b-a3b.json'
    env['ETX_LLM_CONFIG']=str(config)
    commands=[]
    def run(cmd,name):
        cmd=list(map(str,cmd)); commands.append(cmd)
        print('STAGE',name,'evidence',out/(name+'.log'),flush=True)
        with (out/(name+'.log')).open('w') as log:
            rc=subprocess.run(cmd,cwd=paths.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        (out/(name+'.rc')).write_text(str(rc)+'\n')
        if rc: raise RuntimeError(f'{name} exit={rc}; see {out/(name+".log")}')
    code='from examples.vllm_llm.plan_exports import *; import json; d,n,p=exports('+repr(str(config))+'); print(json.dumps(build_settings(d,n,p)))'
    settings=json.loads(subprocess.check_output([sys.executable,'-c',code],cwd=paths.ROOT,env=env,text=True))
    env['ETX_VL_DSL_EXT']=settings['ETX_VL_DSL_EXT']
    cmd=[sys.executable,'-m','etx.importer.vllm_kernels','--vllm',env.get('ETX_VLLM',str(Path.home()/'code/vllm')),
         '--out',out/'imp','--only',settings['EXPORTS'],'--slots',settings['SLOTS'],*settings['BLK']]
    if not local: cmd+=['--compiler','hipcc']
    run(cmd,'import')
    run([sys.executable,'-m','etx.ops.emit',out/'imp/imports.json','--tiles',out/'dsl_tiles.hip','--ops',settings['DSLOPS']],'dsl')
    run([sys.executable,'-m','etx.importer.adapter',out/'imp/imports.json','--tiles',out/'imp_tiles.hip','--header',out/'imports.h'],'adapters')
    run([sys.executable,'-m','etx.importer.bundle','--imports',out/'imp','--exports',settings['EXPORTS'],'--devlibs',libs,'-o',out/'imports.bc'],'bundle')
    env.update(ETX_VL_IMPORTS=str(out/'imp/imports.json'),ETX_VL_ADAPTERS=str(out/'imp_tiles.hip'),ETX_VL_DSL_TILES=str(out/'dsl_tiles.hip'))
    reports={}
    for arm,module in [('control','examples/vllm_llm/model.py'),('envelope','examples/vllm_llm/attention_early_merge/footprint_model.py')]:
        dest=out/arm; dest.mkdir(); env.update(ETX_VL_PLAN=str(dest/'vl_plan.txt'),ETX_VL_CHAINS=str(dest/'chain_tiles.hip'),C6_FOOTPRINT_OUT=str(dest))
        run([sys.executable,'-m','etx','compile',module,'--arch','gfx942','--out',dest,'--inline-tiles'],arm+'_plan')
        common=['-I'+str(paths.ROOT/'etx/runtime/include'),'-I'+str(dest),'-I'+str(paths.ROOT),*BASE_FLAGS.split(),'-DETX_DSL_VECNORM','-DETX_DSL_NT','-DETX_TOPK_KEY32',
                '-Xclang','-mlink-builtin-bitcode','-Xclang',out/'imports.bc']
        if local:
            run(['bash',paths.ROOT/'etx/importer/hip_ir_local.sh',dest/'megakernel_d0.hip',dest/'mk.ll',*common],arm+'_ir')
        else:
            run(['hipcc','--cuda-device-only','-S','-emit-llvm','--offload-arch=gfx942','-O3','-std=c++17','--rocm-device-lib-path='+str(libs),
                 dest/'megakernel_d0.hip','-o',dest/'mk.ll',*common],arm+'_ir')
        run([llvm/'llc','-mcpu=gfx942','-mtriple=amdgcn-amd-amdhsa','-O3',dest/'mk.ll','-o',dest/'mk.s'],arm+'_isa')
        reports[arm]=resources((dest/'mk.s').read_text())
        if arm=='control':
            (dest/'hybrid_host.hip').write_text(hybrid_host())
            syntax=[llvm/'clang++','-x','hip','--cuda-host-only','-fsyntax-only','-nogpuinc','-nogpulib',
                    '--offload-arch=gfx942','-std=c++17','-isystem',headers/'clang','-isystem',headers/'inc',
                    '-include','__clang_hip_runtime_wrapper.h','-D__HIP_PLATFORM_AMD__','-DETX_VL_CHECK',
                    '-I'+str(paths.ROOT/'etx/runtime/include'),'-I'+str(dest),'-I'+str(out),'-I'+str(paths.ROOT)]
            if sys.platform=='darwin': syntax+=['-isystem',paths.ROOT/'etx/importer/shim/mac']
            run([*syntax,dest/'hybrid_host.hip'],'hybrid_host_syntax')
            run([x for x in syntax if x!='-fsyntax-only']+['-c',dest/'hybrid_host.hip','-o',dest/'hybrid_host.o'],'hybrid_host_object')
    record=dict(schema='etx.c6.offline.v1',compiler=compiler,resources=reports,commands=commands,
                sources=source_hashes(),hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.s','.hip','.bc','.json')},
                runnable=False,load='NOT_RUN',execution='NOT_RUN')
    (out/'offline.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--compiler',choices=['local','rocm'],default='local'); a=p.parse_args()
    print(json.dumps(prepare(a.out,a.compiler),indent=2))

if __name__=='__main__': main()
