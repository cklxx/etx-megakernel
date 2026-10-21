"""Compile current whole-model closures offline, without copying baseline trees.

The system HIP runtime library is not present in the preparation image. Device
modules and host objects are real compiler outputs. The ordinary HIP host link
fails closed (exit77) unless an existing local runtime is supplied. This does
not issue an executable/model/fusion certificate.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .protocol import require,sha
from .build_support import ROOT,HOME_CODE,DEPS,cpath,dependencies


def run(cmd,out,name,*,env=None,seconds=180):
    print('LOCAL MODEL BUILD',name,flush=True)
    with (out/(name+'.log')).open('w') as f:
        r=subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=seconds)
    (out/(name+'.exit')).write_text(str(r.returncode)+'\n')
    require(r.returncode==0,f'{name} failed; full log {out/(name+".log")}')


def prepare_qwen(out,reference,*,last_router=False,overlay=None,imports_override=None,host_override=None):
    from examples.vllm_llm.kept_build import KEPT_ENV,BASE_FLAGS
    out=Path(out).resolve();reference=Path(reference).resolve()
    require(shutil.disk_usage(out.parent).free>=2*(1<<30),'storage free floor2GiB')
    out.mkdir(parents=True,exist_ok=False);(out/'imp').mkdir();(out/'tmp').mkdir()
    original=json.loads((reference/'imp/imports.json').read_text())
    (out/'imp/imports.json').write_text(json.dumps(original,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('ETX_','DSL_ABLATE_','BF16_'))}
    env.update(KEPT_ENV,ETX_VL_LAST_ROUTER=str(int(last_router)),ETX_VL_IMPORTS=str(out/'imp/imports.json'),
               ETX_VL_ADAPTERS=str(reference/'imp_tiles.hip'),ETX_VL_DSL_TILES=str(out/'dsl_tiles.hip'),
               ETX_VL_CHAINS=str(out/'chain_tiles.hip'),ETX_VL_PLAN=str(out/'vl_plan.txt'),
               ETX_LLM_CONFIG=str(ROOT/'examples/llm/configs/qwen3-30b-a3b.json'),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'tmp'))
    run([sys.executable,'-m','etx.ops.emit',out/'imp/imports.json','--tiles',out/'dsl_tiles.hip','--ops','gemv,topk,attnprep'],out,'emit-tiles',env=env)
    if overlay:
        # A winner helper supplies a complete exact DSL TU with one mechanism;
        # never mix operator-only source into the model accidentally.
        require(not last_router,'one admitted mechanism per model arm')
        (out/'dsl_tiles.hip').write_bytes(Path(overlay).read_bytes())
    run([sys.executable,'-m','etx','compile','examples/vllm_llm/model.py','--arch','gfx942','--out',out,'--inline-tiles'],out,'plan',env=env,seconds=180)
    if last_router:
        from .router.model_contract import validate
        (out/'model-contract.json').write_text(json.dumps(validate(json.loads((out/'plan.json').read_text())),indent=2)+'\n')
    flags=BASE_FLAGS.split()+['-DETX_DSL_NT','-DETX_DSL_VECNORM','-DETX_TOPK_KEY32','-DETX_VL_CHECK']
    return compile_closure(out,reference,Path(host_override) if host_override else ROOT/'examples/vllm_llm/host.hip',flags,imports=Path(imports_override) if imports_override else reference/'imports.bc',env_record={k:v for k,v in env.items() if k.startswith('ETX_')})


def prepare_routes(out,batch):
    from examples.dsv2lite.arms import ARMS,build_env
    from examples.dsv2lite.pinned import fleet_export
    out=Path(out).resolve();require(batch in (2,4),'G04 B2/B4 only')
    require(shutil.disk_usage(out.parent).free>=2*(1<<30),'storage free floor2GiB')
    out.mkdir(parents=True,exist_ok=False);(out/'tmp').mkdir()
    e=build_env(ARMS['x'+str(batch)])
    env={k:v for k,v in os.environ.items() if not k.startswith(('ETX_','FLEET_'))};env.update(e,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'tmp'))
    fleet=fleet_export()
    # Generate only a private source export for already registered patches; it
    # is ~1MiB and never alters the shared read-only Fleet reference.
    shutil.copytree(fleet/'src',out/'fleet_src')
    levers=e['ETX_FLEET_PATCH'].split()
    run([sys.executable,'-m','examples.dsv2lite.patch_fleet',out/'fleet_src',*levers],out,'registered-patches',env=env)
    run([sys.executable,'-m','etx','compile','examples/dsv2lite/graph.py','--arch','gfx942','--out',out,'--inline-tiles','--relay','on'],out,'plan',env=env)
    flags=e['ETX_CFLAGS'].split()+['-DETX_PROGRAM_ROUTE_CAPTURE=1','-DFLEET_NT_WEIGHTS=1','-I'+cpath(out/'fleet_src')]
    return compile_closure(out,out,ROOT/'examples/dsv2lite/host.hip',flags,env_record=e)


def compile_closure(out,reference,host,flags,*,imports=None,env_record=None):
    tool='/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/'
    docker=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64',
            '-v',str(HOME_CODE)+':/code','-v',str(HOME_CODE)+':'+str(HOME_CODE)+':ro','-v',str(DEPS)+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu',
            '-e','TMPDIR='+cpath(out/'tmp'),'-w',cpath(ROOT),'etx-rocm22']
    common=['-x','hip','--offload-arch=gfx942','-O3','-std=c++17','-nogpuinc',
        '--rocm-device-lib-path=/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode',
        '-isystem','/code/rocm-headers/inc','-include','__clang_hip_runtime_wrapper.h','-D__HIP_PLATFORM_AMD__',
        '-I'+cpath(ROOT/'etx/runtime/include'),'-I'+cpath(out),'-I'+cpath(reference),'-I'+cpath(ROOT),*flags]
    dev_extra=['-Xclang','-mlink-builtin-bitcode','-Xclang',cpath(imports)] if imports else []
    commands=[]
    def native(name,args):
        if args[0].endswith('clang++') and ('-c' in args or '-S' in args):
            args=args+['-Xclang','-dependency-file','-Xclang',cpath(out/(name+'.d')),'-Xclang','-MT','-Xclang',name]
        cmd=docker+args;commands.append(cmd);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');run(cmd,out,name,seconds=240)
    native('compiler',[tool+'clang++','--version'])
    native('device',[tool+'clang++',*common,'--cuda-device-only',*dev_extra,'-Rpass-analysis=kernel-resource-usage','-S',cpath(out/'megakernel_d0.hip'),'-o',cpath(out/'mk.s')])
    native('device-object',[tool+'llvm-mc','-triple=amdgcn-amd-amdhsa','-mcpu=gfx942','-filetype=obj',cpath(out/'mk.s'),'-o',cpath(out/'mk.o')])
    native('device-link',[tool+'ld.lld','-shared',cpath(out/'mk.o'),'-o',cpath(out/'model.hsaco')])
    native('host-object',[tool+'clang++',*common,'--cuda-host-only','-c',cpath(host),'-o',cpath(out/'host.o')])
    originals=[]
    if '-DETX_VL_CHECK' in flags:
        # Compile the unchanged original kernel specializations as real HIP
        # host+device objects. Candidate imported IR is never its own reference.
        from etx.importer.spec import SOURCES
        from etx.importer.vllm_kernels import _flags
        exports={n for n,d in json.loads((reference/'imp/imports.json').read_text()).items() if not d.get('dsl')}
        require(exports=={'attn_mfma16_g8','attn_reduce','rms_norm_2d'},'registered unchanged original checker population; no Triton expansion')
        table=['#include <cstring>']
        prior=dict(os.environ)
        os.environ.update(ETX_ROCM_HEADERS='/code/rocm-headers/inc',ETX_TORCH_HEADERS='/code/rocm-headers/pytorch')
        try:
            for spec in SOURCES:
                kernels=[k for k in spec.kernels if k.export in exports]
                if not kernels:continue
                base=Path(spec.path).stem
                src=out/('orig_'+base+'.hip')
                src.write_text('#include '+json.dumps(str(reference/'imp'/(base+'.hip')))+'\n'+''.join('extern "C" const void* etx_orig_'+k.export+'() { return (const void*)&'+k.spec+'; }\n' for k in kernels))
                obj=out/('orig_'+base+'.o')
                extra=_flags(HOME_CODE/'vllm',spec)
                native('orig-'+base,[tool+'clang++',*common,*extra,'-c',cpath(src),'-o',cpath(obj)])
                originals.append(obj)
                table.extend('extern "C" const void* etx_orig_'+k.export+'();' for k in kernels)
        finally:
            os.environ.clear();os.environ.update(prior)
        table+=['extern "C" bool etx_orig_triton(const char*,const char**,const char**,int*) { return false; }',
                'extern "C" const void* etx_orig_find(const char* n) {']
        for name in sorted(exports):table.append('  if (!strcmp(n,'+json.dumps(name)+')) return etx_orig_'+name+'();')
        table+=['  return nullptr;','}']
        original_table=out/'orig_table.hip';original_table.write_text('\n'.join(table)+'\n')
        native('orig-table',[tool+'clang++',*common,'--cuda-host-only','-c',cpath(original_table),'-o',cpath(out/'orig_table.o')])
        originals.append(out/'orig_table.o')
    # Preserve the genuine missing prerequisite rather than a fake host link.
    runtime=Path(os.environ.get('ETX_PROGRAM_HIP_RUNTIME',str(Path.home()/'code/rocm-headers/lib/libamdhip64.so'))).expanduser().resolve()
    if 'ETX_PROGRAM_HIP_RUNTIME' in os.environ:require(runtime.is_file(),'explicit genuine runtime missing')
    if runtime.is_file():
        native('host-link',[tool+'clang++',*common,*dev_extra,cpath(out/'megakernel_d0.hip'),cpath(host),'-x','none',*[cpath(p) for p in originals],'-L'+cpath(runtime.parent),'-Wl,-rpath-link,'+cpath(runtime.parent),'-Wl,-rpath-link,'+cpath(runtime.parents[3]/'usr/lib/x86_64-linux-gnu'),'-lamdhip64','-o',cpath(out/'run')])
        link=0
    else:
        link=77;(out/'host-link.exit').write_text('77\n');(out/'host-link.log').write_text('UNAVAILABLE: existing local HIP runtime libamdhip64.so absent; original full-model host link not performed. No executable/model certificate.\n')
    runtime_files={str(p):sha(p) for root in (runtime.parent,runtime.parents[3]/'usr/lib/x86_64-linux-gnu') for p in root.glob('*.so*') if p.is_file()} if runtime.is_file() else {}
    identity=dict(schema=1,runtime_files=runtime_files,flags=flags,environment=env_record,commands=commands,host_link_exit=link,
                  host_source_sha256=sha(host),original_reference_objects={str(p):sha(p) for p in originals},dependency_files=dependencies(out),check_enabled='-DETX_VL_CHECK' in flags,gpu_correctness='UNMEASURED',performance='UNMEASURED',
                  hashes={str(p):sha(p) for p in out.iterdir() if p.is_file()},
                  reused_imports={str(imports):sha(imports)} if imports else {},
                  toolchain_seal_sha256=sha('/Users/ckl/.hermes/cache/scratch/etx-experiment-preparation/toolchain.json'))
    (out/'build-identity.json').write_text(json.dumps(identity,indent=2)+'\n');return identity


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['qwen-control','qwen-g01','routes-b2','routes-b4'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args()
    if a.kind.startswith('qwen'):
        require(a.reference is not None,'read-only original kept import closure required')
        r=prepare_qwen(a.out,a.reference,last_router=a.kind=='qwen-g01')
    else:r=prepare_routes(a.out,int(a.kind[-1]))
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
