"""Offline, serial deployment compiler builds, without a GPU or fake HIP library."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .protocol import require,sha

ROOT=Path(__file__).resolve().parents[2]
HOME_CODE=Path.home()/'code'
DEPS=Path('/Users/ckl/.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/link-deps/root')
FLAGS=['-DETX_DSL_COH','-DETX_DSL_NT','-DETX_DSL_VECNORM','-DETX_DSL_WSUM',
       '-DETX_DSL_SCALAR','-DETX_DSL_GU1','-DETX_DSL_RBATCH','-DETX_DSL_NORM_HOIST',
       '-DETX_TOPK_KEY32','-DETX_EMIT_DSL_BIGK=0','-DETX_EMIT_DSL_EXT=0','-DETX_EMIT_FOLD_TOPK=0']


def cpath(path):
    return '/code/'+str(Path(path).resolve().relative_to(HOME_CODE))


def dependencies(out):
    import shlex
    paths=set();image_paths=set()
    for receipt in Path(out).glob('*.d'):
        text=receipt.read_text().replace('\\\n',' ')
        require(': ' in text,'malformed real compiler dependency receipt')
        for name in shlex.split(text.split(': ',1)[1]):
            path=HOME_CODE/name[len('/code/'):] if name.startswith('/code/') else Path(name)
            if path.is_file():paths.add(path.resolve())
            elif name.startswith(('/usr/','/lib/','/opt/')):image_paths.add(name)
            else:raise ValueError('missing actual compiler dependency '+str(path))
    require(paths,'actual compiler dependency receipts absent')
    if image_paths:
        code='import json,sys,hashlib,pathlib;print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in json.load(sys.stdin)},sort_keys=True))'
        r=subprocess.run(['docker','run','--rm','-i','--pull=never','--network=none','--platform','linux/amd64','etx-rocm22','python3','-c',code],input=json.dumps(sorted(image_paths)),text=True,capture_output=True,timeout=30,check=True)
        image_id=subprocess.check_output(['docker','image','inspect','etx-rocm22','--format','{{.Id}}'],text=True).strip()
        receipt=Path(out)/'image-dependencies.json'
        receipt.write_text(json.dumps(dict(image_id=image_id,files=json.loads(r.stdout)),indent=2)+'\n');paths.add(receipt.resolve())
    return {str(p):sha(p) for p in sorted(paths)}


def build(out, sources, host, *, flags=FLAGS):
    from etx.codegen.lowering import emit_lowering_header
    from etx.machine.model import load_machine
    from examples.vllm_llm.bf16_300.build import resources
    out=Path(out).resolve()
    require(out.is_relative_to(HOME_CODE),'build output must be below ~/code')
    require(shutil.disk_usage(out.parent).free>=2*(1<<30),'storage free floor2GiB')
    out.mkdir(exist_ok=False)
    (out/'tmp').mkdir()
    (out/'etx_lowering.h').write_text(emit_lowering_header(load_machine('gfx942')))
    require((DEPS/'usr/lib/x86_64-linux-gnu/libxml2.so.2').is_file(),'missing retained linker dependencies')
    cmds=[]
    def run(name,cmd,*,container=True):
        argv=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64',
              '-v',str(HOME_CODE)+':/code','-v',str(DEPS)+':/link-deps:ro',
              '-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu',
              '-e','TMPDIR='+cpath(out/'tmp'),'-w',cpath(ROOT),'etx-rocm22',*map(str,cmd)] if container else list(map(str,cmd))
        print('BUILD',name,flush=True)
        start=time.monotonic()
        with (out/(name+'.log')).open('w') as f:
            try:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=180,check=False)
            except subprocess.TimeoutExpired:
                cmds.append(dict(name=name,argv=argv,timeout=180));(out/'commands.json').write_text(json.dumps(cmds,indent=2));raise
        cmds.append(dict(name=name,argv=argv,exit=r.returncode,seconds=time.monotonic()-start))
        (out/'commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
        require(r.returncode==0,f'build failed: {name}; full log {out/(name+".log")}')
    tool='/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/'
    run('compiler',[tool+'clang++','--version'])
    run('linker',[tool+'ld.lld','--version'])
    common=['-x','hip','--cuda-device-only','--offload-arch=gfx942','-O3','-std=c++17','-nogpuinc',
            '--rocm-device-lib-path=/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode',
            '-isystem','/code/rocm-headers/inc','-include','__clang_hip_runtime_wrapper.h','-D__HIP_PLATFORM_AMD__',
            '-I'+cpath(ROOT/'etx/runtime/include'),'-I'+cpath(ROOT),'-I'+cpath(out),*flags]
    res={}
    for name,source in sources.items():
        src=out/(name+'.hip');src.write_text(source)
        run(name+'-assembly',[tool+'clang++',*common,'-Xclang','-dependency-file','-Xclang',cpath(out/(name+'.d')),'-Xclang','-MT','-Xclang',name,'-Rpass-analysis=kernel-resource-usage','-S',cpath(src),'-o',cpath(out/(name+'.s'))])
        run(name+'-object',[tool+'clang++','-target','amdgcn-amd-amdhsa','-mcpu=gfx942','-c',cpath(out/(name+'.s')),'-o',cpath(out/(name+'.o'))])
        run(name+'-module',[tool+'ld.lld','-shared',cpath(out/(name+'.o')),'-o',cpath(out/(name+'.hsaco'))])
        res[name]=resources((out/(name+'.s')).read_text())
    # host is a real linked x86 ELF using libdl to bind the actual HIP runtime
    # on the GPU machine. --help does not attempt dlopen or device discovery.
    run('host',[tool+'clang++','-x','c++','-std=c++17','-O2','-D__HIP_PLATFORM_AMD__',
                '-I/code/rocm-headers/inc','-I'+cpath(ROOT),'-Xclang','-dependency-file','-Xclang',cpath(out/'host.d'),'-Xclang','-MT','-Xclang','host','-fuse-ld=lld',cpath(host),'-ldl','-o',cpath(out/'replay')])
    run('host-help',[cpath(out/'replay'),'--help'])
    result=dict(schema=1,flags=list(flags),commands=cmds,resources=res,compiler=(out/'compiler.log').read_text(),
                linker=(out/'linker.log').read_text(),gpu_checks='NOT_RUN',performance='UNMEASURED',
                host_source_sha256=sha(host),dynamic_header_sha256=sha(ROOT/'examples/experiment_program/hip_dynamic.h'),
                dependency_files={**dependencies(out),**{str(p):sha(p) for p in (DEPS/'usr/lib/x86_64-linux-gnu').glob('*') if p.is_file()}},
                files={str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='identity.json'})
    (out/'identity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
