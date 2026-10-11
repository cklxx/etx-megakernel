"""Reproducible CPU device build and OFF identity evidence (no GPU).

Compare the default emitter with the exact starting commit, then compile the
same source filename with/without a disabled diagnostic overlay. Compare
optimized device IR AND ISA bytes, not just source anchors/resource counts.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import types

from etx.codegen import emit_kernel
from etx.codegen.inbody import instrument
from . import dispatcher
from .build import ROOT, FLAGS, current_source, preamble, resources
from .pack import sha
from etx.codegen.lowering import emit_lowering_header
from etx.machine.model import load_machine
from etx.ops import gemv


def check(out, model_build=None):
    out = Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    llvm=Path(os.environ.get('ETX_LLVM','/opt/homebrew/opt/llvm/bin'))
    libs=Path(os.environ['ETX_DEVLIBS'])
    if not (libs/'ocml.bc').is_file(): raise ValueError('device libraries missing')
    if model_build:
        b=Path(model_build).resolve()
        source=(b/'megakernel_d0.hip').read_text()
        # rocm_build_check emits container-mount paths. Map only that mount
        # spelling to the identical local files; no source/body is rewritten.
        source=source.replace('/code/',str(Path.home()/'code')+'/')
        common=['-I'+str(ROOT/'etx/runtime/include'),'-I'+str(b),'-I'+str(ROOT),
                '-Xclang','-mlink-builtin-bitcode','-Xclang',str(b/'imports.bc'),
                '-DETX_DSL_WSUM','-DETX_DSL_SCALAR','-DETX_DSL_GU1','-DETX_DSL_RBATCH',
                '-DETX_DSL_NT','-DETX_DSL_VECNORM','-DETX_TOPK_KEY32']
        emitter_match=None # checked independently on the two-task fixture
    else:
        (out/'etx_lowering.h').write_text(emit_lowering_header(load_machine('gfx942')))
        tile=out/'tiles.hip'; tile.write_text(preamble(gemv.LDS_BYTES)+current_source())
        plan=dispatcher.graph(tile)
        os.environ.pop('ETX_INBODY_EMIT',None)
        source=emit_kernel(plan)
        baseline=types.ModuleType('etx.codegen.baseline'); baseline.__package__='etx.codegen'
        code=subprocess.check_output(['git','show','4a39f92:etx/codegen/kernel.py'],cwd=ROOT,text=True)
        exec(compile(code,'baseline_kernel.py','exec'),baseline.__dict__)
        emitter_match=source==baseline.emit_kernel(plan)
        if not emitter_match: raise ValueError('default emitter changed from starting commit')
        common=['-I'+str(ROOT/'etx/runtime/include'),'-I'+str(out),'-I'+str(ROOT)] + [f for f in FLAGS if not f.startswith('-DETX_EMIT_') and f != '-DETX_DSL_COH']
    report={}
    # Identical source filenames prevent ModuleID/.file noise hiding differences.
    hip=out/'kernel.hip'
    for name,text,flag in [('original',source,[]),('off',instrument(source),[]),('on',instrument(source),['-DETX_INBODY=1'])]:
        hip.write_text(text)
        commands=[['bash',ROOT/'etx/importer/hip_ir_local.sh',hip,out/(name+'.ll')]+common+flag,
                  [llvm/'llc','-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',out/(name+'.ll'),'-o',out/(name+'.s')]]
        for i,cmd in enumerate(commands):
            print(f'CPU BUILD {name} stage={i}',flush=True)
            with (out/f'{name}_{i}.log').open('w') as f:
                subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        report[name]=resources((out/(name+'.s')).read_text())
    # Source hashing changes HIP's unused TU anchor. New guarded source lines
    # change inline-asm source-location metadata. Normalize ONLY these and the
    # assembly .file diagnostic. All instructions/resource metadata still match.
    import re
    def ir(name):
        s=(out/(name+'.ll')).read_text()
        locations=set(re.findall(r'!srcloc !(\d+)',s))
        for loc in locations:
            s=re.sub(rf'(?m)^!{loc} = !\{{i64 \d+\}}$',f'!{loc} = !{{i64 0}}',s)
        s=re.sub(r'__hip_cuid_[0-9a-f]+','__hip_cuid_NORMALIZED',s)
        return re.sub(r"(?m)^; ModuleID = .*\n",'',s)
    def isa(name):
        s=re.sub(r'__hip_cuid_[0-9a-f]+','__hip_cuid_NORMALIZED',(out/(name+'.s')).read_text())
        return re.sub(r'(?m)^\s*\.file\s+.*\n','',s)
    for name in ('original','off','on'):
        subprocess.run([str(llvm/'llvm-mc'),'-triple=amdgcn-amd-amdhsa','-mcpu=gfx942','-filetype=obj',
            str(out/(name+'.s')),'-o',str(out/(name+'.o'))],check=True)
        subprocess.run([str(llvm/'llvm-objcopy'),'-O','binary','--only-section=.text',
            str(out/(name+'.o')),str(out/(name+'.text.bin'))],check=True)
    result=dict(start_commit='4a39f92',default_emitter_byte_equal=emitter_match,
        off_ir_code_equal=ir('original')==ir('off'),off_isa_code_equal=isa('original')==isa('off'),
        off_device_text_byte_equal=(out/'original.text.bin').read_bytes()==(out/'off.text.bin').read_bytes(),
        device_text_sha256={n:sha(out/(n+'.text.bin')) for n in ('original','off','on')},
        normalization=['unused __hip_cuid anchor','inline-asm !srcloc numbers','ModuleID and .file'],
        resources=report,compiler=subprocess.check_output([llvm/'clang++','--version'],text=True),
        device_libraries=str(libs),devlib_hashes={p.name:sha(p) for p in sorted(libs.glob('*.bc'))},
        gpu_perturbation_ms=None,gpu_checks='NOT_RUN_CPU_ONLY',
        paths={n:{s:sha(out/(n+'.'+s)) for s in ('ll','s')} for n in ('original','off','on')})
    (out/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['off_ir_code_equal'] or not result['off_isa_code_equal'] or not result['off_device_text_byte_equal']:
        raise ValueError('disabled overlay changed optimized code')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model-build',type=Path)
    p.add_argument('--vm-off',action='store_true',help='compare ordinary/disabled overlay with installed ROCm container compiler')
    a=p.parse_args()
    print(json.dumps(vm_off(a.out,a.model_build) if a.vm_off else check(a.out,a.model_build),indent=2))


def vm_off(out,build):
    """VM compiler identity, without GPU, network, image pulls or provisioning."""
    out=Path(out).resolve(); build=Path(build).resolve(); out.mkdir(parents=True,exist_ok=False)
    source=(build/'megakernel_d0.hip').read_text()
    (out/'original.hip').write_text(source); (out/'off.hip').write_text(instrument(source))
    def mount(path): return '/code/'+str(Path(path).resolve().relative_to(Path.home()/'code'))
    script=r'''
import json,subprocess,sys
from pathlib import Path
out,build,root=map(Path,sys.argv[1:])
r=Path('/code/rocm-headers'); llvm=r/'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin'
libs=r/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode'
flags=['-x','hip','--cuda-device-only','--offload-arch=gfx942','-O3','-std=c++17','-nogpuinc',
       '--rocm-device-lib-path='+str(libs),'-isystem',str(r/'inc'),'-include','__clang_hip_runtime_wrapper.h',
       '-D__HIP_PLATFORM_AMD__','-I'+str(root/'etx/runtime/include'),'-I'+str(build),'-I'+str(root),
       '-DETX_DSL_WSUM','-DETX_DSL_SCALAR','-DETX_DSL_GU1','-DETX_DSL_RBATCH','-DETX_DSL_NT',
       '-DETX_DSL_VECNORM','-DETX_TOPK_KEY32','-Xclang','-mlink-builtin-bitcode','-Xclang',str(build/'imports.bc')]
for arm in ('original','off'):
    (out/'kernel.hip').write_bytes((out/(arm+'.hip')).read_bytes())
    for suffix,extra in [('ll',['-emit-llvm']),('s',[])]:
        print('CPU VM '+arm+' '+suffix,flush=True)
        with (out/(arm+'_'+suffix+'.log')).open('w') as f:
            subprocess.run([str(llvm/'clang++'),*flags,'-S',*extra,str(out/'kernel.hip'),'-o',str(out/(arm+'.'+suffix))],
                           cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([str(llvm/'llvm-mc'),'-triple=amdgcn-amd-amdhsa','-mcpu=gfx942','-filetype=obj',str(out/(arm+'.s')),'-o',str(out/(arm+'.o'))],check=True)
    subprocess.run([str(llvm/'llvm-objcopy'),'-O','binary','--only-section=.text',str(out/(arm+'.o')),str(out/(arm+'.text.bin'))],check=True)
(out/'compiler.txt').write_text(subprocess.check_output([str(llvm/'clang++'),'--version'],text=True))
'''
    home_code=Path.home()/'code'
    cmd=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64',
         '-v',str(home_code)+':/code','-v',str(home_code)+':'+str(home_code),
         '--entrypoint','python3','etx-rocm22','-c',script,mount(out),mount(build),mount(ROOT)]
    with (out/'docker.log').open('w') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    import re
    def normalize(name,suffix):
        s=(out/(name+'.'+suffix)).read_text()
        if suffix=='ll':
            for loc in set(re.findall(r'!srcloc !(\d+)',s)):
                s=re.sub(rf'(?m)^!{loc} = !\{{i64 \d+\}}$',f'!{loc} = !{{i64 0}}',s)
            s=re.sub(r"(?m)^; ModuleID = .*\n",'',s)
        else: s=re.sub(r'(?m)^\s*\.file\s+.*\n','',s)
        return re.sub(r'__hip_cuid_[0-9a-f]+','__hip_cuid_NORMALIZED',s)
    result=dict(start_commit='4a39f92',off_ir_code_equal=normalize('original','ll')==normalize('off','ll'),
        off_isa_code_equal=normalize('original','s')==normalize('off','s'),
        off_device_text_byte_equal=(out/'original.text.bin').read_bytes()==(out/'off.text.bin').read_bytes(),
        compiler=(out/'compiler.txt').read_text(),resources={n:resources((out/(n+'.s')).read_text()) for n in ('original','off')},
        artifact_hashes={n+'.'+s:sha(out/(n+'.'+s)) for n in ('original','off') for s in ('hip','ll','s','text.bin')},
        normalization=['ModuleID/.file','unused HIP CUID','inline-asm source-location metadata'],
        gpu_perturbation_ms=None,gpu_checks='NOT_RUN_CPU_ONLY',docker_command=cmd[:cmd.index('-c')])
    (out/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['off_device_text_byte_equal'] or not result['off_isa_code_equal'] or not result['off_ir_code_equal']:
        raise ValueError('VM disabled overlay changed code')
    return result


if __name__=='__main__': main()
