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
        subprocess.run([str(llvm/'llc'),'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3','-filetype=obj',
            str(out/(name+'.ll')),'-o',str(out/(name+'.o'))],check=True)
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
    a=p.parse_args(); print(json.dumps(check(a.out,a.model_build),indent=2))


if __name__=='__main__': main()
