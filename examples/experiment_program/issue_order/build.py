"""Offline ROCm22 build: real gfx942 objects/code object + x86 host executable.

No fake runtime or device invocation. Build output must be a fresh directory.
The dynamic host loads the real HIP module API only on an explicitly run GPU host.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from .source import transform, prep_source, MARKERS, ATTENTION_SHA
from .evidence import sha, require, EXPORTS, native_events, native_admission

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent

def rename_prep(text,arm):
    names=('dsl_ap_bf','dsl_ap_f','dsl_ap_pk','dsl_ap_scalar','dsl_run_dsl_attnprep','etx_tile_dsl_attnprep','etx_pre_dsl_attnprep')
    return re.sub(r'\b(?:'+ '|'.join(names)+r')\b',lambda m:m[0]+'_'+arm,text)

def dependency_flags(path):
    # Clang22 HIP driver ignores -MF for device-only actions. Pass the real
    # cc1 dependency writer explicitly, then require its receipt below.
    return ['-Xclang','-dependency-file','-Xclang',str(path),'-Xclang','-MT','-Xclang','etx-issue-order','-Xclang','-sys-header-deps']

def dependency_inputs(out,required):
    import shlex
    out=Path(out);result={}
    for name in required:require((out/name).is_file(),'missing compiler device dependency receipt '+name)
    for dep in out.rglob('*.d'):
        for name in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
            p=Path(name).resolve()
            if str(p).startswith('/code/'):
                require(p.is_file(),'missing mounted compiler dependency '+str(p));result[str(p)]=sha(p)
    return result

def command(cmd,out,name,env=None):
    print('STAGE',name,' '.join(map(str,cmd)),flush=True)
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    (out/(name+'.rc')).write_text(str(result.returncode)+'\n')
    if result.returncode: raise RuntimeError(f'{name} exit={result.returncode}: {out/(name+".log")}')

def kernel_source(record,imports):
    from etx.codegen.lowering import emit_lowering_header
    from etx.machine.model import load_machine
    from etx.importer.adapter import emit_adapters
    from etx.ops import coh
    pre='#include "etx_lowering.h"\n#include "etx/primitives.h"\n#include "etx/tile_util.h"\n'
    if record=='G06':
        pre='#define ETX_IMPORT_LDS 16\n#define ETX_IMPORT_SLOTS 1\n'+pre+'#include "etx/import.h"\n'+coh.HELPERS
        for arm in ('control','candidate'):
            pre+=f'#include "{arm}/body.hip"\n'
    else: pre+=emit_adapters(imports)
    for arm in ('control','candidate'):
        call=f'dsl_run_dsl_attnprep_{arm}(etx_imp_argblock(c.args,0),blockIdx.x);' if record=='G06' else f'etx_tile_io_{arm}(&c);'
        pre+=f'''\nextern "C" __global__ __launch_bounds__(1024, 1) void io_{arm}(void** table) {{
  etx_import_init(); __syncthreads();
  etx_ctx c{{}}; c.args=table; c.coord[0]=blockIdx.x;
  {call}
  // Every writing wave drains; imported adapters retain original slot rounds.
  ETX_RELEASE_DOMAIN(); __syncthreads();
}}\n'''
    return pre,emit_lowering_header(load_machine('gfx942'))

def prepare(record,out,attention_source,reference_imports,vllm,headers):
    from etx.importer.spec import SOURCES
    from etx.importer.vllm_kernels import _flags
    from etx.importer.ir_import import import_kernel
    from etx.importer.bundle import bundle
    from examples.vllm_llm.bf16_300.build import resources
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    headers=Path(headers); vllm=Path(vllm)
    llvm=headers/'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin'
    libs=headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode'
    os.environ.update(ETX_LLVM=str(llvm),ETX_TORCH_HEADERS=str(headers/'pytorch'),ETX_ROCM_HEADERS=str(headers/'inc'))
    compiler=llvm/'clang++'; commands=[]
    def run(cmd,name): commands.append(list(map(str,cmd))); command(cmd,out,name)
    original=json.loads(Path(reference_imports).read_text())
    desc=original[EXPORTS[record]]
    text=prep_source() if record=='G06' else Path(attention_source).read_text()
    if record!='G06': require(sha(attention_source)==ATTENTION_SHA,'frozen source sha mismatch')
    changed=transform(record,text)
    # Store the original unmodified text separately and never use candidate
    # output as the reference authority for either IR or byte checks.
    imports={}; body_sources={}
    for arm,s in (('control',text),('candidate',changed)):
        armout=out/arm; armout.mkdir()
        if record=='G06':
            # Rename identifiers only to make both math islands coexist.
            s=rename_prep(s,arm)
            (armout/'body.hip').write_text(s); body_sources[arm]=s
        else:
            (armout/'attention.hip').write_text(s); body_sources[arm]=s
            spec=next(k for source in SOURCES for k in source.kernels if k.export==EXPORTS[record])
            srcspec=next(source for source in SOURCES if spec in source.kernels)
            flags=_flags(vllm,srcspec)
            run([compiler,'-x','hip','--cuda-device-only','--offload-arch=gfx942','-nogpuinc','-nogpulib','-isystem',headers/'inc','-include','__clang_hip_runtime_wrapper.h','-O3','-gline-tables-only','-std=c++17','-S','-emit-llvm',*dependency_flags(armout/'native.d'),*flags,armout/'attention.hip','-o',armout/'native.ll'],arm+'_native_ir')
            imported,info=import_kernel((armout/'native.ll').read_text(),spec.match,'etx_imp_io_'+arm,tuple(desc['block']),1024,spec.lds_cap,spec.inline,desc['slots'])
            # All ABI, slot/thread/barrier/LDS fields must match original import.
            actual=info.to_json()
            for key in ('block','threads','slots','grid_offset','arg_bytes','lds_bytes_per_slot','lds_bytes','barriers','slot_threads','slot_barrier'):
                require(list(actual[key])==list(desc[key]) if key=='block' else actual[key]==desc[key],'changed original import protocol '+key)
            imports['io_'+arm]=actual
            (out/('io_'+arm+'.ll')).write_text(imported)
            run([llvm/'llvm-link',out/('io_'+arm+'.ll'),'-o',out/('io_'+arm+'.bc')],arm+'_assemble')
    if imports: bundle(out,list(imports),libs,out/'imports.bc')
    s,lowering=kernel_source(record,imports)
    (out/'kernel.hip').write_text(s); (out/'etx_lowering.h').write_text(lowering)
    common=['-x','hip','--cuda-device-only','--offload-arch=gfx942','-O3','-gline-tables-only','-std=c++17','-nogpuinc','--rocm-device-lib-path='+str(libs),'-isystem',str(headers/'inc'),'-include','__clang_hip_runtime_wrapper.h','-D__HIP_PLATFORM_AMD__','-DETX_DSL_COH','-I'+str(ROOT/'etx/runtime/include'),'-I'+str(out),'-I'+str(ROOT)]
    if imports: common+=['-Xclang','-mlink-builtin-bitcode','-Xclang',str(out/'imports.bc')]
    run([compiler,*common,'-Rpass-analysis=kernel-resource-usage',*dependency_flags(out/'kernel.d'),'-S',out/'kernel.hip','-o',out/'kernel.s'],'device_isa')
    run([compiler,'-target','amdgcn-amd-amdhsa','-mcpu=gfx942','-c',out/'kernel.s','-o',out/'kernel.o'],'device_object')
    run([llvm/'ld.lld','-shared',out/'kernel.o','-o',out/'kernel.hsaco'],'device_link')
    run([llvm/'llvm-readobj','--notes',out/'kernel.hsaco'],'native_metadata')
    run([llvm/'llvm-objdump','-d',out/'kernel.hsaco'],'native_disassembly')
    # Native host executable uses only dlopen, not a fake or missing HIP stub.
    pointer_offsets=[p['offset'] for p in desc['params'] if p['ty'].startswith('ptr')]
    layout=(f'#define IO_RECORD "{record}"\n#define IO_EXPORT "{EXPORTS[record]}"\n#define IO_ARG_BYTES {desc["arg_bytes"]}\n#define IO_GRID_OFF {desc["grid_offset"]}\n#define IO_SLOTS {desc["slots"]}\nstatic const int io_ptr_offsets[]={{'+','.join(map(str,pointer_offsets))+'};\n')
    (out/'io_layout.h').write_text(layout)
    run([compiler,'-x','c++','-O2','-std=c++17','-D__HIP_PLATFORM_AMD__','-I'+str(headers/'inc'),'-I'+str(out),'-I'+str(ROOT),'-fuse-ld=lld','-MD','-MF',out/'host.d',HERE/'replay.cpp','-ldl','-o',out/'replay'],'host_link')
    run([out/'replay','--help'],'host_help')
    parsed=resources((out/'kernel.s').read_text())
    events={}; rs={}
    asm=(out/'kernel.s').read_text()
    for arm in ('control','candidate'):
        match=re.search(r'(?ms)^io_'+arm+r':.*?^\.Lfunc_end\d+:',asm)
        require(match is not None,'missing native kernel')
        markers=MARKERS[record]
        if record=='G06': markers=tuple(rename_prep(x,arm) for x in markers)
        events[arm]=native_events(match[0],body_sources[arm],markers,arm+('/body.hip' if record=='G06' else '/attention.hip'))
        rs[arm]=[r for r in parsed if r['kernel']=='io_'+arm]
    try: admission=native_admission(events['control'],events['candidate'],rs['control'],rs['candidate'])
    except ValueError as error: admission={'status':'BLOCKED_NATIVE_MAPPING','timing_eligible':False,'reason':str(error)}
    closure=[HERE/p for p in ('__init__.py','__main__.py','build.py','source.py','evidence.py','receipts.py','paths.py','replay.cpp')]+list((ROOT/'etx/runtime/include').rglob('*.h'))+[ROOT/p for p in ('etx/ops/attnprep.py','etx/ops/coh.py','etx/importer/ir_import.py','etx/importer/adapter.py','etx/importer/bundle.py','etx/importer/spec.py','etx/codegen/lowering.py','etx/machine/model.py','examples/experiment_program/hip_dynamic.h','examples/experiment_program/protocol.py')]
    dependency_files=dependency_inputs(out,['kernel.d','host.d']+([] if record=='G06' else ['control/native.d','candidate/native.d']))
    identity={'schema':'etx.issue-order.build.v1','record':record,'commands':commands,'compiler':subprocess.check_output([compiler,'--version'],text=True),'resources':rs,'native_events':events,'admission':admission,'original_imports_sha256':sha(reference_imports),'original_imports':original,'native_input_status':'REQUIRED_SEPARATE_SEAL','gpu_execution':'NOT_RUN','math_flags':'-O3; no unsafe/fast math; default clang contraction unchanged','hashes':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted(closure)},'dependency_files':dependency_files}
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps({'record':record,'admission':admission,'resources':rs},indent=2),flush=True)
    return identity
