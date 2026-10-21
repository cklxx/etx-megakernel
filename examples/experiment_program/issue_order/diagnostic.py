"""Optional native per-wave landmarks, matched empty and uninstrumented controls.

Instrumentation never authorizes optimization. Attribution fails closed unless
all native allocations match, every declared wave/task is covered and measured
perturbation is <=500ns. No clock-derived removable-time claim is made locally.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
from .source import statement_inventory, RECORDS
from .evidence import require, sha
from .build import ROOT,HERE,command,kernel_source
from examples.experiment_program.protocol import resolve_path
from .paths import dependency_path


def verify_build(path):
    # Identical seals, with explicit mounted-vs-host path resolution. This
    # function also runs *inside* the offline compiler image (/code mount).
    path=resolve_path(path);value=json.loads((path/'identity.json').read_text())
    for name,digest in value['hashes'].items():require(sha(path/name)==digest,'stale original native build '+name)
    for name,digest in value['source_hashes'].items():require(sha(ROOT/name)==digest,'stale original implementation '+name)
    for name,digest in value.get('dependency_files',{}).items():
        local=dependency_path(name)
        require(sha(local)==digest,'stale original mounted dependency '+name)
    return value

PHASES=4
TRACE_NATIVE=32*16*PHASES
TRACE_PLACEMENT=TRACE_NATIVE+32*16
TRACE_TOTAL=TRACE_PLACEMENT+32*16
TRACE_HEADER=r'''#pragma once
#include <hip/hip_runtime.h>
#define IO_TRACE_NATIVE 2048
#define IO_TRACE_TOTAL 3072
#define IO_TRACE_PLACEMENT 2560
extern __device__ unsigned long long io_trace_values[IO_TRACE_NATIVE];
extern __device__ unsigned long long io_trace_final_values[512];
extern __device__ unsigned long long io_trace_placement_values[512];
extern __device__ unsigned io_trace_enabled;
static __device__ __attribute__((always_inline)) void io_trace_stamp(unsigned phase) {
  if ((threadIdx.x % 64)==0 && io_trace_enabled) {
    asm volatile("" ::: "memory");
    const unsigned flat=blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z);
    io_trace_values[(flat*16+threadIdx.x/64)*4+phase]=__builtin_amdgcn_s_memrealtime();
    asm volatile("" ::: "memory");
  }
}
'''
TRACE_EXPORT=r'''
__device__ unsigned long long io_trace_values[IO_TRACE_NATIVE];
__device__ unsigned long long io_trace_final_values[512];
__device__ unsigned long long io_trace_placement_values[512];
__device__ unsigned io_trace_enabled;
extern "C" __global__ void io_trace_reset(unsigned enabled) {
  for(unsigned i=threadIdx.x;i<IO_TRACE_NATIVE;i+=blockDim.x)io_trace_values[i]=0;
  for(unsigned i=threadIdx.x;i<512;i+=blockDim.x){io_trace_final_values[i]=0;io_trace_placement_values[i]=0;}
  if(threadIdx.x==0)io_trace_enabled=enabled;
}
extern "C" __global__ void io_trace_export(unsigned long long* out) {
  for(unsigned i=threadIdx.x;i<IO_TRACE_NATIVE;i+=blockDim.x)out[i]=io_trace_values[i];
  for(unsigned i=threadIdx.x;i<512;i+=blockDim.x)out[IO_TRACE_NATIVE+i]=io_trace_final_values[i];
  for(unsigned i=threadIdx.x;i<512;i+=blockDim.x)out[IO_TRACE_PLACEMENT+i]=io_trace_placement_values[i];
}
'''

def trace_header(record):
    require(record in RECORDS,'unknown trace shape')
    # Prep's admitted native grid is strictly (4,1,1). Reading unused y/z or
    # gridDim would add hidden dispatch/kernarg requirements to this wrapper.
    return TRACE_HEADER.replace('blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z)','blockIdx.x') if record=='G06' else TRACE_HEADER

def insert_before(text,marker,stamp):
    require(marker in text,'missing diagnostic marker '+marker)
    return text.replace(marker,'io_trace_stamp('+str(stamp)+');\n'+marker,1)

def instrument(record,text,phase=None):
    old=text
    if record=='G06':
        text=insert_before(text,'      const long long pos =',0)
        text=insert_before(text,'      float lo[4], hi[4], ss = 0.f;',1)
        text=insert_before(text,'      const float rms_rcp = rsqrtf(ss / (float)HD + eps);',2)
    elif record=='G07':
        start=text.index('// Grid: (num_heads, num_seqs).\ntemplate <typename scalar_t, typename OUTT, int HEAD_SIZE')
        stop=text.index('\n}',start)+2;body=text[start:stop]
        body=insert_before(body,'  const scalar_t* tmp_out_ptr =',0)
        body=body.replace('  __syncthreads();','  io_trace_stamp(1);\n  __syncthreads();\n  io_trace_stamp(2);',1)
        body=insert_before(body,'    float global_exp_sum = rescaled_exp_sum[0];',3)
        text=text[:start]+body+text[stop:]
    elif record=='G08':
        text=insert_before(text,'  constexpr int VTOKENS_PER_LANE =',0)
        start=text.index('  // v fetches are 16head elems across lanes x 16 tokens per lane')
        begin=text.index('{',text.index('  for (int vhe_depth',start));depth=1;end=begin+1
        while depth:
            depth+=(text[end]=='{')-(text[end]=='}');end+=1
        text=text[:end]+'\n  io_trace_stamp(1);\n'+text[end:]
        text=insert_before(text,'  // fetch Q in shared across warps and then write to registers',2)
        text=text.replace('  __syncthreads();','  __syncthreads();\n  io_trace_stamp(3);',1)
    else: raise ValueError('unknown diagnostic record')
    if phase is not None:
        require(phase in range(4),'invalid sparse phase')
        text=re.sub(r'io_trace_stamp\(([0-3])\);',lambda m:m[0] if int(m[1])==phase else '',text)
    stripped=re.sub(r'io_trace_stamp\([0-3]\);','',text)
    require(statement_inventory(stripped)==statement_inventory(old),'probe changed original body')
    return text

def qualify(uninstrumented,empty,probe,base_ns,empty_ns,probe_ns,complete):
    fields=('vgpr','sgpr','agpr','lds_bytes','scratch_bytes_lane','vgpr_spills','sgpr_spills')
    equal=all(k in uninstrumented and k in empty and k in probe and uninstrumented[k]==empty[k]==probe[k] for k in fields)
    valid=all(isinstance(x,(float,int)) and 0<x<1e12 for x in (base_ns,empty_ns,probe_ns))
    qualified=bool(equal and valid and complete and abs(probe_ns-base_ns)<=500 and abs(empty_ns-base_ns)<=500)
    return {'status':'QUALIFIED_EXPOSURE_ONLY' if qualified else 'UNQUALIFIED','allocation_equal':equal,'coverage_complete':complete,'perturbation_ns':abs(probe_ns-base_ns) if valid else None,'optimization_claim':False}

def expected_stamps(record,position,phases=None):
    if record=='G06':
        issued={(b,w,p) for b in range(4) for w in range(9) for p in range(3)};blocks=4
    elif record=='G07':
        issued={(b,w,p) for b in range(32) for w in range(2) for p in range(3)}
        issued|={(b,0,3) for b in range(32)};blocks=4
    elif record=='G08':
        active=(position+1+255)//256
        issued={(h*5+part,w,p) for h in range(4) for part in range(active) for w in range(4) for p in range(4)};blocks=20
    else: raise ValueError('unknown diagnostic record')
    if phases is not None:issued={row for row in issued if row[2] in phases}
    return {(b*16+w)*4+p for b,w,p in issued}|{TRACE_NATIVE+b*16+w for b in range(blocks) for w in range(16)}|{TRACE_PLACEMENT+b*16+w for b in range(blocks) for w in range(16)}

def check_stamps(record,position,rows,phases=None):
    indices=[int(x['index']) for x in rows]
    require(len(indices)==len(set(indices)),'duplicate native wave stamp')
    require(set(indices)==expected_stamps(record,position,phases),'missing/unexpected active task/wave landmarks')
    require(all(int(x['tick'])>0 for x in rows),'invalid clock stamp')
    # The final endpoint population includes every physical wave on every task;
    # no task0 shortcut. Native issue phases deliberately are not retirement.
    final=[x for x in rows if TRACE_NATIVE<=int(x['index'])<TRACE_PLACEMENT]
    placement=[x for x in rows if int(x['index'])>=TRACE_PLACEMENT]
    require(all(0<=((int(x['tick'])-1)>>32)<8 for x in placement),'invalid actual XCD ID')
    return {'last_completion_tick':max(int(x['tick']) for x in final),'last_completion_indices':[int(x['index']) for x in final if int(x['tick'])==max(int(y['tick']) for y in final)],'coverage':len(rows),'placements':{str(int(x['index'])-TRACE_PLACEMENT):{'xcd':(int(x['tick'])-1)>>32,'raw_hw_id':(int(x['tick'])-1)&0xffffffff} for x in placement},'clock':'s_memrealtime','issue_vs_retirement':'phases are issue landmarks; final wrapper phases follow original all-wave drain'}

def diagnostic_host():
    text=(HERE/'replay.cpp').read_text()
    text=text.replace('unsigned warmup=8,samples=50,seed=617;','unsigned warmup=1,samples=1,seed=617,probe_mode=1;')
    text=text.replace('if(key=="--warmup")warmup=n;','if(key=="--probe-mode"){need(n<=1,"probe mode");probe_mode=n;}else if(key=="--warmup")warmup=n;')
    marker='  hipStream_t stream;'
    add='''  hipFunction_t reset_trace,export_trace;HIP(hipModuleGetFunction(&reset_trace,mod,"io_trace_reset"));HIP(hipModuleGetFunction(&export_trace,mod,"io_trace_export"));
  void* trace_output=nullptr;HIP(hipMalloc(&trace_output,3072*8));
'''
    require(text.count(marker)==1,'host trace insertion marker');text=text.replace(marker,add+marker,1)
    marker='    auto resetstart=std::chrono::steady_clock::now();'
    add='''    void* reset_args[]={&probe_mode};HIP(hipModuleLaunchKernel(reset_trace,1,1,1,1024,1,1,0,stream,reset_args,nullptr));wait(stream);
'''
    require(text.count(marker)==1,'host reset marker');text=text.replace(marker,add+marker,1)
    marker='    for(auto& b:buffers){std::vector<unsigned char> actual'
    add='''    void* export_args[]={&trace_output};HIP(hipModuleLaunchKernel(export_trace,1,1,1,1024,1,1,0,stream,export_args,nullptr));wait(stream);
    std::vector<unsigned long long> stamps(3072);HIP(hipMemcpy(stamps.data(),trace_output,3072*8,hipMemcpyDeviceToHost));
    for(unsigned i=0;i<stamps.size();++i)if(stamps[i])std::cout<<(i>=2560?"IO_PLACEMENT arm=":"IO_STAMP arm=")<<(arm?"candidate":"control")<<" epoch="<<epoch<<" index="<<i<<(i>=2560?" raw=":" tick=")<<stamps[i]<<std::endl;
'''
    require(text.count(marker)==1,'host export marker');text=text.replace(marker,add+marker,1)
    return text

def prepare(base,out,headers,vllm,phase=None):
    from etx.importer.spec import SOURCES
    from etx.importer.vllm_kernels import _flags
    from etx.importer.ir_import import import_kernel
    from etx.importer.bundle import bundle
    from examples.vllm_llm.bf16_300.build import resources
    import os
    base=Path(base).resolve();original=verify_build(base);record=original['record'];out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    headers=Path(headers);vllm=Path(vllm);llvm=headers/'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin';libs=headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode'
    os.environ.update(ETX_LLVM=str(llvm),ETX_ROCM_HEADERS=str(headers/'inc'),ETX_TORCH_HEADERS=str(headers/'pytorch'))
    (out/'trace.h').write_text(trace_header(record));imports={};commands=[]
    def run(cmd,name):commands.append(list(map(str,cmd)));command(cmd,out,name)
    for arm in ('control','candidate'):
        d=out/arm;d.mkdir();filename='body.hip' if record=='G06' else 'attention.hip'
        (d/filename).write_text(instrument(record,(base/arm/filename).read_text(),phase))
        if record!='G06':
            export={'G07':'attn_reduce','G08':'attn_mfma16_g8'}[record]
            spec=next(k for source in SOURCES for k in source.kernels if k.export==export);srcspec=next(s for s in SOURCES if spec in s.kernels);desc=original['original_imports'][export]
            run([llvm/'clang++','-x','hip','--cuda-device-only','--offload-arch=gfx942','-nogpuinc','-nogpulib','-isystem',headers/'inc','-include','__clang_hip_runtime_wrapper.h','-O3','-gline-tables-only','-std=c++17','-S','-emit-llvm',*_flags(vllm,srcspec),'-include',out/'trace.h',d/filename,'-o',d/'native.ll'],arm+'_ir')
            text,info=import_kernel((d/'native.ll').read_text(),spec.match,'etx_imp_io_'+arm,tuple(desc['block']),1024,spec.lds_cap,spec.inline,desc['slots'])
            imports['io_'+arm]=info.to_json();(out/('io_'+arm+'.ll')).write_text(text)
            run([llvm/'llvm-link',out/('io_'+arm+'.ll'),'-o',out/('io_'+arm+'.bc')],arm+'_assemble')
    if imports:bundle(out,list(imports),libs,out/'imports.bc')
    text,lowering=kernel_source(record,imports)
    text='#include "trace.h"\n'+text+TRACE_EXPORT
    text=text.replace('  ETX_RELEASE_DOMAIN(); __syncthreads();\n}', '  ETX_RELEASE_DOMAIN(); __syncthreads();\n  if((threadIdx.x%64)==0&&io_trace_enabled){const unsigned index=blockIdx.x*16+threadIdx.x/64;io_trace_final_values[index]=__builtin_amdgcn_s_memrealtime();io_trace_placement_values[index]=1ull+((unsigned long long)(__builtin_amdgcn_s_getreg((3<<11)|20)&15)<<32)+(unsigned)__builtin_amdgcn_s_getreg((31<<11)|4);}\n}')
    require(text.count('const unsigned index=blockIdx.x*16')==2,'all-wave completion placement')
    (out/'kernel.hip').write_text(text);(out/'etx_lowering.h').write_text(lowering)
    common=['-x','hip','--cuda-device-only','--offload-arch=gfx942','-O3','-gline-tables-only','-std=c++17','-nogpuinc','--rocm-device-lib-path='+str(libs),'-isystem',headers/'inc','-include','__clang_hip_runtime_wrapper.h','-D__HIP_PLATFORM_AMD__','-DETX_DSL_COH','-I'+str(ROOT/'etx/runtime/include'),'-I'+str(out),'-I'+str(ROOT)]
    if imports:common+=['-Xclang','-mlink-builtin-bitcode','-Xclang',out/'imports.bc']
    run([llvm/'clang++',*common,'-Rpass-analysis=kernel-resource-usage','-S',out/'kernel.hip','-o',out/'kernel.s'],'device_isa')
    run([llvm/'clang++','-target','amdgcn-amd-amdhsa','-mcpu=gfx942','-c',out/'kernel.s','-o',out/'kernel.o'],'device_object');run([llvm/'ld.lld','-shared',out/'kernel.o','-o',out/'kernel.hsaco'],'device_link')
    run([llvm/'llvm-readobj','--notes',out/'kernel.hsaco'],'native_metadata')
    (out/'io_layout.h').write_text((base/'io_layout.h').read_text());(out/'diagnostic_replay.cpp').write_text(diagnostic_host())
    run([llvm/'clang++','-x','c++','-O2','-std=c++17','-D__HIP_PLATFORM_AMD__','-I'+str(headers/'inc'),'-I'+str(out),'-I'+str(ROOT),'-fuse-ld=lld',out/'diagnostic_replay.cpp','-ldl','-o',out/'replay'],'host_link');run([out/'replay','--help'],'host_help')
    rs=[r for r in resources((out/'kernel.s').read_text()) if r['kernel'] in ('io_control','io_candidate')]
    def sgpr(assembly,name):
        found=re.search(r'(?s)\.name:\s*'+re.escape(name)+r'\s.*?\.sgpr_count:\s*(\d+)',assembly)
        require(found is not None,'missing native SGPR metadata');return int(found[1])
    base_resources={arm:dict(original['resources'][arm][0],sgpr=sgpr((base/'kernel.s').read_text(),'io_'+arm)) for arm in ('control','candidate')}
    for entry in rs:entry['sgpr']=sgpr((out/'kernel.s').read_text(),entry['kernel'])
    same={arm:all(next(r for r in rs if r['kernel']=='io_'+arm)[k]==base_resources[arm][k] for k in ('vgpr','sgpr','agpr','lds_bytes','scratch_bytes_lane','vgpr_spills','sgpr_spills')) for arm in ('control','candidate')}
    value={'schema':'etx.issue-order.diagnostic.v1','record':record,'base_build_sha256':sha(base/'identity.json'),'probe_phases':list(range(3) if record=='G06' else range(4)) if phase is None else [phase],'resources':rs,'base_resources':base_resources,'allocation_equal':same,'status':'PREPARED_REQUIRES_RUNTIME_QUALIFICATION' if all(same.values()) else 'PREPARED_ATTRIBUTION_RESOURCE_REJECTED','commands':commands,'gpu_execution':'NOT_RUN','hashes':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},'source_hashes':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},'required_coverage':{str(p):sorted(expected_stamps(record,p,None if phase is None else [phase])) for p in (0,1023,1024,1040)},'clock_scope':'s_memrealtime; issue landmarks vs original post-drain all-physical-wave completion'}
    (out/'identity.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items() if k not in ('hashes','commands','required_coverage')},indent=2));return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--inside',action='store_true',help=argparse.SUPPRESS);p.add_argument('--phase',type=int,choices=range(4));p.add_argument('--headers',type=Path,default=Path.home()/'code/rocm-headers');p.add_argument('--vllm',type=Path,default=Path.home()/'code/vllm');a=p.parse_args()
    if a.inside:prepare(a.base,a.out,a.headers,a.vllm,a.phase);return
    require(not a.out.exists(),'refuse existing diagnostic output')
    def mapped(path):return '/code/'+str(Path(path).resolve().relative_to(Path.home()/'code'))
    cmd=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64','-v',str(Path.home()/'code')+':/code','-v',str(Path.home()/'.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/link-deps/root')+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu','-e','TMPDIR='+mapped(a.out.parent),'-e','PYTHONPATH='+mapped(ROOT),'-e','PYTHONDONTWRITEBYTECODE=1','-w',mapped(ROOT),'etx-rocm22','python3','-m','examples.experiment_program.issue_order.diagnostic','--inside','--base',mapped(a.base),'--out',mapped(a.out),'--headers',mapped(a.headers),'--vllm',mapped(a.vllm)]
    if a.phase is not None:cmd+=['--phase',str(a.phase)]
    subprocess.run(cmd,check=True,timeout=900)
if __name__=='__main__':main()
