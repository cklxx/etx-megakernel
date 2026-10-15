"""Build the combined C6 probe. No command in --local executes device code.

Outputs are new directories. The original import IR is shared by both arms.
The full-model kept build is a separate prerequisite, not a probe certificate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXPORTS = 'attn_mfma16_g8,attn_reduce'
FLAGS = ['-DETX_DSL_COH', '-DETX_DSL_VECNORM']


def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def clean_env():
    # Keep toolchain and session paths. Drop every inherited ETX experiment lever.
    keep={'ETX_LLVM','ETX_DEVLIBS','ETX_ROCM_HEADERS','ETX_VLLM',
          'ETX_TORCH_HEADERS','ETX_IMPORT_SHIM','ETX_SESS_LOGDIR','ETX_SESS_REMAINING_SECONDS'}
    return {k:v for k,v in os.environ.items()
            if (not k.startswith('ETX_') or k in keep)
            and not k.startswith(('DSL_ABLATE_','C6_'))}


def capture_host():
    source = (HERE.parent/'host.hip').read_text()
    marker = '// ETX_VL_UC builds:'
    source = source.replace(marker, '#include "examples/vllm_llm/attention_early_merge/capture.h"\n'+marker, 1)
    marker = '    int bad = 0, not_imported = 0;'
    source = source.replace(marker, '    C6Capture c6_capture;\n'+marker, 1)
    marker = '      // Same DSL namespace as model.py:src_of'
    hook = r'''
      if (const char* base=getenv("C6_CAPTURE_DIR")) {
        if (H!=2048 || NH!=32 || NKV!=4 || HD!=128 || parts!=5 || NR!=8 || nreq!=1 || !qknr)
          die("C6 requires kept head-local Qwen3-30B-A3B, five partitions, one request");
        if (x.role=="prep") {
          const auto& at=insts.at(x.n+1); const auto& rd=insts.at(x.n+2);
          if (at.exp!="attn_mfma16_g8" || rd.exp!="attn_reduce" || x.exp!="dsl_attnprep" ||
              at.layer!=x.layer || rd.layer!=x.layer) die("C6 constituent order");
          const std::string pl="L"+std::to_string(x.layer)+".";
          c6_capture.begin(base,x.layer,check_pos,{
            {"qkv",qkv,(size_t)NQKV*2}, {"qw",W(pl+"qn"),(size_t)HD*2},
            {"kw",W(pl+"kn"),(size_t)HD*2}, {"cos_sin",W("cos_sin"),m.t.at("cos_sin").second*2},
            {"positions",positions,8}, {"slot_mapping",slot_mapping,8},
            {"kc",kc[x.layer],cache_elems*2}, {"vc",vc[x.layer],cache_elems*2},
            {"block_tables",block_tables,(size_t)MAXBLK*4}, {"seq_lens",seq_lens,4}, {"kv_scale",kv_scale,4},
            {"exp_sums",exp_sums,(size_t)NH*parts*4}, {"max_logits",max_logits,(size_t)NH*parts*4},
            {"tmp_out",tmp_out,(size_t)NH*parts*HD*2}, {"attn_out",attn_out,(size_t)NH*HD*2}},
            {{x.exp,ab.data()+x.off},{at.exp,ab.data()+at.off},{rd.exp,ab.data()+rd.off}});
        }
      }
'''
    if source.count(marker) != 1:
        raise ValueError('capture host source changed')
    source = source.replace(marker, hook+marker, 1)
    marker = '      ++checked;'
    if source.count(marker) != 1:
        raise ValueError('capture completion hook changed')
    return source.replace(marker, '      if (getenv("C6_CAPTURE_DIR") && x.role=="reduce") c6_capture.finish();\n'+marker, 1)


def hybrid_host():
    source=capture_host()
    marker='int main(int argc, char** argv) {'
    source=source.replace(marker,'#include "examples/vllm_llm/attention_early_merge/hybrid.h"\n'+marker,1)
    source=source.replace('  hst.setup();','  hst.setup();\n  C6Hybrid c6_hybrid; c6_hybrid.setup(hst);',1)
    marker='      if (unfused) { if (!hst.run_unfused())'
    if source.count(marker)!=1: raise ValueError('hybrid step hook changed')
    return source.replace(marker,'      if (c6_hybrid.enabled) { if (!c6_hybrid.step(hst,insts)) die("C6 hybrid step failed"); }\n      else if (unfused) { if (!hst.run_unfused())',1)


def retained_source(source):
    pattern=r'extern "C" __global__ void __launch_bounds__\([^\n]*\) etx_megakernel_d0\(etx_params p\) \{'
    source,count=re.subn(pattern,'static __device__ __attribute__((always_inline)) void c6_kept_dispatch(etx_params p) {',source)
    if count!=2: raise ValueError('kept dispatcher signatures changed')
    return source+'\n'+(HERE/'probe.hip').read_text()+r'''
// The non-null branch is not a runtime mode. It preserves the actual kept
// dispatcher in this entry's compiler allocation. The host always passes null.
extern "C" __global__ __launch_bounds__(1024, 1) void c6_probe(C6Args a) {
  if (a.retained) c6_kept_dispatch(*a.retained);
  else c6_run(a);
}
'''

def prepare(out, local=False, diagnostic=False, compiler='local', footprint=None):
    from etx.codegen.lowering import emit_lowering_header
    from etx.machine.model import load_machine
    from etx.ops import attnprep, coh
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    env=clean_env()
    headers=Path(env.get('ETX_ROCM_HEADERS', str(Path.home()/'code/rocm-headers')))
    llvm=Path(env.get('ETX_LLVM','/opt/homebrew/opt/llvm/bin' if local else '/opt/rocm/llvm/bin'))
    libs=Path(env.get('ETX_DEVLIBS',str(headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')))
    if not local and 'ETX_DEVLIBS' not in env:
        candidates=list(Path('/opt').glob('rocm*/**/ocml.bc'))
        if not candidates: raise ValueError('missing ROCm device libraries: set ETX_DEVLIBS')
        libs=candidates[0].parent
    env.update(ETX_ROCM_HEADERS=str(headers),ETX_LLVM=str(llvm),ETX_DEVLIBS=str(libs))
    commands=[]
    def run(cmd,name):
        cmd=list(map(str,cmd)); commands.append(cmd)
        print('STAGE',name,'evidence',out/(name+'.log'),flush=True)
        with (out/(name+'.log')).open('w') as f:
            rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        (out/(name+'.rc')).write_text(str(rc)+'\n')
        if rc: raise RuntimeError(f'{name} exit={rc}: {out/(name+".log")}')
    retained=None
    if footprint:
        from .evidence import identity
        fp=Path(footprint).resolve()
        identity(fp,'offline.json',runnable=False)
        shutil.copytree(fp/'imp',out/'imp')
        for name in ('imports.bc','imports.h'):
            shutil.copy2(fp/name,out/name)
        source=retained_source((fp/'control/megakernel_d0.hip').read_text())
        retained=dict(directory=str(fp),identity_sha=sha(fp/'offline.json'),
                      headers={str(p.relative_to(fp)):sha(p) for p in sorted(fp.rglob('*')) if p.is_file() and p.suffix in ('.h','.inc')})
    else:
        (out/'etx_lowering.h').write_text(emit_lowering_header(load_machine('gfx942')))
        args=[sys.executable,'-m','etx.importer.vllm_kernels','--vllm',env.get('ETX_VLLM',str(Path.home()/'code/vllm')),
              '--out',out/'imp','--only',EXPORTS,'--slots','attn_mfma16_g8=1,attn_reduce=8']
        if not local or compiler=='rocm': args+=['--compiler','hipcc']
        run(args,'import')
        # Append the original DSL prep layout and body, without changing either import.
        manifest=json.loads((out/'imp/imports.json').read_text()); manifest.update(attnprep.import_entries())
        (out/'imp/imports.json').write_text(json.dumps(manifest,indent=2)+'\n')
        run([sys.executable,'-m','etx.importer.adapter',out/'imp/imports.json','--tiles',out/'adapters.hip','--header',out/'imports.h'],'adapters')
        run([sys.executable,'-m','etx.importer.bundle','--imports',out/'imp','--exports',EXPORTS,'--devlibs',libs,'-o',out/'imports.bc'],'bundle')
        source=('#include "etx_lowering.h"\n#include "etx/primitives.h"\n#include "etx/tile_util.h"\n'
                '#include "adapters.hip"\n'+coh.HELPERS+attnprep.TILE+(HERE/'probe.hip').read_text())
    (out/'probe.hip').write_text(source)
    (out/'capture_host.hip').write_text(capture_host())
    (out/'hybrid_host.hip').write_text(hybrid_host())
    common=['-I'+str(ROOT/'etx/runtime/include'),'-I'+str(out),'-I'+str(ROOT),*[f for f in FLAGS if not retained or f!='-DETX_DSL_COH'],
            '-Xclang','-mlink-builtin-bitcode','-Xclang',str(out/'imports.bc')]
    if retained:
        from examples.vllm_llm.kept_build import BASE_FLAGS
        common+=['-DC6_RETAINED_ENVELOPE','-I'+str(fp),'-I'+str(fp/'control'),
                 *BASE_FLAGS.split(),'-DETX_DSL_NT','-DETX_TOPK_KEY32']
    if diagnostic: common+=['-DC6_DIAGNOSTIC']
    if local:
        if compiler=='local':
            run(['bash',ROOT/'etx/importer/hip_ir_local.sh',out/'probe.hip',out/'probe.ll',*common],'device_ir')
        else:
            run(['hipcc','--cuda-device-only','-S','-emit-llvm','-O3','-std=c++17','--offload-arch=gfx942',
                 '--rocm-device-lib-path='+str(libs),out/'probe.hip','-o',out/'probe.ll',*common],'device_ir')
        run([llvm/'llc','-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',out/'probe.ll','-o',out/'probe.s'],'device_isa')
        syntax=[llvm/'clang++','-x','hip','--cuda-host-only','-fsyntax-only','-nogpuinc','-nogpulib',
                '--offload-arch=gfx942','-std=c++17','-isystem',headers/'clang','-isystem',headers/'inc',
                '-include','__clang_hip_runtime_wrapper.h','-D__HIP_PLATFORM_AMD__',
                '-I'+str(ROOT/'etx/runtime/include'),'-I'+str(out),'-I'+str(ROOT)]
        if retained: syntax+=['-DC6_RETAINED_ENVELOPE','-I'+str(fp),'-I'+str(fp/'control')]
        if sys.platform=='darwin': syntax+=['-isystem',ROOT/'etx/importer/shim/mac']
        run([*syntax,HERE/'launcher.hip'],'host_syntax')
    else:
        run(['hipcc','-O3','-std=c++17','--offload-arch=gfx942',out/'probe.hip',HERE/'launcher.hip',
             '-o',out/'run',*common,'-Rpass-analysis=kernel-resource-usage'],'link')
        run(['hipcc','--cuda-device-only','-S','-emit-llvm','-O3','-std=c++17','--offload-arch=gfx942',
             out/'probe.hip','-o',out/'probe.ll',*common],'device_ir')
        run([llvm/'llc','-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',out/'probe.ll','-o',out/'probe.s'],'device_isa')
    from examples.vllm_llm.bf16_300.build import resources
    from examples.vllm_llm.kept_build import source_hashes
    files=['probe.hip','probe.ll','probe.s','imports.bc','imports.h','adapters.hip','capture_host.hip','hybrid_host.hip','imp/imports.json','imp/attention.ll','imp/attn_reduce.ll']
    # Importer uses export names for its files. Record every retained IR file.
    files=[f for f in files if (out/f).is_file()]
    files+=['launcher.hip'] if (out/'launcher.hip').is_file() else []
    if not local: files+=['run']
    identity=dict(schema='etx.c6.build.v1',runnable=not local,diagnostic=diagnostic,
        retained_footprint=retained,compiler=subprocess.check_output([llvm/'clang++','--version'],text=True),commands=commands,
        resources=resources((out/'probe.s').read_text()),
        hashes={f:sha(out/f) for f in files},sources=source_hashes(),
        imports={p.name:sha(p) for p in sorted((out/'imp').glob('*.ll'))},
        gpu_load='NOT_RUN',gpu_execution='NOT_RUN',full_megakernel_resources='PENDING',timing_eligible=False)
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--local',action='store_true')
    p.add_argument('--footprint',type=Path,help='fresh offline.py output; required for timing'); p.add_argument('--diagnostic',action='store_true'); p.add_argument('--compiler',choices=['local','rocm'],default='local')
    a=p.parse_args(); print(json.dumps(prepare(a.out,a.local,a.diagnostic,a.compiler,a.footprint),indent=2))


if __name__=='__main__': main()
