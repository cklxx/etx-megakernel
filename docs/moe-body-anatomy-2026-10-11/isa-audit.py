"""CPU-only ISA reproduction; no GPU, network, or model timing.
Run from repo root with .venv/bin/python docs/moe-body-anatomy-2026-10-11/isa-audit.py.
Outputs isolated artifacts under /tmp; source location markers affect debugging only.
All static counts include fallback paths and must not be sold as dynamic latency.
"""
from pathlib import Path
import os, subprocess, json, hashlib, re, collections
from etx.ops import gemv,prims,coh
from etx.codegen.lowering import emit_lowering_header
from etx.machine.model import load_machine
root=Path(__file__).resolve().parents[2]
os.chdir(root)
out=Path('/tmp/etx-moeanat-20261011');out.mkdir(exist_ok=True)
(out/'etx_lowering.h').write_text(emit_lowering_header(load_machine('gfx942')))
# Source line mapping restores original Python raw-string line numbers, including primitive expansion.
orig=Path('etx/ops/gemv.py').read_text().splitlines()
stream=[]
for lineno in range(111,696):
 line=orig[lineno-1]
 stream += [f'#line {107 if "uniform@@" in line else 117} "etx/machine/arch/gfx942.yaml"' if '@@ETX_PRIM:' in line else f'#line {lineno} "etx/ops/gemv.py"',prims.expand(line, None)]
base='''#define ETX_IMPORT_LDS 21120
#define ETX_IMPORT_SLOTS 8
#include "etx_lowering.h"
#include "etx/abi.h"
#include "etx/import.h"
#include "etx/primitives.h"
#include "etx/tile_util.h"
#define DSL_NB 1
#define DSL_K_MAX 4096
#define DSL_GRID_OFF 144
'''+ '\n#line 15 "etx/ops/coh.py"\n'+coh.HELPERS+'\n'+'\n'.join(stream)+'\n'
flags=['-DETX_DSL_COH','-DETX_DSL_NT','-DETX_DSL_VECNORM','-DETX_DSL_WSUM','-DETX_DSL_SCALAR','-DETX_DSL_GU1','-DETX_DSL_RBATCH','-DETX_EMIT_DSL_BIGK=0','-DETX_EMIT_DSL_EXT=0','-DETX_EMIT_FOLD_TOPK=0']
env=dict(os.environ,ETX_DEVLIBS='/Users/ckl/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')
for name,start,end in [('up',727,791),('down',793,833)]:
 body='\n'.join(f'#line {ln} "etx/ops/gemv.py"\n'+orig[ln-1] for ln in range(start,end))
 body += f'\n#line {791 if name == "up" else 833} "etx/ops/gemv.py"\n#endif'
 text=base+f'\nextern "C" __global__ __launch_bounds__(1024,1) void anatomy_{name}(const unsigned char* a, int blk) {{\n blk=etx_uniform(blk); const int nblk = *(const int DSL_ARG_AS*)(uintptr_t)(a + 156); if (blk >= nblk) return;\n const DslArgs d = dsl_args(a); float* xs = (float*)etx_lds_arena;\n'+body+'\n#line 1 "publication"\n ETX_RELEASE_DOMAIN(); __syncthreads();\n}\n'
 p=out/f'{name}.hip';p.write_text(text)
 cmd=['bash','etx/importer/hip_ir_local.sh',str(p),str(out/f'{name}.ll'),'-gline-tables-only','-Ietx/runtime/include','-I'+str(out),'-I.']+flags
 with (out/f'{name}.compile.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['/opt/homebrew/opt/llvm/bin/llc','-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',str(out/f'{name}.ll'),'-o',str(out/f'{name}.s')],check=True)
 print(name, 'compiled', flush=True)

report = ["CPU ISA audit: NOT A GPU MEASUREMENT", "Source: " + subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          "Compiler: " + subprocess.check_output(['/opt/homebrew/opt/llvm/bin/clang++','--version'],text=True).splitlines()[0],
          "Libraries: " + env['ETX_DEVLIBS'], "Flags: " + ' '.join(flags),
          "Wrapper: dynamic args, explicit scalar blk, plain MoE mode determined at runtime; all fallback paths retained.",
          "Static instruction copies across alternatives do not count trips or elapsed time. No model allocation/correctness proof."]
for name in ['up','down']:
    report += ['','FORM '+name]
    for suffix in ['hip','ll','s']:
        p=out/f'{name}.{suffix}'
        report += [str(p)+': sha256='+hashlib.sha256(p.read_bytes()).hexdigest()]
    asm=(out/f'{name}.s').read_text().splitlines()
    counts=collections.defaultdict(collections.Counter); examples={}; loc=''
    for lineno,line in enumerate(asm,1):
        if '.loc' in line: loc=line
        if not re.match(r'^\s+(s_|v_|ds_|global_|buffer_|flat_)[a-z]',line): continue
        m=re.search(r'etx/ops/gemv.py:(\d+):',loc)
        if m:
            source=int(m[1]); op=line.strip().split()[0]
            counts[source][op]+=1; examples.setdefault((source,op),(lineno,line.strip()))
    for source in [183,184,185,186,201,209,220,223,224,240,245,250,255,258,260,267,274,277,396,397,405,407,773,774,775,777,790,831,832,143,152,153,159,160]:
        if source not in counts: continue
        report += [f'gemv.py:{source} static copies: '+json.dumps(dict(counts[source]),sort_keys=True)]
        for op in counts[source]:
            if op.startswith(('ds_','global_')) or op in ['s_barrier','v_exp_f32_e32','v_div_scale_f32','v_div_fmas_f32','v_div_fixup_f32','v_rcp_f32_e32','v_pk_mul_f32']:
                lineno,line=examples[source,op]
                report += [f'  {name}.s:{lineno}: {line}']
    report += ['Metadata (isolated, local compiler):']
    report += [f'  {name}.s:{i}: {v.strip()}' for i,v in enumerate(asm,1) if re.match(r'\s+\.(vgpr_count|sgpr_count|agpr_count|vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size|group_segment_fixed_size):',v)]
report += ['', 'All compile logs retained in /tmp/etx-moeanat-20261011; no GPU launch performed.']
(root/'docs/moe-body-anatomy-2026-10-11/isa-audit.txt').write_text('\n'.join(report)+'\n')
