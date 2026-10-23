"""Full-plan OFF .text identity and ON polling-path ISA; CPU only."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import types

from etx.codegen import emit_kernel, emit_lowering_header
from etx.contracts.compiler import _load_example
from etx.passes import compile_graph, PassOptions
from etx.tools.verify_gaps import digest, report
from .verify_gaps_cpu import ROOT, FLAGS


def run(out, retained, dry_run=False):
    out=Path(out).absolute();out.mkdir(parents=True,exist_ok=True)
    retained=Path(retained).absolute(); b=retained/'baseline'
    os.environ.update(ETX_VL_IMPORTS=str(b/'imports.json'),ETX_VL_DSL_TILES=str(b/'dsl_tiles.hip'),
        ETX_VL_ADAPTERS=str(ROOT/'build/qwen300-rocm-on/imp_tiles.hip'),ETX_VL_ROUTER_NARROW='0',
        ETX_LLM_CONFIG=str(ROOT/'examples/llm/configs/qwen3-30b-a3b.json'))
    os.environ.pop('ETX_INBODY_EMIT',None)
    module=_load_example(str(ROOT/'examples/vllm_llm/model.py'))
    plan=compile_graph(module.build(),'gfx942',{},options=PassOptions(inline_tiles=True))
    baseline=types.ModuleType('etx.codegen.baseline');baseline.__package__='etx.codegen'
    exec(compile(subprocess.check_output(['git','show','9f91f3a:etx/codegen/kernel.py'],cwd=ROOT,text=True),'<baseline>','exec'),baseline.__dict__)
    original=baseline.emit_kernel(plan)
    os.environ['ETX_PREFETCH_ISOLATE_POLLER']='0';plan.options=replace(plan.options,prefetch_isolate_poller=False);off=emit_kernel(plan)
    os.environ['ETX_PREFETCH_ISOLATE_POLLER']='1';plan.options=replace(plan.options,prefetch_isolate_poller=True);on=emit_kernel(plan)
    if off!=original:raise ValueError('OFF emitter differs from starting commit')
    sources={'original':original,'off':off,'on':on}
    (out/'etx_lowering.h').write_text(emit_lowering_header(plan.machine))
    print(f'ISOLATION source_off_identical=True tasks={len(plan.tasks)} workers={plan.workers_per_device()}',flush=True)
    headers=Path.home()/'code/rocm-headers';llvm=headers/'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin'
    libs=headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode'
    docker=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64',
        '-v',f'{Path.home()/"code"}:{Path.home()/"code"}:ro', '-v',f'{retained}:{retained}:ro',
        '-v',f'{out}:/out','-w',str(ROOT),'-e',f'ETX_ROCM_HEADERS={headers}',
        '-e',f'ETX_LLVM={llvm}','-e',f'ETX_DEVLIBS={libs}','etx-rocm22']
    for arm,source in sources.items():
        (out/'kernel.hip').write_text(source);(out/(arm+'.hip')).write_text(source)
        commands=[docker+['bash',str(ROOT/'etx/importer/hip_ir_local.sh'),'/out/kernel.hip',f'/out/{arm}.ll',
            '-gline-tables-only',*FLAGS,'-I/out','-I'+str(ROOT/'etx/runtime/include'),'-I'+str(ROOT/'etx/runtime'),
            '-Xclang','-mlink-builtin-bitcode','-Xclang',str(ROOT/'build/qwen300-rocm-on/imports.bc')],
            docker+[str(llvm/'llc'),'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',f'/out/{arm}.ll','-o',f'/out/{arm}.s'],
            docker+[str(llvm/'llvm-mc'),'-triple=amdgcn-amd-amdhsa','-mcpu=gfx942','-filetype=obj',f'/out/{arm}.s','-o',f'/out/{arm}.o'],
            docker+[str(llvm/'llvm-objcopy'),'-O','binary','--only-section=.text',f'/out/{arm}.o',f'/out/{arm}.text.bin']]
        for stage,cmd in enumerate(commands):
            print(f'ISOLATION CPU {arm} stage={stage} '+('DRY_RUN' if dry_run else 'BUILD'),flush=True)
            if not dry_run:
                with (out/f'{arm}-{stage}.log').open('w') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    if dry_run:return
    identity=(out/'original.text.bin').read_bytes()==(out/'off.text.bin').read_bytes()
    if not identity:raise ValueError('OFF device .text mismatch')
    evidence=dict(start_commit='9f91f3a',off_source_identical=True,off_text_identical=identity,
        text_sha256={arm:digest(out/(arm+'.text.bin')) for arm in sources},
        resources={arm:{k:v['resources'] for k,v in report(out/(arm+'.s'))['kernels'].items()} for arm in sources},
        flag='ETX_PREFETCH_ISOLATE_POLLER=1 at emission',gpu_validation='NOT_RUN_CPU_ONLY',
        keep_threshold=dict(round_trip_us_lt=2.0,total_gain_ms_per_step_ge=0.030))
    (out/'identity.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('ISOLATION '+json.dumps(evidence,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--retained',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();run(a.out,a.retained,a.dry_run)


if __name__=='__main__':main()
