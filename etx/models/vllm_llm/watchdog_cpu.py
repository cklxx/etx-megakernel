"""Compile the opt-in producer ledger and emit a synthetic watchdog snapshot.

CPU only. Local LLVM gfx942 device compilation uses ETX_DEVLIBS. Host syntax
uses the installed offline ROCm image. The synthetic event is NOT a GPU hang.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import subprocess

from etx.ops import gemv
from etx.codegen import emit_kernel,emit_lowering_header,emit_plan_json
from etx.codegen.plan_header import emit_plan_header
from etx.tools import inbody,watchdog
from etx.models.vllm_llm.bf16_300 import dispatcher
from etx.models.vllm_llm.bf16_300.build import current_source,preamble
from .verify_gaps_cpu import FLAGS
from . import paths


def synthetic(out):
    plan=dict(types={'gemv_0':dict(id=0),'consumer_0':dict(id=1)},
        descs=[dict(id=i,type=0 if i<3 else 1,coord=[i],worker=i,domain=0) for i in range(4)],
        events=[dict(name='ready',offset=0,shape=[1])],ev_inst=[['ready',[0],[0,1,2]]],task_in=[[],[],[],[0]])
    words=[inbody.MAGIC,2,4,1,1,10]
    words.extend([100,110,140,0,105,115,170,1,130,140,0,2,0,0,0,0])
    words.extend([0]*4+[0]*3+[120,130,150,160,0,0,0,0])
    words.extend([3,1,1,0,1,2,3,3,0,0,0,150,160,180,190,0,0,1,0])
    (out/'synthetic-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'synthetic.raw').write_bytes(struct.pack('<'+'Q'*len(words),*words))
    result=watchdog.dump(plan,inbody.read_raw(out/'synthetic.raw'))
    if result['events'][0]['unobserved_producer_tasks']!=[2]:raise ValueError('synthetic watchdog failed')
    (out/'synthetic-dump.json').write_text(json.dumps(result,indent=2)+'\n')
    print('WATCHDOG SYNTHETIC '+json.dumps(dict(event='ready',expected=3,remaining=1,
        completed_calls_observed=2,unobserved_producer_tasks=[2],last_stamp_ticks=140,cause=None,removability_evidence=False),sort_keys=True),flush=True)


def run(out,dry_run=False):
    out=Path(out).absolute();out.mkdir(parents=True,exist_ok=True)
    synthetic(out)
    tile=out/'tiles.hip';tile.write_text(preamble(gemv.LDS_BYTES)+current_source())
    plan=dispatcher.graph(tile)
    os.environ['ETX_INBODY_EMIT']='1'
    plan.options=replace(plan.options,inbody_emit=True)
    (out/'kernel.hip').write_text(emit_kernel(plan))
    (out/'etx_lowering.h').write_text(emit_lowering_header(plan.machine))
    (out/'plan_data.h').write_text(emit_plan_header(plan))
    (out/'plan.json').write_text(emit_plan_json(plan))
    (out/'host.hip').write_text('#include "plan_data.h"\n#include "etx/host.h"\nvoid syntax(etx_host& h) { h.write_inbody("snapshot"); h.dump_events(); }\n')
    headers=Path.home()/'code/rocm-headers'
    libs=Path(os.environ.get('ETX_DEVLIBS',str(headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')))
    if not (libs/'ocml.bc').exists():raise ValueError('ETX_DEVLIBS missing')
    llvm=Path(os.environ.get('ETX_LLVM','/opt/homebrew/opt/llvm/bin'))
    native=headers/'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin'
    common=['-DETX_INBODY=1','-DETX_PRODUCER_LEDGER=1',*FLAGS,'-I'+str(out),
        '-I'+str(paths.ROOT/'etx/runtime/include'),'-I'+str(paths.ROOT/'etx/runtime'),'-I'+str(paths.ROOT)]
    env={**os.environ,'ETX_DEVLIBS':str(libs)}
    docker=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64',
        '-v',f'{Path.home()/"code"}:{Path.home()/"code"}:ro','-v',f'{out}:{out}', 'etx-rocm22']
    commands=[['bash',str(paths.ROOT/'etx/importer/hip_ir_local.sh'),str(out/'kernel.hip'),str(out/'ledger.ll'),*common],
        [str(llvm/'llc'),'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',str(out/'ledger.ll'),'-o',str(out/'ledger.s')],
        docker+[str(native/'clang++'),'-x','hip','--cuda-host-only','-fsyntax-only','-nogpuinc','-nogpulib','-std=c++17',
            '-D__HIP_PLATFORM_AMD__','-isystem',str(headers/'inc'),'-isystem',str(headers/'clang'),
            '-include','__clang_hip_runtime_wrapper.h',*common,str(out/'host.hip')]]
    for n,cmd in enumerate(commands):
        print(f'WATCHDOG CPU stage={n} '+('DRY_RUN' if dry_run else 'BUILD'),flush=True)
        if not dry_run:
            with (out/f'stage-{n}.log').open('w') as log:subprocess.run(cmd,cwd=paths.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    print('WATCHDOG CPU '+('DRY_RUN_OK' if dry_run else 'COMPILE_OK'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();run(a.out,a.dry_run)


if __name__=='__main__':main()
