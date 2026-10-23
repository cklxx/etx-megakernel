"""Compile only the exact wave-0 argument hook and event poll primitives.

This is a static specialization of the polling path, not a launched kernel or
an executed trace of the full dispatcher. Full-plan OFF identity is separate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from etx.codegen import emit_lowering_header
from etx.machine import load_machine
from etx.tools.verify_gaps import report
from . import paths


def run(out,dry_run=False):
    out=Path(out).absolute();out.mkdir(parents=True,exist_ok=True)
    (out/'etx_lowering.h').write_text(emit_lowering_header(load_machine('gfx942')))
    prefix='''#define ETX_IMPORT_LDS 21120
#define ETX_IMPORT_SLOTS 8
#include "etx_lowering.h"
#include "etx/abi.h"
#include "etx/primitives.h"
#include "etx/import.h"
extern "C" __global__ void k_polling_wave0(etx_params p, const unsigned char* a) {
'''
    llvm=Path(os.environ.get('ETX_LLVM','/opt/homebrew/opt/llvm/bin'))
    if 'ETX_DEVLIBS' not in os.environ:raise ValueError('set ETX_DEVLIBS explicitly')
    for arm in ('original','isolated'):
        (out/'poll.hip').write_text(prefix+('  etx_imp_touch(a, 160);\n' if arm=='original' else '')+'  etx_wait_DEVICE(p.events, p.ctrl_abort);\n}\n')
        (out/(arm+'.hip')).write_bytes((out/'poll.hip').read_bytes())
        commands=[['bash',str(paths.ROOT/'etx/importer/hip_ir_local.sh'),str(out/'poll.hip'),str(out/(arm+'.ll')),
            '-gline-tables-only','-I'+str(out),'-I'+str(paths.ROOT/'etx/runtime/include'),'-I'+str(paths.ROOT/'etx/runtime')],
            [str(llvm/'llc'),'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',str(out/(arm+'.ll')),'-o',str(out/(arm+'.s'))]]
        for stage,cmd in enumerate(commands):
            print(f'POLLING CPU {arm} stage={stage} '+('DRY_RUN' if dry_run else 'BUILD'),flush=True)
            if not dry_run:
                with (out/f'{arm}-{stage}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        if not dry_run:
            result=report(out/(arm+'.s'),out/(arm+'.ll'))
            (out/(arm+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            print('POLLING '+arm+' MEMORY/WAIT ORDER',flush=True)
            for row in result['kernels']['k_polling_wave0']['memory_order']['sequence']:
                print(f"  {row['index']}: {row['text']} scope={row['wait_scope']}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();run(a.out,a.dry_run)


if __name__=='__main__':main()
