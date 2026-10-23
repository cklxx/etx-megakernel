"""Offline rebuild of retained, certified CPU variants; no device execution.

The retained inputs must match whole-resources.json before reuse. Docker uses
only an installed image, no network or GPU devices. A failed arm stops here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from etx.tools.verify_gaps import digest
from . import paths

FLAGS = ['-DETX_DSL_NT','-DETX_DSL_VECNORM','-DETX_DSL_WSUM','-DETX_DSL_SCALAR',
         '-DETX_DSL_GU1','-DETX_DSL_RBATCH','-DETX_TOPK_KEY32']


def run(out, retained, dry_run=False):
    out = Path(out).absolute(); out.mkdir(parents=True, exist_ok=True)
    certificate = json.loads((paths.ROOT/'docs/isa-fixes-2026-10-11/whole-resources.json').read_text())
    bundle = paths.ROOT/'build/qwen300-rocm-on/imports.bc'
    if digest(bundle) != certificate['import_bundle_sha256']:
        raise ValueError('retained import bundle identity mismatch')
    headers = Path.home()/'code/rocm-headers'
    llvm = headers/'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin'
    libs = headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode'
    docker = ['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64',
        '-v',f'{Path.home()/"code"}:{Path.home()/"code"}:ro', '-v',f'{Path(retained).absolute()}:/retained:ro',
        '-v',f'{out}:/out','-w',str(paths.ROOT),'-e',f'ETX_ROCM_HEADERS={headers}',
        '-e',f'ETX_LLVM={llvm}','-e',f'ETX_DEVLIBS={libs}','etx-rocm22']
    for arm in ('baseline','c1','c5'):
        b = Path(retained)/arm
        for name, expected in certificate['modes'][arm]['sha256'].items():
            if digest(b/name) != expected:
                raise ValueError(f'{arm} retained input mismatch: {name}')
        work=out/arm; work.mkdir(exist_ok=True)
        # Remap only the old worktree/include filenames to identical local
        # bodies. Retain the certified whole plan and import bundle.
        source=(b/'megakernel_d0.hip').read_text()
        for old,new in [('/Users/ckl/code/.wt/etx-isafixes',str(paths.ROOT)),
                        (str(b/'dsl_tiles.hip'),f'/retained/{arm}/dsl_tiles.hip')]:
            source=source.replace(old,new)
        (work/'kernel.hip').write_text(source)
        common = [*FLAGS,*(['-DETX_DSL_NORM_HOIST'] if arm=='c5' else []),
            '-I'+str(paths.ROOT/'etx/runtime/include'),'-I'+str(paths.ROOT/'etx/runtime'),
            f'-I/retained/{arm}','-Xclang','-mlink-builtin-bitcode','-Xclang',str(bundle)]
        cmds = [docker+['bash',str(paths.ROOT/'etx/importer/hip_ir_local.sh'),f'/out/{arm}/kernel.hip',
                 f'/out/{arm}/debug.ll','-gline-tables-only',*common],
            docker+[str(llvm/'llc'),'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',
                 f'/out/{arm}/debug.ll','-o',f'/out/{arm}/debug.s'],
            docker+[str(llvm/'llc'),'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',
                 '-print-before=amdgpu-lower-module-lds','-print-after=amdgpu-lower-module-lds',
                 '-stop-after=amdgpu-lower-module-lds',f'/retained/{arm}/whole.ll','-o',f'/out/{arm}/lds.mir'],
            docker+[str(llvm/'llc'),'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',
                 '-stop-after=prologepilog',f'/retained/{arm}/whole.ll','-o',f'/out/{arm}/allocated.mir']]
        for stage,cmd in enumerate(cmds):
            print(f'VERIFY CPU {arm} stage={stage} '+('DRY_RUN' if dry_run else 'BUILD'),flush=True)
            if not dry_run:
                with (work/f'stage-{stage}.log').open('w') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--retained',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); run(a.out,a.retained,a.dry_run)


if __name__=='__main__': main()
