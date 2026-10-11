#!/usr/bin/env python3
"""Offline whole-plan compile using kept imports, freshly generated plans/tiles.

Run census.py modes first with --out /tmp/etx-isafixes-isa.
No GPU, no importer regeneration, Docker uses an already installed image.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
B=ROOT/'build/qwen300-rocm-on'
OUT=Path('/tmp/etx-isafixes-whole'); OUT.mkdir(exist_ok=True)
headers=Path.home()/'code/rocm-headers'; llvm=headers/'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin'
libs=headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode'
FLAGS=['-DETX_DSL_NT','-DETX_DSL_VECNORM','-DETX_DSL_WSUM','-DETX_DSL_SCALAR','-DETX_DSL_GU1','-DETX_DSL_RBATCH','-DETX_TOPK_KEY32']
for mode in ('baseline','c1','c5','combined'):
    work=OUT/mode; work.mkdir(exist_ok=True)
    env={**os.environ,'ETX_VL_ROUTER_NARROW':'1' if mode in ('c1','combined') else '0'}
    # Use the census's exact source. Only the plan changes to select its router export.
    shutil.copy(Path('/tmp/etx-isafixes-isa')/mode/'dsl_tiles.hip',work/'dsl_tiles.hip')
    imp=json.loads((B/'imp/imports.json').read_text())
    if mode in ('c1','combined'):
        imp['dsl_gemv_router_fold']={**imp['dsl_gemv_fold'],'export':'etx_tile_dsl_gemv_router_fold','kernel':'dsl_gemv_router_fold','dsl':'router'}
    (work/'imports.json').write_text(json.dumps(imp))
    env.update(ETX_VL_IMPORTS=str(work/'imports.json'), ETX_VL_DSL_TILES=str(work/'dsl_tiles.hip'),
               ETX_VL_ADAPTERS=str(B/'imp_tiles.hip'),ETX_LLM_CONFIG=str(ROOT/'examples/llm/configs/qwen3-30b-a3b.json'))
    commands=[[sys.executable,'-m','etx','compile','examples/vllm_llm/model.py','--arch','gfx942','--out',str(work),'--inline-tiles']]
    docker=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64',
            '-v',f'{Path.home()/"code"}:{Path.home()/"code"}:ro','-v',f'{OUT}:{OUT}',
            '-w',str(ROOT),'-e',f'ETX_ROCM_HEADERS={headers}','-e',f'ETX_LLVM={llvm}','-e',f'ETX_DEVLIBS={libs}','etx-rocm22']
    commands += [docker+['bash',str(ROOT/'etx/importer/hip_ir_local.sh'),str(work/'megakernel_d0.hip'),str(work/'whole.ll'),
                          *FLAGS,*(['-DETX_DSL_NORM_HOIST'] if mode in ('c5','combined') else []),
                          '-I'+str(work),'-I'+str(ROOT/'etx/runtime/include'),'-I'+str(ROOT/'etx/runtime'),
                          '-Xclang','-mlink-builtin-bitcode','-Xclang',str(B/'imports.bc')],
                 docker+[str(llvm/'llc'),'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',str(work/'whole.ll'),'-o',str(work/'whole.s')]]
    for n,cmd in enumerate(commands):
        print(f'WHOLE BUILD {mode} stage={n}',flush=True)
        with (OUT/f'{mode}-stage-{n}.log').open('w') as f:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
