"""Build runnable original/probe modules and replay host; CPU local compile too.

--local compiles gfx942 IR/ISA and syntax-checks the host, without HIP runtime,
GPU, network, or a runnable code object. Omit --local on the VM compiler BEFORE
the measurement window. Keep build outputs per sampling tuple/pair.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from . import paths
from .build import FLAGS, preamble, control_source, resources
from .launch_source import candidate_source
from .pack import sha
from .source import PAIRS
from etx.codegen.lowering import emit_lowering_header
from etx.machine.model import load_machine
from etx.ops import gemv


def prepare(out, pair, task, wave, row, local=False):
    if gemv.DSL_NB != 1 or gemv.DSL_EXT or gemv.ARG_BYTES != 160:
        raise ValueError('probe build requires original bf16 batch-one 160-byte DSL layout')
    if pair not in PAIRS or task < 0 or not 0 <= wave < 16 or not 0 <= row < 64:
        raise ValueError('invalid probe selector')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'etx_lowering.h').write_text(emit_lowering_header(load_machine('gfx942')))
    sources = dict(original=control_source(), candidate=preamble(gemv.LDS_BYTES)+candidate_source(pair,task,wave,row))
    llvm = Path(os.environ.get('ETX_LLVM', '/opt/homebrew/opt/llvm/bin' if local else '/opt/rocm/llvm/bin'))
    headers = Path(os.environ.get('ETX_ROCM_HEADERS', str(Path.home()/'code/rocm-headers')))
    common = ['-I'+str(paths.ROOT/'etx/runtime/include'), '-I'+str(out), '-I'+str(paths.ROOT)] + FLAGS
    reports, commands = {}, []
    def run(cmd, name):
        commands.append(list(map(str,cmd)))
        print('CPU BUILD' if local else 'BUILD', name, flush=True)
        with (out/(name+'.log')).open('w') as f:
            subprocess.run(list(map(str,cmd)), cwd=paths.ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    for arm, source in sources.items():
        hip = out/(arm+'.hip'); hip.write_text(source)
        if local:
            run(['bash', paths.ROOT/'etx/importer/hip_ir_local.sh',hip,out/(arm+'.ll')]+common, arm+'_ir')
        else:
            run(['hipcc','-O3','-std=c++17','--offload-arch=gfx942','--genco',hip,'-o',out/(arm+'.hsaco')]+common, arm+'_module')
            # Same compiler, source, optimization and flags for resource report.
            run(['hipcc','-O3','-std=c++17','--offload-arch=gfx942','--cuda-device-only','-S','-emit-llvm',hip,'-o',out/(arm+'.ll')]+common, arm+'_ir')
        run([llvm/'llc','-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx942','-O3',out/(arm+'.ll'),'-o',out/(arm+'.s')], arm+'_isa')
        reports[arm] = resources((out/(arm+'.s')).read_text())
    host = paths.HERE/'launcher.hip'
    if local:
        run([llvm/'clang++','-x','hip','--cuda-host-only','-fsyntax-only','-nogpuinc','-nogpulib',
             '--offload-arch=gfx942','-std=c++17','-isystem',headers/'clang','-isystem',headers/'inc',
             '-include','__clang_hip_runtime_wrapper.h','-isystem',paths.ROOT/'etx/importer/shim/mac','-D__HIP_PLATFORM_AMD__',host], 'host_syntax')
    else:
        run(['hipcc','-O2','-std=c++17',host,'-o',out/'launcher'], 'host')
    files = ['original.hip','candidate.hip','original.ll','candidate.ll','original.s','candidate.s']
    if not local: files += ['original.hsaco','candidate.hsaco','launcher']
    identity = dict(version=1,pair=pair,sampling=dict(task=task,wave=wave,row=row),
        capture_math='bf16_batch1_moe_mode0',resources=reports,commands=commands,
        compiler=subprocess.check_output([llvm/'clang++','--version'],text=True),
        hashes={name:sha(out/name) for name in files}, host_source_sha256=sha(host),
        runnable=not local,gpu_checks='NOT_RUN',timing_eligible=False,
        resource_metadata_basis='same-source IR/llc; inspect module resource attributes on device too')
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pair',choices=PAIRS,required=True)
    p.add_argument('--task',type=int,required=True)
    p.add_argument('--wave',type=int,required=True)
    p.add_argument('--row',type=int,required=True)
    p.add_argument('--local',action='store_true')
    a=p.parse_args()
    print(json.dumps(prepare(a.out,a.pair,a.task,a.wave,a.row,a.local),indent=2))


if __name__=='__main__': main()
