"""Offline ROCm 7.2.4 / LLVM22 builds, device link and native replay host link."""
import hashlib,json,subprocess
from pathlib import Path
from etx.codegen.lowering import emit_lowering_header
from etx.machine.model import load_machine
from etx.ops import gemv
from examples.vllm_llm.bf16_300.build import preamble,resources,FLAGS
from .source import VARIANTS,translation_unit
ROOT=Path(__file__).resolve().parents[3]
FLAGS=FLAGS+['-DETX_DSL_NORM_HOIST','-DETX_EMIT_DSL_TOUCH=1']


def prepare(out):
    if gemv.DSL_NB!=1 or gemv.DSL_EXT:raise ValueError('fixed Qwen B1 non-EXT source generation only')
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    from .capture import checking_host
    baseline=out/'original-host.hip'
    baseline.write_bytes(subprocess.check_output(['git','show','b573664faa4e7988dff353bc03aca4ce8561a729:examples/vllm_llm/host.hip'],cwd=ROOT))
    checking_host(baseline,out/'capture_host.hip')
    (out/'etx_lowering.h').write_text(emit_lowering_header(load_machine('gfx942')))
    for v in VARIANTS: (out/(v+'.hip')).write_text(translation_unit(v,preamble(gemv.LDS_BYTES)))
    (out/'compile.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
R=/code/rocm-headers
RL=$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
cd "$ROOT"
mkdir -p "$OUT/tmp"
export TMPDIR="$OUT/tmp"
for V in control g11_control g09 g10_up g10_down g11_probe g11_empty; do
  echo "COMPILE $V gfx942 ROCm22"
  "$RL/bin/clang++" -x hip --cuda-device-only --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path="$DL" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -Ietx/runtime/include -I "$OUT" -I . '''+' '.join(FLAGS)+''' -Rpass-analysis=kernel-resource-usage -S "$OUT/$V.hip" -o "$OUT/$V.s" 2> "$OUT/$V.compile.log"
done
python3 -m examples.experiment_program.row_ops.padding "$OUT" > "$OUT/allocation-matching.log"
for V in control g11_control g09 g10_up g10_down g11_probe g11_empty; do
  "$RL/bin/clang++" -target amdgcn-amd-amdhsa -mcpu=gfx942 -c "$OUT/$V.s" -o "$OUT/$V.o"
  "$RL/bin/ld.lld" -shared "$OUT/$V.o" -o "$OUT/$V.hsaco"
  "$RL/bin/llvm-objdump" --amdhsa-code-object-metadata "$OUT/$V.hsaco" > "$OUT/$V.metadata" 2>&1 || "$RL/bin/llvm-readobj" --notes "$OUT/$V.hsaco" > "$OUT/$V.metadata"
  "$RL/bin/llvm-objdump" -d "$OUT/$V.hsaco" > "$OUT/$V.isa"
done
"$RL/bin/clang++" -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ -isystem "$R/inc" -I . examples/experiment_program/row_ops/replay.cpp -ldl -o "$OUT/replay" > "$OUT/host.compile.log" 2>&1
echo "HOST-LINK COMPLETE replay"
CLOSURE=/code/etx/build/rocm22/final_kept
"$RL/bin/clang++" -x hip --cuda-host-only --offload-arch=gfx942 -O2 -std=c++17 -nogpuinc -nogpulib -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -DETX_VL_CHECK -Ietx/runtime/include -I "$CLOSURE" -I . -c "$OUT/capture_host.hip" -o "$OUT/capture_host.o" > "$OUT/capture_host.compile.log" 2>&1
echo "CAPTURE-HOST genuine original closure compile complete"
''')
    return out


def compile(out):
    out=prepare(out)
    def mount(p):
        return '/code/'+str(p.relative_to(Path.home()/'code'))
    command=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64','-v',str(Path.home()/'code')+':/code','-v','/Users/ckl/.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/link-deps/root:/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu','-e','ROOT='+mount(ROOT),'-e','OUT='+mount(out),'etx-rocm22','bash',mount(out/'compile.sh')]
    with (out/'build.log').open('w') as f: subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=300)
    return seal(out)


def seal(out):
    out=Path(out)
    for v in VARIANTS:
        expected=translation_unit(v,preamble(gemv.LDS_BYTES))
        if (out/(v+'.hip')).read_text()!=expected:raise ValueError('generated source changed since compile')
    if not (out/'replay').is_file():raise ValueError('native linked replay missing')
    native={}
    for v in VARIANTS:
        # Compiler remarks and complete ISA/notes retained even if metadata parser
        # cannot classify a new compiler format. This is no GPU residency claim.
        note=(out/(v+'.metadata')).read_text()
        native[v]=resources(note)
        from .elf import descriptors,allocation_signature
        actual=descriptors(out/(v+'.hsaco'))
        for r in native[v]:
            import re
            block=re.search(r'(?s)\.name:\s*'+re.escape(r['kernel'])+r'.*?\.sgpr_count:\s*(\d+)',note)
            r['sgpr']=int(block[1]) if block else None
            r['descriptor']=actual[r['kernel']]
            r['physical_signature']=allocation_signature(actual[r['kernel']],r['agpr'])
    files=[p for p in out.iterdir() if p.is_file() and p.name!='identity.json']
    g09=next(r for r in native['g09'] if r['kernel']=='row_dsl_moe_up')
    control=next(r for r in native['control'] if r['kernel']=='row_dsl_moe_up')
    g09_admitted=not(g09['vgpr_spills'] or g09['sgpr_spills'] or g09['scratch_bytes_lane']) and g09['static_blocks_per_cu']>=control['static_blocks_per_cu']
    tools=[Path.home()/'code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin'/n for n in ('clang++','lld','llc')]
    tool_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tools}
    closure=Path.home()/'code/etx/build/rocm22/final_kept'
    closure_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [closure/'plan_data.h',closure/'imports.h',closure/'plan.json']}
    identity=dict(capture_compile_closure=closure_hashes,capture_closure_certificate_transferred=False,tool_hashes=tool_hashes,math_flags=FLAGS,g11_allocation_matching=json.loads((out/'g11-allocation-matching.json').read_text()) if (out/'g11-allocation-matching.json').exists() else None,g09_resource_admitted=g09_admitted,compiler='offline etx-rocm22 ROCm7.2.4 LLVM22 gfx942',resources=native,
                  hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},gpu_executed=False)
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return identity
