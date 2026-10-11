#!/usr/bin/env python3
"""Scoped IR comparison of the unchanged norm scaling/packing math.

Extracts actual statements from both emitters, lifts their identical operands
into one test kernel, compiles optimized gfx942 IR, then uses llvm-diff.
Does not claim equality of the changed load/control-flow schedule.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from etx.ops import gemv
OUT=Path('/tmp/etx-isafixes-math'); OUT.mkdir(exist_ok=True)
old=types.ModuleType('baseline'); old.__file__=str(ROOT/'etx/ops/gemv.py')
exec(subprocess.check_output(['git','show','9af85ae:etx/ops/gemv.py'],text=True),old.__dict__)

def body(source):
    start=source.index('      const float v0 = dsl_bfr(xs[8 * i + 2 * j] * sv')
    return source[start:source.index('\n    }',start)]

assert body(old.STREAM)==body(gemv.STREAM)
headers=Path.home()/'code/rocm-headers'
env={**os.environ,'ETX_DEVLIBS':str(headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')}
for name,source in [('before',old.STREAM),('after',gemv.STREAM)]:
    hip=OUT/'math.hip'
    hip.write_text('''#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
typedef unsigned dsl_u32x4 __attribute__((ext_vector_type(4)));
static __device__ float dsl_bfr(float v) { return __bfloat162float(__float2bfloat16(v)); }
extern "C" __global__ void k_norm_math(float* xs, const dsl_u32x4* wp, dsl_u32x4* op, float sv) {
  const int i=threadIdx.x; const dsl_u32x4 w=wp[i]; dsl_u32x4 o;
#pragma unroll
  for (int j=0;j<4;++j) {
'''+body(source)+'''\n  }\n  op[i]=o;\n}\n''')
    subprocess.run(['bash',str(ROOT/'etx/importer/hip_ir_local.sh'),str(hip),str(OUT/(name+'.ll'))],env=env,check=True)
llvm=Path(os.environ.get('ETX_LLVM','/opt/homebrew/opt/llvm/bin'))
subprocess.run([str(llvm/'llvm-diff'),str(OUT/'before.ll'),str(OUT/'after.ll'),'k_norm_math'],check=True)
print('Norm scaling/rounding/packing statements SHA256',hashlib.sha256(body(gemv.STREAM).encode()).hexdigest())
print('llvm-diff before.ll after.ll k_norm_math: exit 0 (no differences)')
print('Scope: extracted arithmetic island only; whole hoisted function IR differs.')
