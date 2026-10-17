#!/usr/bin/env bash
# Native LLVM check of an already generated ROCm batch plan/import bundle.
# Useful when the small x86 container cannot hold the optimizer's working set.
# This does NOT substitute for the ROCm compiler preflight before a GPU session.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?directory containing b2/b4/b8 generated plans and imports}
shift
[ "$#" -gt 0 ] || set -- 2 4 8
R=${ETX_ROCM_HEADERS:-$HOME/code/rocm-headers}
export ETX_DEVLIBS=${ETX_DEVLIBS:-$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode}
for b in "$@"; do
  case "$b" in 2|4|8) ;; *) exit 2;; esac
  D=$OUT/b$b
  .venv/bin/python - "$D" "$b" <<'PY'
from pathlib import Path
import json
import math
import os
import sys
import time
from examples.vllm_llm.batch_correctness import clean_env, execute
d = Path(sys.argv[1])
batch = int(sys.argv[2])
# rocm_build_check.sh mounts ~/code as /code. Only remap include paths.
source = (d / 'megakernel_d0.hip').read_text()
(d / 'megakernel_native.hip').write_text(source.replace('#include "/code/', '#include "' + str(Path.home() / 'code') + '/'))
seconds = float(os.environ.get('ETX_QWEN_NATIVE_SECONDS', '1200'))
if not math.isfinite(seconds) or seconds <= 0:
    raise ValueError('native compile budget must be finite and positive')
deadline = time.monotonic() + seconds
env = clean_env(batch)
llvm = Path(env.get('ETX_LLVM', '/opt/homebrew/opt/llvm/bin'))
record = dict(schema=1, batch=batch, status='NOT-COMPLETE', gpu_correctness='unmeasured', deadline_s=seconds)
try:
    execute(['bash', 'etx/importer/hip_ir_local.sh', str(d / 'megakernel_native.hip'), str(d / 'native.ll'),
             '-DETX_DSL_NT', '-DETX_DSL_VECNORM', '-Ietx/runtime/include', '-I', str(d), '-I', '.',
             '-Xclang', '-mlink-builtin-bitcode', '-Xclang', str(d / 'imports.bc')],
            d / 'native-ir.log', env, seconds)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError('native compile budget exhausted')
    execute([str(llvm / 'llc'), '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx942', '-O3',
             str(d / 'native.ll'), '-o', str(d / 'native.s')], d / 'native-isa.log', env, remaining)
    if not all(p.is_file() and p.stat().st_size for p in (d / 'native.ll', d / 'native.s')):
        raise ValueError('compiler did not produce nonempty IR and assembly')
    record['status'] = 'COMPILED'
except Exception as exc:
    record['error'] = str(exc)
    raise
finally:
    (d / 'native-check.json').write_text(json.dumps(record, indent=2) + '\n')
PY
  echo "BATCH-NATIVE OK batch=$b; GPU correctness remains unmeasured"
done
