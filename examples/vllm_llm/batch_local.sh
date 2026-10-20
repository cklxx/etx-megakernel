#!/usr/bin/env bash
# No GPU: exact fixed recipe with ROCm's own compiler, plus host syntax checking.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?output directory under ~/code}
shift
[ "$#" -gt 0 ] || set -- 2 4 8
for b in "$@"; do
  # Clear inherited experiments just as the GPU runner does.
  .venv/bin/python - "$b" "$OUT" <<'PY'
import os, subprocess, sys
from pathlib import Path
from examples.vllm_llm.batch_correctness import clean_env
b = int(sys.argv[1])
out = Path(sys.argv[2]).resolve() / f'b{b}'
subprocess.run(['bash', 'examples/vllm_llm/rocm_build_check.sh', 'qwen3-30b-a3b', 'default', str(out)],
               # Match build.sh's two default device-only flags exactly.
               env=dict(clean_env(b), ETX_CFLAGS='-DETX_DSL_NT -DETX_DSL_VECNORM'), check=True)
headers = Path(os.environ.get('ETX_ROCM_HEADERS', str(Path.home() / 'code/rocm-headers')))
llvm = Path(os.environ.get('ETX_LLVM', '/opt/homebrew/opt/llvm/bin'))
subprocess.run([str(llvm / 'clang++'), '-x', 'hip', '--cuda-host-only', '-fsyntax-only',
                '-nogpuinc', '-nogpulib', '--offload-arch=gfx942', '-std=c++17',
                '-isystem', str(headers / 'clang'), '-isystem', str(headers / 'inc'),
                '-include', '__clang_hip_runtime_wrapper.h', '-isystem', 'etx/importer/shim/mac',
                '-D__HIP_PLATFORM_AMD__', '-DETX_VL_CHECK', '-Ietx/runtime/include',
                '-I', str(out), '-I', '.', 'examples/vllm_llm/host.hip'], check=True)
print(f'BATCH-LOCAL OK batch={b} host_and_device_compiled=true', flush=True)
PY
done
