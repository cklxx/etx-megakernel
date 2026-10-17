#!/usr/bin/env bash
# Real GLM model device/host build. No fetching or provisioning.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE=${1:?validated staged model directory}
OUT=${2:?new build directory}
PY=${ETX_PYTHON:-.venv/bin/python}
export ETX_GLM_STAGE="$STAGE"
"$PY" -m examples.glm53.model_build --stage "$STAGE" --out "$OUT" --batch "${ETX_GLM_BATCH:-1}" ${ETX_GLM_FIXTURE:+--allow-fixture}
if [ "${3:-}" = --local ]; then
  for d in 0 1; do
    bash etx/importer/hip_ir_local.sh "$OUT/megakernel_d$d.hip" "$OUT/mk$d.ll" -Ietx/runtime/include -I "$OUT" -I .
    "${ETX_LLVM:-/opt/homebrew/opt/llvm/bin}/llc" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 "$OUT/mk$d.ll" -o "$OUT/mk$d.s"
  done
  R=${ETX_ROCM_HEADERS:-$HOME/code/rocm-headers}
  "${ETX_LLVM:-/opt/homebrew/opt/llvm/bin}/clang++" -x hip --cuda-host-only -fsyntax-only -nogpuinc -nogpulib \
    --offload-arch=gfx942 -std=c++17 -isystem "$R/clang" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h \
    -isystem etx/importer/shim/mac -D__HIP_PLATFORM_AMD__ -Ietx/runtime/include -I "$OUT" -I . examples/glm53/model_host.hip
else
  hipcc -O3 -std=c++17 --offload-arch=gfx942 -Ietx/runtime/include -I "$OUT" -I . \
    "$OUT/megakernel_d0.hip" "$OUT/megakernel_d1.hip" examples/glm53/model_host.hip -pthread -o "$OUT/run"
fi

"$PY" - "$OUT" "${3:-}" <<'PYEND'
from pathlib import Path
import sys
from examples.glm53.model_build import finalize
finalize(Path(sys.argv[1]),local=sys.argv[2]=='--local')
print('GLM-MODEL-BUILD OK; hardware correctness requires qualification',flush=True)
PYEND
