#!/usr/bin/env bash
# GPU-ready transport build, no download or provisioning. Use --local for a Mac compile check.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?new build output directory}
PY=${ETX_PYTHON:-.venv/bin/python}
"$PY" -m etx compile examples/glm53/boundary.py --arch gfx942 --out "$OUT" --inline-tiles --bind "B=${ETX_GLM_BATCH:-1}"
if [ "${2:-}" = --local ]; then
  for d in 0 1; do
    bash etx/importer/hip_ir_local.sh "$OUT/megakernel_d$d.hip" "$OUT/mk$d.ll" -Ietx/runtime/include -I "$OUT" -I .
    "${ETX_LLVM:-/opt/homebrew/opt/llvm/bin}/llc" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 "$OUT/mk$d.ll" -o "$OUT/mk$d.s"
  done
  R=${ETX_ROCM_HEADERS:-$HOME/code/rocm-headers}
  "${ETX_LLVM:-/opt/homebrew/opt/llvm/bin}/clang++" -x hip --cuda-host-only -fsyntax-only -nogpuinc -nogpulib \
    --offload-arch=gfx942 -std=c++17 -isystem "$R/clang" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h \
    -isystem etx/importer/shim/mac -D__HIP_PLATFORM_AMD__ -Ietx/runtime/include -I "$OUT" -I . examples/glm53/boundary_host.hip
else
  hipcc -O2 -std=c++17 --offload-arch=gfx942 -Ietx/runtime/include -I "$OUT" -I . \
    "$OUT/megakernel_d0.hip" "$OUT/megakernel_d1.hip" examples/glm53/boundary_host.hip -pthread -o "$OUT/run"
fi
echo 'TRANSPORT-BUILD OK; GPU result unmeasured'
