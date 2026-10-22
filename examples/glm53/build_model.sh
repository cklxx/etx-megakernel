#!/usr/bin/env bash
# Real GLM model device/host build. No fetching or provisioning.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE=${1:?validated staged model directory}
OUT=${2:?new build directory}
PY=${ETX_PYTHON:-.venv/bin/python}
shift 2
LOCAL=0
if [ "${1:-}" = --local ]; then LOCAL=1; shift; fi
# Never allow hidden compiler flags to change a recorded configuration.
for name in HIPCC_COMPILE_FLAGS_APPEND HIPCC_LINK_FLAGS_APPEND CFLAGS CXXFLAGS CPPFLAGS CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH CCC_OVERRIDE_OPTIONS; do
  if [ -n "${!name:-}" ]; then printf 'Refusing unrecorded compiler environment %s\n' "$name" >&2; exit 2; fi
done
if [ -n "${ETX_GLM_OPTIONS:-}" ]; then set -- --options "$ETX_GLM_OPTIONS" "$@"; fi
export ETX_GLM_STAGE="$STAGE"
"$PY" -m examples.glm53.model_build --stage "$STAGE" --out "$OUT" --batch "${ETX_GLM_BATCH:-1}" --gate "${ETX_GLM_GATE:-production}" ${ETX_GLM_FIXTURE:+--allow-fixture} "$@"
if [ "$LOCAL" = 1 ]; then
  for source in "$OUT"/megakernel_d*.hip; do
    d=${source##*/}; d=${d#megakernel_d}; d=${d%.hip}
    printf 'GLM-MODEL-COMPILE rank=%s source=%s\n' "$d" "$source"
    bash etx/importer/hip_ir_local.sh "$source" "$OUT/mk$d.ll" -Ietx/runtime/include -I "$OUT" -I .
    "${ETX_LLVM:-/opt/homebrew/opt/llvm/bin}/llc" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 "$OUT/mk$d.ll" -o "$OUT/mk$d.s"
  done
  R=${ETX_ROCM_HEADERS:-$HOME/code/rocm-headers}
  "${ETX_LLVM:-/opt/homebrew/opt/llvm/bin}/clang++" -x hip --cuda-host-only -fsyntax-only -nogpuinc -nogpulib \
    --offload-arch=gfx942 -std=c++17 -isystem "$R/clang" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h \
    -isystem etx/importer/shim/mac -D__HIP_PLATFORM_AMD__ -Ietx/runtime/include -I "$OUT" -I . -include glm53_options.h examples/glm53/model_host.hip
else
  # One megakernel per participating rank, compiled in parallel, then linked with the host.
  ls "$OUT"/megakernel_d*.hip | xargs -P 8 -I{} sh -c 'hipcc -O3 -std=c++17 --offload-arch=gfx942 -Ietx/runtime/include -I "$1" -I . -include glm53_options.h -c "$2" -o "${2%.hip}.o"' _ "$OUT" {}
  hipcc -O3 -std=c++17 --offload-arch=gfx942 -Ietx/runtime/include -I "$OUT" -I . -include glm53_options.h \
    "$OUT"/megakernel_d*.o examples/glm53/model_host.hip -pthread -o "$OUT/run"
  # Refuse a binary whose kernels cannot all be co-resident (e.g. ROCm 6.4.1: 1 wave/SIMD for a
  # 2-workgroup/CU plan) here, from its code objects, instead of at the device launch.
  "$PY" -m examples.glm53.residency "$OUT" --cus "${ETX_GLM_CUS:-304}" --llvm "${ETX_ROCM_LLVM:-/opt/rocm/llvm/bin}"
fi

"$PY" - "$OUT" "$LOCAL" <<'PYEND'
from pathlib import Path
import sys
from examples.glm53.model_build import finalize
finalize(Path(sys.argv[1]),local=sys.argv[2]=='1')
print('GLM-MODEL-BUILD OK; hardware correctness requires qualification',flush=True)
PYEND
