#!/usr/bin/env bash
# Device-only LLVM IR of a (sliced) HIP source with the real ROCm 7.2.4 headers and the local LLVM, for
# developing the importer without a GPU. The VM uses its own hipcc instead (etx/importer/hip_ir_vm.sh).
#   bash etx/importer/hip_ir_local.sh in.hip out.ll [extra clang flags]
set -euo pipefail
IN=$1; OUT=$2; shift 2
R=${ETX_ROCM_HEADERS:-$HOME/code/rocm-headers}
SHIM=${ETX_IMPORT_SHIM:-$(cd "$(dirname "$0")" && pwd)/shim}
LLVM=${ETX_LLVM:-/opt/homebrew/opt/llvm/bin}
# ETX_DEVLIBS=<dir of ocml.bc, ockl.bc, oclc_*.bc>: link the device libraries (the final megakernel compile;
# imports are compiled without them so their thread/block ids stay __ockl_* calls)
if [ -n "${ETX_DEVLIBS:-}" ]; then LIBS=(--rocm-device-lib-path="$ETX_DEVLIBS"); else LIBS=(-nogpulib); fi
"$LLVM/clang++" -x hip --cuda-device-only --offload-arch=gfx942 "${LIBS[@]}" -nogpuinc -std=c++17 -O3 -S -emit-llvm \
  -D__HIP_PLATFORM_AMD__ -DUSE_ROCM -DNDEBUG -isystem "$R/clang" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h -include etx_cuda2hip.h \
  -I "$SHIM" -isystem "$R/pytorch" $( [ "$(uname)" = Darwin ] && echo -isystem "$SHIM/mac" ) "$@" "$IN" -o "$OUT"
