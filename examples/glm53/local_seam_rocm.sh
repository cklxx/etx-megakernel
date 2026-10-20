#!/usr/bin/env bash
# Cached x86 ROCm 7.2.4 toolchain only; never pull an image or use GPU/network.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=${1:?output directory under ~/code}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
case "$OUT" in "$HOME/code/"*) ;; *) printf '%s\n' 'output must be under ~/code' >&2; exit 2;; esac
"${ETX_PYTHON:-.venv/bin/python}" -m etx compile examples/glm53/seam.py --arch gfx942 --out "$OUT" --inline-tiles --bind "B=${ETX_GLM_BATCH:-1}"
docker run --rm --pull=never --network=none --platform linux/amd64 \
  -v "$HOME/code:$HOME/code" -e ROOT="$ROOT" -e OUT="$OUT" etx-rocm22 bash -c '
set -euo pipefail
cd "$ROOT"
R="${ROOT%%/code/*}/code/rocm-headers"
RL=$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
for d in 0 1; do
  "$RL/bin/clang++" -x hip --cuda-device-only --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc \
    --rocm-device-lib-path="$DL" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h \
    -D__HIP_PLATFORM_AMD__ -Ietx/runtime/include -I "$OUT" -I . \
    -S "$OUT/megakernel_d$d.hip" -o "$OUT/rocm_mk$d.s"
done
"$RL/bin/clang++" -x hip --cuda-host-only -fsyntax-only --offload-arch=gfx942 -std=c++17 -nogpuinc -nogpulib \
  -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ \
  -Ietx/runtime/include -I "$OUT" -I . examples/glm53/seam_host.hip
printf "%s\n" "SEAM-ROCM-LOCAL OK; no GPU executed"
'
