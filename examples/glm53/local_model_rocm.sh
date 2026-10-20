#!/usr/bin/env bash
# Both real model kernels + host device pass, ROCm 7.2.4 O3, no GPU/network.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?generated model build directory under ~/code}
ROOT=$PWD
OUT=$(cd "$OUT" && pwd)
docker run --rm --pull=never --network=none --platform linux/amd64 \
  -v "$HOME/code:/code" -v "$HOME/code:$HOME/code" -e ROOT="$ROOT" -e OUT="$OUT" etx-rocm22 bash -c '
set -euo pipefail
cd "$ROOT"
R=/code/rocm-headers
RL=$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
for d in 0 1; do
  "$RL/bin/clang++" -x hip --cuda-device-only --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc \
    --rocm-device-lib-path="$DL" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h \
    -D__HIP_PLATFORM_AMD__ -Ietx/runtime/include -I "$OUT" -I . \
    -Rpass-analysis=kernel-resource-usage -S "$OUT/megakernel_d$d.hip" -o "$OUT/rocm$d.s" 2> "$OUT/rocm$d.log"
  echo "GLM-ROCM-DEVICE PASS rank=$d"
done
"$RL/bin/clang++" -x hip --cuda-device-only --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc \
  --rocm-device-lib-path="$DL" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h \
  -D__HIP_PLATFORM_AMD__ -Ietx/runtime/include -I "$OUT" -I . \
  -S examples/glm53/model_host.hip -o "$OUT/rocm-host.s" 2> "$OUT/rocm-host.log"
echo "GLM-ROCM-HOST PASS"
'
