#!/usr/bin/env bash
# Every generated rank kernel + host device pass, ROCm 7.2.4 O3, no GPU/network.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?generated model build directory under ~/code}
ROOT=$PWD
OUT=$(cd "$OUT" && pwd)
# The rank count comes from the build, never a constant; refuse kernels it does not list.
RANKS=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]))["devices"];assert type(d) is int and d in (2,4,8),d;print(d)' "$OUT/build-resources.json")
FOUND=$(ls "$OUT" | grep -c '^megakernel_d[0-9]*\.hip$' || true)
[ "$FOUND" = "$RANKS" ] || { echo "GLM-ROCM-DEVICE FAIL build-resources devices=$RANKS but $FOUND megakernel_d*.hip"; exit 1; }
docker run --rm --pull=never --network=none --platform linux/amd64 \
  -v "$HOME/code:/code" -v "$HOME/code:$HOME/code" -e ROOT="$ROOT" -e OUT="$OUT" -e RANKS="$RANKS" etx-rocm22 bash -c '
set -euo pipefail
cd "$ROOT"
R=/code/rocm-headers
RL=$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
for d in $(seq 0 $((RANKS-1))); do
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
echo "GLM-ROCM-DEVICES PASS ranks=$RANKS"
'
