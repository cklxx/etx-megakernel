#!/usr/bin/env bash
# Recompile the preserved local plans with build.sh's exact default device flags.
# No imports/plans regenerated and no GPU access. The ordinary batch_local.sh now
# passes these flags during its initial compile too.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BATCH=${1:?2,4,8}
case "$BATCH" in 2|4|8) ;; *) exit 2;; esac
OUT=$ROOT/build/qwen-batch-rocm/b$BATCH
docker run --rm --pull=never --network=none --platform linux/amd64 \
  -v "$HOME/code:/code" -v "$HOME/code:$HOME/code" \
  -e ROOT="/code/${ROOT#$HOME/code/}" -e OUT="/code/${OUT#$HOME/code/}" etx-rocm22 bash -c '
set -euo pipefail
cd "$ROOT"
R=/code/rocm-headers
RL=$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
"$RL/bin/clang++" -x hip --cuda-device-only --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc \
  --rocm-device-lib-path="$DL" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h \
  -D__HIP_PLATFORM_AMD__ -DETX_DSL_NT -DETX_DSL_VECNORM -Ietx/runtime/include -I "$OUT" -I . \
  -Xclang -mlink-builtin-bitcode -Xclang "$OUT/imports.bc" -Rpass-analysis=kernel-resource-usage \
  -S "$OUT/megakernel_d0.hip" -o "$OUT/mk_exact.s" 2> "$OUT/resource_exact.log"
echo "EXACT-DEVICE OK $OUT flags=-DETX_DSL_NT,-DETX_DSL_VECNORM"
'
