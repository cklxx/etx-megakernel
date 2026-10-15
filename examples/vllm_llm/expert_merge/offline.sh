#!/usr/bin/env bash
# VM-matching ROCm 7.2.4 / LLVM 22. No network, GPU, load or execution.
set -euo pipefail
OUT=${1:?generated probe directory}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(cd "$OUT" && pwd)"
PY=${ETX_PYTHON:-$ROOT/.venv/bin/python}
"$PY" -m examples.vllm_llm.expert_merge.prepare verify "$OUT"
C_ROOT=/code/${ROOT#"$HOME/code/"}
docker run --rm --pull=never --network=none --platform linux/amd64 \
  -v "$HOME/code:/code" -v "$OUT:$OUT" -e ROOT="$C_ROOT" -e OUT="$OUT" etx-rocm22 bash -c '
set -euo pipefail
cd "$ROOT"
R=/code/rocm-headers
RL=$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
FLAGS=(-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH -DETX_DSL_NORM_HOIST -DETX_DSL_VECNORM -DETX_TOPK_KEY32)
BASE=(-x hip --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path="$DL" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -Ietx/runtime/include -I "$OUT" -I . "${FLAGS[@]}")
"$RL/bin/clang++" --version
"$RL/bin/clang++" "${BASE[@]}" --cuda-device-only -Rpass-analysis=kernel-resource-usage -S "$OUT/megakernel_d0.hip" -o "$OUT/mk-rocm22.s" 2> "$OUT/resource-rocm22.log"
cat "$OUT/resource-rocm22.log"
"$RL/bin/clang++" "${BASE[@]}" --cuda-device-only -S examples/vllm_llm/expert_merge/host.hip -o "$OUT/host-device-rocm22.s"
"$RL/bin/clang++" "${BASE[@]}" --cuda-host-only -c examples/vllm_llm/expert_merge/host.hip -o "$OUT/host-rocm22.o"
echo "OFFLINE device assembly + host device pass + x86 host object OK; link/load/execute not done"
'
