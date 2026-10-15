#!/usr/bin/env bash
# No network, runtime link, device load or GPU execution. Scratch output only.
set -euo pipefail
OUT=${1:?absolute new scratch output directory}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
case "$OUT" in /Users/ckl/.hermes/cache/scratch/*) ;; *) echo 'output must be task scratch' >&2; exit 2;; esac
[ ! -e "$OUT" ] || { echo 'output exists' >&2; exit 2; }
mkdir -p "$OUT"
docker run --rm --pull=never --network=none --platform linux/amd64 \
 -v "$HOME/code:$HOME/code" -v "$HOME/.hermes/cache/scratch:$HOME/.hermes/cache/scratch" \
 -e ROOT="$ROOT" -e OUT="$OUT" -e TASK_HOME="$HOME" etx-rocm22 bash -c '
set -euo pipefail
R=$TASK_HOME/code/rocm-headers
export ETX_ROCM_HEADERS=$R ETX_LLVM=$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin
export ETX_DEVLIBS=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
export ETX_VLLM=$TASK_HOME/code/vllm ETX_TORCH_HEADERS=$R/pytorch
mkdir -p "$OUT/bin"
cat > "$OUT/bin/hipcc" <<EOF
#!/usr/bin/env bash
exec "$ETX_LLVM/clang++" -nogpuinc -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h "\$@"
EOF
chmod +x "$OUT/bin/hipcc"; export PATH=$OUT/bin:$PATH
cd "$ROOT"
python3 -m examples.vllm_llm.attention_early_merge.offline --compiler rocm --out "$OUT/full"
python3 -m examples.vllm_llm.attention_early_merge.build --local --compiler rocm --footprint "$OUT/full" --out "$OUT/probe"
python3 -m examples.vllm_llm.attention_early_merge.build --local --compiler rocm --diagnostic --footprint "$OUT/full" --out "$OUT/diagnostic"
'
