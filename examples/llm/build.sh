#!/usr/bin/env bash
# Build the ETX megakernel for one prepared model. Usage: bash examples/llm/build.sh <packed-dir> [hipcc flags]
set -euo pipefail
DIR="$(cd "${1:?packed model dir from prep.py}" && pwd)"; shift || true
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="$(basename "$DIR")"; OUT="${ETX_OUT:-$ROOT/build/llm/$NAME}"
mkdir -p "$OUT"
cd "$ROOT"
GRAPH="${ETX_LLM_GRAPH:-examples/llm/model.py}"     # examples/llm/model_sliced.py: one slice per XCD
ETX_LLM_CONFIG="$DIR/config.json" "$ROOT/.venv/bin/python" -m etx compile "$GRAPH" --arch gfx942 --out "$OUT" --inline-tiles | grep -E "tasks=|workers/domain"
U=(); [ "${ETX_LLM_THREADS:-256}" -ge 512 ] && U=(-DLLM_GEMV_U=8)   # 8 waves per CU: half the registers per wave (see gemv.h)
hipcc -O3 -std=c++17 --offload-arch=gfx942 "${U[@]}" "$@" -I "$ROOT/etx/runtime/include" -I "$OUT" -I "$ROOT" \
  -Rpass-analysis=kernel-resource-usage "$OUT/megakernel_d0.hip" "$ROOT/examples/llm/host.hip" -o "$OUT/run" 2>&1 \
  | grep -E "Function Name|VGPRs:|ScratchSize|VGPRs Spill|error" | sed -E "s/.*remark: +//; s/ \[-Rpass.*//" | paste - - - - || true
echo "built $OUT/run"
