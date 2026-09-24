#!/usr/bin/env bash
# Build the ETX megakernel for one prepared model. Usage: bash examples/llm/build.sh <packed-dir> [hipcc flags]
set -euo pipefail
DIR="$(cd "${1:?packed model dir from prep.py}" && pwd)"; shift || true
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="$(basename "$DIR")"; OUT="${ETX_OUT:-$ROOT/build/llm/$NAME}"
mkdir -p "$OUT"
cd "$ROOT"
ETX_LLM_CONFIG="$DIR/config.json" "$ROOT/.venv/bin/python" -m etx compile examples/llm/model.py --arch gfx942 --out "$OUT" --inline-tiles | grep -E "tasks=|workers/domain"
hipcc -O3 -std=c++17 --offload-arch=gfx942 "$@" -I "$ROOT/etx/runtime/include" -I "$OUT" -I "$ROOT" \
  "$OUT/megakernel_d0.hip" "$ROOT/examples/llm/host.hip" -o "$OUT/run"
echo "built $OUT/run"
