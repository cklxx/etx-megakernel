#!/usr/bin/env bash
# Build the ETX port of fleet-mi300x. Usage: bash examples/dsv2lite/build.sh <path-to-fleet-mi300x> [extra hipcc flags]
set -euo pipefail
FLEET="${1:?path to fleet-mi300x}"; shift || true
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${ETX_OUT:-$ROOT/build/dsv2lite}"
mkdir -p "$OUT"
INLINE="${ETX_INLINE_TILES:-1}"
RELAY="${ETX_RELAY_PLAN:-auto}"   # auto | on | off (compile-time; ETX_RELAY=0 at run time also disables it)     # 1: single-TU kernel (tile bodies included and inlinable); 0: separate shim TU
if [ "$INLINE" = "1" ]; then
  "$ROOT/.venv/bin/python" -m etx compile examples/dsv2lite/graph.py --arch gfx942 --out "$OUT" --inline-tiles --relay "$RELAY" | grep -E "tasks=|workers/domain"
  SHIM=""
else
  "$ROOT/.venv/bin/python" -m etx compile examples/dsv2lite/graph.py --arch gfx942 --out "$OUT" --relay "$RELAY" | grep -E "tasks=|workers/domain"
  SHIM="$ROOT/examples/dsv2lite/fleet_shim.hip"
fi
RDC="${ETX_RDC:-1}"                # 0: whole-program device compile per TU (fleet builds this way); needs INLINE=1
RDCFLAG="-fgpu-rdc"; if [ "$RDC" = "0" ] && [ "$INLINE" = "1" ]; then RDCFLAG=""; fi
hipcc -O3 -std=c++17 --offload-arch=gfx942 $RDCFLAG -DFLEET_NT_WEIGHTS=1 "$@" \
  -I "$FLEET/src" -I "$ROOT/etx/runtime/include" -I "$OUT" -I "$ROOT" \
  "$OUT/megakernel_d0.hip" $SHIM "$ROOT/examples/dsv2lite/host.hip" -o "$OUT/run"
echo "built $OUT/run  (run it from $FLEET so build/weights.bin, fleet_cache.bin, golden_tokens.txt resolve)"
