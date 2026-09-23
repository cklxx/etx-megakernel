#!/usr/bin/env bash
# Build the ETX port of fleet-mi300x. Usage: bash examples/dsv2lite/build.sh <path-to-fleet-mi300x> [extra hipcc flags]
set -euo pipefail
FLEET="${1:?path to fleet-mi300x}"; shift || true
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/build/dsv2lite"
mkdir -p "$OUT"
"$ROOT/.venv/bin/python" -m etx compile examples/dsv2lite/graph.py --arch gfx942 --out "$OUT" | grep -E "tasks=|workers/domain"
hipcc -O3 -std=c++17 --offload-arch=gfx942 -fgpu-rdc -DFLEET_NT_WEIGHTS=1 "$@" \
  -I "$FLEET/src" -I "$ROOT/etx/runtime/include" -I "$OUT" \
  "$OUT/megakernel_d0.hip" "$ROOT/examples/dsv2lite/fleet_shim.hip" "$ROOT/examples/dsv2lite/host.hip" -o "$OUT/run"
echo "built $OUT/run  (run it from $FLEET so build/weights.bin, fleet_cache.bin, golden_tokens.txt resolve)"
