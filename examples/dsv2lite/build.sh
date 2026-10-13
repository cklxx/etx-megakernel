#!/usr/bin/env bash
# Build the ETX port of fleet-mi300x. Usage: bash examples/dsv2lite/build.sh <path-to-fleet-mi300x> [extra hipcc flags]
set -euo pipefail
FLEET="${1:?path to fleet-mi300x}"; shift || true
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${ETX_OUT:-$ROOT/build/dsv2lite}"
mkdir -p "$OUT"
# Measurement mode patches only the private source copy. Isolate repetitions
# reuse legacy (layer, kind, coordinate) slots, so reject them rather than
# overwrite the native stamps with an appended task's data.
STAMPS2=0
for flag in "$@"; do
  case "$flag" in -DETX_DSV2_STAMPS2|-DETX_DSV2_STAMPS2=1) STAMPS2=1 ;; esac
done
if [ "$STAMPS2" = 1 ] && [ -n "${ETX_DSV2_ISOLATE:-}" ]; then
  echo "ETX_DSV2_STAMPS2 rejects ETX_DSV2_ISOLATE: repeated tasks share prologue slots" >&2
  exit 2
fi
# ETX_FLEET_PATCH="gemv_noinline ...": build against a patched copy of fleet's sources (examples/dsv2lite/patch_fleet.py)
if [ -n "${ETX_FLEET_PATCH:-}" ] || [ "$STAMPS2" = 1 ]; then
  rm -rf "$OUT/fleet_src"; mkdir -p "$OUT/fleet_src"; cp -r "$FLEET/src/." "$OUT/fleet_src/"
  if [ -n "${ETX_FLEET_PATCH:-}" ]; then
    (cd "$ROOT" && "$ROOT/.venv/bin/python" -m examples.dsv2lite.patch_fleet "$OUT/fleet_src" $ETX_FLEET_PATCH) || exit 1
  fi
  if [ "$STAMPS2" = 1 ]; then
    (cd "$ROOT" && "$ROOT/.venv/bin/python" -m examples.dsv2lite.patch_stamps "$OUT/fleet_src") || exit 1
  fi
  FLEET_SRC="$OUT/fleet_src"
else
  FLEET_SRC="$FLEET/src"
fi
INLINE="${ETX_INLINE_TILES:-1}"
RELAY="${ETX_RELAY_PLAN:-on}"   # on: reserve one CU per XCD like fleet's scheduler CU (3.72 vs 3.78 ms/token with 38 workers); the relay itself runs only with ETX_RELAY=1     # 1: single-TU kernel (tile bodies included and inlinable); 0: separate shim TU
compile_plan() {
  if [ "${ETX_BUILD_VERBOSE:-0}" = 1 ]; then
    "$ROOT/.venv/bin/python" -m etx compile examples/dsv2lite/graph.py --arch gfx942 --out "$OUT" "$@" --relay "$RELAY"
  else
    "$ROOT/.venv/bin/python" -m etx compile examples/dsv2lite/graph.py --arch gfx942 --out "$OUT" "$@" --relay "$RELAY" | grep -E "tasks=|workers/domain"
  fi
}
if [ "$INLINE" = "1" ]; then
  compile_plan --inline-tiles
  SHIM=""
else
  compile_plan
  SHIM="$ROOT/examples/dsv2lite/fleet_shim.hip"
fi
# D2: the plan (ETX_DSV2_COH) and the tiles (lever coherent_loads) must agree; refused before the compiler runs
(cd "$ROOT" && "$ROOT/.venv/bin/python" -m examples.dsv2lite.coh_check "$OUT/megakernel_d0.hip" "$FLEET_SRC") || exit 2
RDC="${ETX_RDC:-0}"                # 0 (default): whole-program device compile, as fleet builds (3.72 vs 3.90 ms/token, scratch 112 -> 0 B/lane); needs INLINE=1
RDCFLAG="-fgpu-rdc"; if [ "$RDC" = "0" ] && [ "$INLINE" = "1" ]; then RDCFLAG=""; fi
hipcc -O3 -std=c++17 --offload-arch=gfx942 $RDCFLAG -DFLEET_NT_WEIGHTS=1 "$@" \
  -I "$FLEET_SRC" -I "$ROOT/etx/runtime/include" -I "$OUT" -I "$ROOT" \
  "$OUT/megakernel_d0.hip" $SHIM "$ROOT/examples/dsv2lite/host.hip" -o "$OUT/run"
echo "built $OUT/run  (run it from $FLEET so build/weights.bin, fleet_cache.bin, golden_tokens.txt resolve)"
