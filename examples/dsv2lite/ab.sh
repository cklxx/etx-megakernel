#!/usr/bin/env bash
# A/B every M3 lever on one VM. Usage: bash examples/dsv2lite/ab.sh <fleet-mi300x> [baseline-git-rev]
# Builds each variant into build/ab/<name>, runs 32 tokens x 2 from the fleet dir, prints one line per variant.
set -uo pipefail
FLEET="${1:?path to fleet-mi300x}"; BASE_REV="${2:-3cabf9b}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AB="$ROOT/build/ab"; mkdir -p "$AB"
ARGS=(--tokens 32 --context 1024 --repeat 2)
build() {   # name, env..., -- hipcc flags...
  local name="$1"; shift
  local envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
  env "${envs[@]}" ETX_OUT="$AB/$name" bash "$ROOT/examples/dsv2lite/build.sh" "$FLEET" -Rpass-analysis=kernel-resource-usage "$@" \
      > "$AB/$name.build.log" 2>&1 || { echo "$name: BUILD FAILED (see $AB/$name.build.log)"; return 1; }
  grep -A12 "Function Name: etx_megakernel_d0" "$AB/$name.build.log" | grep -E "VGPRs:|AGPRs|ScratchSize|Occupancy" | tr -s ' ' | sed "s/^/  $name /" | head -4
}
run() {     # label, dir, env...
  local label="$1" dir="$2"; shift 2
  (cd "$FLEET" && env "$@" "$dir/run" "${ARGS[@]}") > "$AB/$label.run.log" 2>&1
  local med match layers
  med=$(grep -o "median [0-9.]* ms" "$AB/$label.run.log" | awk '{print $2}')
  match=$(grep -c " ok$" "$AB/$label.run.log")
  layers=$(grep -o "consecutive layers within the gate: [0-9]* of [0-9]*" "$AB/$label.run.log" | awk '{print $6"/"$8}')
  printf "%-34s median %s ms/token  tokens ok %s  layers %s\n" "$label" "${med:-FAIL}" "$match" "${layers:-?}"
}
# baseline: the 4.09 ms build (git rev), same VM
if [ ! -x "$AB/base/run" ]; then
  rm -rf "$AB/base_src"; git -C "$ROOT" worktree add -f "$AB/base_src" "$BASE_REV" > /dev/null 2>&1 || cp -r "$ROOT" "$AB/base_src"
  ln -sfn "$ROOT/.venv" "$AB/base_src/.venv"
  (cd "$AB/base_src" && ETX_OUT="$AB/base" bash examples/dsv2lite/build.sh "$FLEET" -Rpass-analysis=kernel-resource-usage > "$AB/base.build.log" 2>&1) || echo "base: BUILD FAILED"
  [ -x "$AB/base/run" ] || ln -sfn "$AB/base_src/build/dsv2lite" "$AB/base"    # older build.sh ignores ETX_OUT
  grep -A12 "Function Name: etx_megakernel_d0" "$AB/base.build.log" | grep -E "VGPRs:|ScratchSize" | tr -s ' ' | sed "s/^/  base /" | head -3
fi
build new                                   # relay + local acquire + lean loop + const params + layer immediates
build nocp       -- -DFLEET_CONST_PARAMS=0  # params through the global pointer again
build nordc      ETX_RDC=0                  # no -fgpu-rdc (fleet's build)
build relayoff   ETX_RELAY_PLAN=off         # 38 workers/XCD, no relay
run base          "$AB/base"
run new           "$AB/new"
run new_relay0    "$AB/new" ETX_RELAY=0
run new_acqcons   "$AB/new" ETX_RELAY_ACQ=consumer
run nocp          "$AB/nocp"
run nordc         "$AB/nordc"
run relayoff      "$AB/relayoff"
run base_again    "$AB/base"
run new_again     "$AB/new"
