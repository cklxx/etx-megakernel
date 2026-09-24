#!/usr/bin/env bash
# Round-3 A/B: previous best (rev), share immediates, + relay spin, and fleet itself, alternating.
# Usage: bash examples/dsv2lite/ab2.sh <fleet-mi300x> <prev-rev> [rounds]
set -uo pipefail
FLEET="${1:?}"; PREV="${2:?}"; ROUNDS="${3:-3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; AB="$ROOT/build/ab2"; mkdir -p "$AB"
rm -rf "$AB/prev_src"; git -C "$ROOT" worktree add -f "$AB/prev_src" "$PREV" >/dev/null 2>&1
ln -sfn "$ROOT/.venv" "$AB/prev_src/.venv"
(cd "$AB/prev_src" && ETX_OUT="$AB/prev" bash examples/dsv2lite/build.sh "$FLEET" >"$AB/prev.build.log" 2>&1) || echo "prev build failed"
ETX_OUT="$AB/imm" bash "$ROOT/examples/dsv2lite/build.sh" "$FLEET" >"$AB/imm.build.log" 2>&1 || echo "imm build failed"
ETX_OUT="$AB/spin" bash "$ROOT/examples/dsv2lite/build.sh" "$FLEET" -DETX_RELAY_SPIN=1 >"$AB/spin.build.log" 2>&1 || echo "spin build failed"
cd "$FLEET"
for r in $(seq 1 "$ROUNDS"); do
  for v in prev imm spin fleet; do
    if [ "$v" = fleet ]; then
      x=$(./build/fleet_decode_nt --graph build/taskgraph_d16.bin --repeat 2 2>&1)
      m=$(echo "$x" | grep -oE "median [0-9.]+ ms" | tail -1); ok=$(echo "$x" | grep -oE "tokens: [0-9]+/[0-9]+" | tail -1)
    else
      x=$("$AB/$v/run" --tokens 32 --context 1024 --repeat 2 2>&1)
      m=$(echo "$x" | grep -oE "median [0-9.]+ ms" | tail -1); ok="ok=$(echo "$x" | grep -c " ok$") $(echo "$x" | grep -oE "gate: [0-9]+ of [0-9]+")"
    fi
    printf "round %d  %-6s %s  %s\n" "$r" "$v" "$m" "$ok"
  done
done
