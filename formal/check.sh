#!/usr/bin/env bash
# Portable CPU formal checks. No Docker, GPU or package installation.
# LEAN=/path/to/lean TLC_JAR=/path/to/tla2tools.jar bash formal/check.sh [old] [outdir]
# "old" also requires the two known-bad models to produce actual counterexamples.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${2:-$ROOT/build/formal}
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
LEAN=${LEAN:-lean}
TLC_JAR=${TLC_JAR:?set TLC_JAR to the local tla2tools.jar}
command -v "$LEAN" >/dev/null || { echo "Lean unavailable: $LEAN" >&2; exit 1; }
command -v java >/dev/null || { echo 'java unavailable' >&2; exit 1; }
[[ -f "$TLC_JAR" ]] || { echo "Missing TLC_JAR: $TLC_JAR" >&2; exit 1; }
TLC_JAR="$(cd "$(dirname "$TLC_JAR")" && pwd)/$(basename "$TLC_JAR")"
"$LEAN" --version > "$OUT/versions.log"
java -version >> "$OUT/versions.log" 2>&1
printf 'Lean: formal/Gemv.lean\n'
"$LEAN" "$ROOT/formal/Gemv.lean" > "$OUT/lean.log" 2>&1 || { cat "$OUT/lean.log"; exit 1; }
# A successful Lean exit with sorryAx would still not prove the theorem.
if grep -qE 'sorryAx|declaration uses .sorry.' "$OUT/lean.log"; then
  cat "$OUT/lean.log"; exit 1
fi
cd "$ROOT/formal"
check_tlc() {
  local module=$1 cfg=$2 expected=$3 rc=0
  printf 'TLC: %s (%s, expected %s)\n' "$module" "$cfg" "$expected"
  java -XX:+UseParallelGC -cp "$TLC_JAR" tlc2.TLC -workers 2 \
    -metadir "$OUT/states-$cfg" -config "$cfg.cfg" "$module.tla" > "$OUT/$cfg.log" 2>&1 || rc=$?
  if [[ "$expected" == pass ]]; then
    [[ "$rc" == 0 ]] && grep -q 'No error has been found' "$OUT/$cfg.log" || { cat "$OUT/$cfg.log"; return 1; }
  else
    # A parse error or missing tool is not an expected invariant violation.
    [[ "$rc" != 0 ]] && grep -qE 'Invariant .* is violated|Temporal properties were violated|Deadlock reached' "$OUT/$cfg.log" \
      || { cat "$OUT/$cfg.log"; return 1; }
  fi
}
check_tlc ImportRounds ImportRounds pass
check_tlc EventFlush EventFlush pass
check_tlc DataVis DataVis pass
if [[ "${1:-}" == old ]]; then
  check_tlc ImportRoundsOld ImportRoundsOld fail
  # DataVisBad is a configuration of DataVis, not a separate .tla module.
  check_tlc DataVis DataVisBad fail
fi
printf 'Formal checks passed; full logs: %s\n' "$OUT"
