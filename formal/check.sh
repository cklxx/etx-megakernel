#!/usr/bin/env bash
# Formal checks of the ETX runtime protocols and the DSL GEMV (docs/MODEL-vllm-kernels.md section 14).
#   bash formal/check.sh            Lean: formal/Gemv.lean (the stream loop's row sums); TLC: ImportRounds, EventFlush
#   bash formal/check.sh old        also the negative tests: the first rounds protocol must FAIL (ImportRoundsOld),
#                                   and the coherence policy without the imported->coherent write-back (DataVisBad)
# Lean from ~/.elan (any 4.x); TLC from ~/tools/tla/tla2tools.jar in the etx-rocm22 container (a JRE is installed there).
set -uo pipefail
cd "$(dirname "$0")/.."
LEAN=$(ls -d ~/.elan/toolchains/*/bin/lean 2>/dev/null | tail -1)
echo "== Lean: formal/Gemv.lean"; "$LEAN" formal/Gemv.lean 2>&1 | grep -E "error|axioms" ; echo "lean exit ${PIPESTATUS[0]}"
mods="ImportRounds EventFlush DataVis"; [ "${1:-}" = old ] && mods="$mods ImportRoundsOld DataVisBad"
docker run --rm --platform linux/amd64 -v "$HOME/code/etx/formal:/f" -v "$HOME/tools/tla:/t" etx-rocm22 bash -c "
  apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq default-jre-headless >/dev/null 2>&1; cd /f
  for m in $mods; do echo \"== TLC: \$m\"; java -XX:+UseParallelGC -cp /t/tla2tools.jar tlc2.TLC -workers 4 -config \$m.cfg \$m.tla 2>&1 \
    | grep -E 'Error|violated|Deadlock|Temporal|Invariant|completed|No error|distinct states found, 0 states left' | head -6; done"
