#!/usr/bin/env bash
# Build and run the phase-0 calibration on the current GPU; prints numbers to
# paste into etx/machine/arch/<arch>.yaml costs:. Requires hipcc (AMD) for now.
set -euo pipefail
ARCH="${1:-gfx942}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/build/calib_$ARCH"
mkdir -p "$OUT"
"$ROOT/.venv/bin/python" - "$ARCH" "$OUT" <<'EOF'
import sys
from etx.codegen import emit_lowering_header
from etx.machine import load_machine
arch, out = sys.argv[1], sys.argv[2]
open(f"{out}/etx_lowering.h", "w").write(emit_lowering_header(load_machine(arch)))
EOF
for b in atomic_pingpong flag_latency queue_contention; do
  hipcc -O2 --offload-arch="$ARCH" -I "$ROOT/etx/runtime/include" -I "$OUT" "$ROOT/bench/calib/$b.hip" -o "$OUT/$b"
  echo "== $b"; "$OUT/$b" | tee "$OUT/$b.txt"
done
