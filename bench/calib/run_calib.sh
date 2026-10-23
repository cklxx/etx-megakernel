#!/usr/bin/env bash
# Build and run the phase-0 calibration on the current GPU; prints numbers to
# paste into etx/machine/arch/<arch>.yaml costs:. Requires hipcc (AMD) for now.
# Implementation: etx/experiments/calibration.py behind etx.external.run_native_probe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${ETX_PYTHON:-$ROOT/.venv/bin/python}" -m etx.tools.native_probe calibration "$@"
