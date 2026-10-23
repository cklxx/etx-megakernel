#!/usr/bin/env bash
# CPU-only compile/link. Use an existing ROCm directory or local Docker image.
# bash tools/cpu_device_check.sh <codegen-dir> <rocm-directory-or-image>
# Implementation: etx.external.check_codegen (etx/build/codegen.py).
set -euo pipefail
exec env PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}" "${ETX_PYTHON:-python3}" -m etx.tools.device_check "$@"
