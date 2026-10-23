#!/usr/bin/env bash
# Build the ETX port of fleet-mi300x. Usage: bash examples/dsv2lite/build.sh <path-to-fleet-mi300x> [extra hipcc flags]
# Recipe: examples/dsv2lite/build.json; variants ETX_FLEET_PATCH, ETX_INLINE_TILES, ETX_RELAY_PLAN, ETX_RDC,
# ETX_DSV2_D5, ETX_COHERENCE=l1 (adds l1_loads), -DETX_DSV2_STAMPS2 (refuses ETX_DSV2_ISOLATE), ETX_BUILD_VERBOSE.
# Implementation: etx.external.build_model (etx/models/dsv2lite.py). ETX_MODEL_COMPILER=docker:<image> compiles locally.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${ETX_PYTHON:-$ROOT/.venv/bin/python}" -m etx.tools.model_build "$ROOT/examples/dsv2lite/build.json" "$@"
