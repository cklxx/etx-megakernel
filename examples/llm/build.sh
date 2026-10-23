#!/usr/bin/env bash
# Build the ETX megakernel for one prepared model. Usage: bash examples/llm/build.sh <packed-dir> [hipcc flags]
# Recipe: examples/llm/build.json (ETX_LLM_GRAPH=examples/llm/model_sliced.py: one slice per XCD; ETX_OUT).
# Implementation: etx.external.build_model. ETX_MODEL_COMPILER=docker:<image> compiles in a local image.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${ETX_PYTHON:-$ROOT/.venv/bin/python}" -m etx.tools.model_build "$ROOT/examples/llm/build.json" "$@"
