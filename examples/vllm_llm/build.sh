#!/usr/bin/env bash
# Build the ETX megakernel of vLLM's kernels for one model packed by examples/vllm_llm/prep.py.
#   VLLM=~/vllm [ETX_HIPCC_FLAGS="-DETX_EV_STRIDE=1 ..."] bash examples/vllm_llm/build.sh <packed-dir>
# Steps: import the kernels this model needs (device IR from vLLM's sources with ROCm's hipcc), adapters,
# one bitcode bundle with the device-library functions they call, the ETX plan, the final hipcc link.
# Implementation: etx/models/vllm_llm/build.sh behind etx.external.run_harness_command.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${ETX_PYTHON:-$ROOT/.venv/bin/python}" -m etx.tools.harness qwen-vllm vllm_build --root "$ROOT" "$@"
