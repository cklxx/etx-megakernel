#!/usr/bin/env bash
# Run with system Python: a missing/broken .venv must itself be reported.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -B -u -m examples.vllm_llm.vm_preflight "$@"
