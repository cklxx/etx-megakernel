#!/usr/bin/env bash
# Full local gate, no GPU, network, model download or provisioning.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?new preflight directory under ~/code}
mkdir -p "$OUT"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
from examples.vllm_llm.batch_correctness import execution_package
Path(sys.argv[1], 'execution-package.json').write_text(json.dumps(execution_package(), indent=2)+'\n')
PY
# Serial matrix builds bound the local compiler's peak memory.
for model in qwen3-8b qwen2.5-1.5b; do
  for variant in base chain xcd xcd_s1 uc uc_s1; do
    LOCAL_MATRIX="$model:$variant" bash examples/vllm_llm/local_matrix.sh "$OUT/matrix"
  done
done
for model in qwen3-30b-a3b mixtral-8x7b; do
  for variant in base chain; do
    LOCAL_MATRIX="$model:$variant" bash examples/vllm_llm/local_matrix.sh "$OUT/matrix"
  done
done
bash examples/vllm_llm/batch_local.sh "$OUT/rocm"
.venv/bin/python -m etx.tools.gpu_jobs examples/sessions/qwen_batch_correctness.py \
  --gpus 0 --logdir "$OUT/dryrun" --dry-run
.venv/bin/python -m pytest -q
