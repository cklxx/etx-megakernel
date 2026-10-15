#!/bin/bash
set -euo pipefail
cd /root/etx
mkdir -p /root/setup-live
printf 'START %s\n' "$(date -u +%FT%TZ)"
ln -sfn /opt/rocm/core-7.14/lib/llvm/lib/clang/23/include /root/rocm-headers/clang
IMAGE='vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7'
test -s /root/llm/qwen3-30b-a3b/hf/REVISION
docker run --rm --init --ipc=host --shm-size=16g --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add=video -v /root:/root -w /root/etx -e PYTHONUNBUFFERED=1 -e HOME=/root -e HIP_VISIBLE_DEVICES=0 --entrypoint timeout "$IMAGE" -k 5 240 stdbuf -oL -eL python3 examples/vllm_llm/prep.py --src /root/llm/qwen3-30b-a3b/hf --out /root/llm/qwen3-30b-a3b --maxlen 1072
cp examples/llm/results/golden_c1024_qwen3-30b-a3b.txt /root/llm/qwen3-30b-a3b/golden_c1024.txt
/root/etx-python/bin/python - <<'PY'
import json,hashlib
from pathlib import Path
model=Path('/root/llm/qwen3-30b-a3b')
Path('/root/qwen-g6.json').write_text(json.dumps(dict(model=str(model),hf=str(model/'hf'),golden_sha256=hashlib.sha256((model/'golden_c1024.txt').read_bytes()).hexdigest()),indent=2)+'\n')
PY
printf 'COMPLETE %s\n' "$(date -u +%FT%TZ)"
