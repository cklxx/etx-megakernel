#!/bin/bash
set -euo pipefail
IMAGE='vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7'
CAP=$(/root/etx-python/bin/python -c 'import json,time; print(int(json.load(open("/root/c6-live-20261015/lease.json"))["deadline"]-time.time()-15))')
test "$CAP" -gt 0
printf 'C6 run cap %s; original lease preserved\n' "$CAP"
docker run --rm --init --pull=never --network=none --ipc=host --shm-size=16g --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add=video --name etx-c6-run-723 -v /root:/root -w /root/etx -e HIP_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 --entrypoint timeout "$IMAGE" -k 5 "$CAP" bash -lc '
export PATH=/root/etx-python/bin:/opt/rocm/bin:$PATH PYTHONDONTWRITEBYTECODE=1 ETX_VLLM=/root/vllm-pinned VLLM=/root/vllm-pinned ETX_ROCM_HEADERS=/root/c6-headers-723 ETX_TORCH_HEADERS=/root/rocm-headers/pytorch ETX_LLVM=/opt/rocm-7.2.3/lib/llvm/bin
export ETX_DEVLIBS=$(find /opt/rocm-7.2.3 -name ocml.bc -print -quit | xargs dirname)
python -u -m examples.vllm_llm.attention_early_merge.session --work /root/c6-live-20261015 --model /root/llm/qwen3-30b-a3b run
'
