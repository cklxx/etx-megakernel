#!/bin/bash
set -euo pipefail
IMAGE='vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7'
docker run --rm --init --pull=never --network=none --ipc=host --shm-size=16g --name etx-c6-build-723 -v /root:/root -w /root/etx --entrypoint timeout "$IMAGE" -k 5 1800 bash -lc '
set -e
mkdir -p /root/c6-headers-723
ln -sfn /opt/rocm-7.2.3/include /root/c6-headers-723/inc
ln -sfn /opt/rocm-7.2.3/lib/llvm/lib/clang/22/include /root/c6-headers-723/clang
for d in hip hipcub rocprim; do ln -sfn inc/$d /root/c6-headers-723/$d; done
export PATH=/root/etx-python/bin:/opt/rocm/bin:$PATH PYTHONDONTWRITEBYTECODE=1 ETX_VLLM=/root/vllm-pinned VLLM=/root/vllm-pinned ETX_ROCM_HEADERS=/root/c6-headers-723 ETX_TORCH_HEADERS=/root/rocm-headers/pytorch ETX_LLVM=/opt/rocm-7.2.3/lib/llvm/bin
export ETX_DEVLIBS=$(find /opt/rocm-7.2.3 -name ocml.bc -print -quit | xargs dirname)
test -f "$ETX_DEVLIBS/ocml.bc"
python -u -m examples.vllm_llm.attention_early_merge.session --work /root/c6-live-20261015 --model /root/llm/qwen3-30b-a3b build
'
