#!/usr/bin/env bash
# Session script for plan v2 phase 1: GEMV microbenchmark sweep, then the sliced graph (post/merge folded,
# K/V staged) fused vs unfused on the three models with per-layer dump checks, then vLLM on the MoE model.
#   nohup bash ~/etx/examples/llm/vm_bench3.sh > ~/bench3.log 2>&1 &
set -uo pipefail
cd ~/etx; log() { printf '\n=== %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
MODELS=("Qwen/Qwen3-30B-A3B qwen3-30b-a3b" "Qwen/Qwen3-8B qwen3-8b" "Qwen/Qwen2.5-1.5B qwen2.5-1.5b")
G=golden_c1024.txt
IMG=${VLLM_IMAGE:-rocm/vllm:latest}
DOCKER="sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 -v $HOME:$HOME -w $HOME/etx"

log "envs, downloads, vLLM image (background)"
command -v docker >/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq docker.io > /tmp/docker_apt.log 2>&1; }
(sudo docker pull -q $IMG > /tmp/pull.log 2>&1; echo PULL-DONE >> /tmp/pull.log) &
sudo apt-get update -qq > /tmp/apt.log 2>&1; sudo apt-get install -y -qq python3.12-venv >> /tmp/apt.log 2>&1
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/python -c "import yaml" 2>/dev/null || .venv/bin/pip install -q pyyaml
[ -d ~/llmenv ] || python3 -m venv ~/llmenv
. ~/llmenv/bin/activate
pip install -q --upgrade pip; pip install -q numpy pyyaml "huggingface_hub[hf_transfer]" hf_transfer safetensors
for m in "${MODELS[@]}"; do set -- $m
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$1', local_dir='$HOME/llm/$2/hf', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model'], max_workers=16)
print('DL-DONE $2')" > /tmp/dl_$2.log 2>&1 &
done
pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch 2>&1 | tail -1
pip install -q "transformers>=4.51,<5" accelerate 2>&1 | tail -1

log "GEMV microbenchmark (U x NT x wg/CU)"
mkdir -p build
for U in 16 32 48; do for NT in 0 1; do
  hipcc -O3 -std=c++17 --offload-arch=gfx942 -I etx/runtime/include -I . -DLLM_NT_WEIGHTS=$NT -DLLM_GEMV_U=$U bench/gemv_bench.hip -o build/gemv_bench_${U}_$NT 2> /tmp/gb.log || { echo "gemv bench build failed U=$U NT=$NT"; grep error /tmp/gb.log | head -3; continue; }
  build/gemv_bench_${U}_$NT 8,16,32,64,128 512,768,1536,2048,4096,12288
done; done
wait
for m in "${MODELS[@]}"; do set -- $m; tail -1 /tmp/dl_$2.log; done

for m in "${MODELS[@]}"; do set -- $m; name=$2
  log "prep + build $name (sliced graph, phase-1 tiles)"
  HIP_VISIBLE_DEVICES=0 ETX_LLM_CTX=2048 ETX_LLM_CHUNK=64 python3 examples/llm/prep.py --model "$1" --out ~/llm/$name --gen 32 --context 1024 --sliced > /tmp/prep_$name.log 2>&1 || { echo "PREP FAILED $name"; tail -5 /tmp/prep_$name.log; continue; }
  sed -i "s/^cfg CTX .*/cfg CTX 2048/; s/^cfg CHUNK .*/cfg CHUNK 64/" ~/llm/$name/llm.manifest
  ETX_LLM_GRAPH=examples/llm/model_sliced.py ETX_OUT=$HOME/etx/build/p1/$name bash examples/llm/build.sh ~/llm/$name -DLLM_NT_WEIGHTS=1 -Rpass-analysis=kernel-resource-usage > /tmp/b_$name.log 2>&1 || { echo "BUILD FAILED $name"; grep " error" /tmp/b_$name.log | head; continue; }
  grep -E "tasks=|workers" /tmp/b_$name.log | tr "\n" " "; grep -A14 "Function Name: etx_megakernel_d0" /tmp/b_$name.log | grep -E "VGPRs:|ScratchSize" | sed "s/.*remark: *//; s/ \[-R.*//" | tr "\n" " "; echo
  for mode in fused unfused fused; do flag=$([ "$mode" = unfused ] && echo --unfused || echo)
    timeout 1800 build/p1/$name/run ~/llm/$name --golden $G --repeat 2 $flag 2>&1 | grep -E "^run 1|^RESULT|rror|failed" | tr "\n" " "; echo " [$name $mode]"
  done
  timeout 1800 build/p1/$name/run ~/llm/$name --golden $G --repeat 3 --teacher 2>&1 | grep -E "^run" | tr "\n" ";"; echo
  log "trace $name"; ETX_TRACE=1 build/p1/$name/run ~/llm/$name --golden $G --repeat 1 --teacher --gen 2 2>&1 | grep -E "^\s+[a-z_]+_10 |lmfold|lmhead"
done

until grep -q PULL-DONE /tmp/pull.log; do sleep 10; done
for A in 0 1; do
  log "vLLM qwen3-30b-a3b AITER=$A"
  timeout 1800 $DOCKER -e VLLM_ROCM_USE_AITER=$A $IMG python3 examples/llm/vllm_bench.py --model ~/llm/qwen3-30b-a3b/hf --golden ~/llm/qwen3-30b-a3b/$G --label qwen3-30b-a3b 2>&1 | grep -E "^VLLM|rror" | tail -2
done
log "BENCH3 DONE"
