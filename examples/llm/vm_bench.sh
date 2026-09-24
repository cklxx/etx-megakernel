#!/usr/bin/env bash
# Head-to-head on one MI300X VM: ETX megakernel vs the same tiles unfused (HIP graph) vs vLLM (default and AITER),
# all at a 1024-token context with the same prompt ids, batch 1, greedy; plus DeepSeek-V2-Lite with fleet.
#   nohup bash ~/etx/examples/llm/vm_bench.sh > ~/bench.log 2>&1 &
# ETX and vLLM run on GPU 0; fleet's bootstrap (DeepSeek weights, golden, fleet binary) runs on GPU 1 meanwhile.
set -uo pipefail
cd ~/etx
log() { printf '\n=== %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
MODELS=("Qwen/Qwen2.5-1.5B qwen2.5-1.5b" "Qwen/Qwen3-8B qwen3-8b" "Qwen/Qwen3-30B-A3B qwen3-30b-a3b")
IMG=${VLLM_IMAGE:-rocm/vllm:latest}
DOCKER="sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 -v $HOME:$HOME -w $HOME/etx"

log "background: fleet bootstrap on GPU 1, vLLM image pull"
[ -d ~/fleet-mi300x ] && (cd ~/fleet-mi300x && HIP_VISIBLE_DEVICES=1 nohup bash scripts/hotaisle_bootstrap.sh > ~/bootstrap.log 2>&1 &)
command -v docker >/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq docker.io > /tmp/docker_apt.log 2>&1; }
(sudo docker pull -q $IMG > /tmp/pull.log 2>&1; echo PULL-DONE >> /tmp/pull.log) &

log "python envs and downloads"
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
wait
for m in "${MODELS[@]}"; do set -- $m; tail -1 /tmp/dl_$2.log; done; tail -1 /tmp/pull.log

for m in "${MODELS[@]}"; do set -- $m
  log "prep + build $2"
  HIP_VISIBLE_DEVICES=0 ETX_LLM_CTX=2048 ETX_LLM_CHUNK=32 python3 examples/llm/prep.py --model "$1" --out ~/llm/$2 --gen 32 --context 1024 > /tmp/prep_$2.log 2>&1 || { echo "PREP FAILED $2"; tail -5 /tmp/prep_$2.log; continue; }
  sed -i "s/^cfg CTX .*/cfg CTX 2048/; s/^cfg CHUNK .*/cfg CHUNK 32/" ~/llm/$2/llm.manifest
  ETX_OUT=$HOME/etx/build/b/$2 bash examples/llm/build.sh ~/llm/$2 > /tmp/b_$2.log 2>&1 || { echo "BUILD FAILED $2"; grep " error" /tmp/b_$2.log | head; }
  ETX_OUT=$HOME/etx/build/b/$2-nt bash examples/llm/build.sh ~/llm/$2 -DLLM_NT_WEIGHTS=1 > /tmp/bnt_$2.log 2>&1 || echo "NT BUILD FAILED $2"
done

G=golden_c1024.txt
for m in "${MODELS[@]}"; do set -- $m
  name=$2
  log "ETX $name (context 1024)"
  for bin in "$name" "$name-nt"; do
    for mode in fused unfused; do
      flag=$([ "$mode" = unfused ] && echo --unfused || echo)
      HIP_VISIBLE_DEVICES=0 timeout 1800 build/b/$bin/run ~/llm/$name --golden $G --repeat 2 $flag 2>&1 | grep -E "^run 1|^RESULT|^mode" | tr "\n" " "; echo " [$bin $mode]"
    done
  done
  HIP_VISIBLE_DEVICES=0 timeout 1800 build/b/$name/run ~/llm/$name --golden $G --repeat 1 --teacher 2>&1 | grep -E "^run 0" | sed "s/^/teacher: /"
done

until grep -q PULL-DONE /tmp/pull.log; do sleep 10; done
for m in "${MODELS[@]}"; do set -- $m
  for A in 1 0; do
    log "vLLM $2 AITER=$A"
    timeout 1800 $DOCKER -e VLLM_ROCM_USE_AITER=$A $IMG python3 examples/llm/vllm_bench.py --model ~/llm/$2/hf --golden ~/llm/$2/$G --label $2 \
      --json ~/llm/$2/vllm_aiter$A.json 2>&1 | grep -E "^VLLM|Error|error" | tail -3
  done
done

log "DeepSeek-V2-Lite: wait for fleet's bootstrap"
until grep -q "=== .* variants" ~/bootstrap.log 2>/dev/null || grep -q "BOOTSTRAP-DONE" ~/bootstrap.log 2>/dev/null; do sleep 20; done
pkill -f hotaisle_bootstrap.sh; sleep 1; pkill -f fleet_decode; sleep 2
grep -E "per-token latency" ~/bootstrap.log | tail -2 | sed "s/^/fleet (GPU 1, bootstrap): /"
cd ~/fleet-mi300x
HIP_VISIBLE_DEVICES=0 ./build/fleet_decode_nt --graph build/taskgraph_d16.bin --repeat 2 2>&1 | grep -E "per-token|tokens:" | tail -2 | sed "s/^/fleet GPU0: /"
cd ~/etx && ETX_OUT=$HOME/etx/build/ds bash examples/dsv2lite/build.sh ~/fleet-mi300x > /tmp/ds.log 2>&1 || echo "DS BUILD FAILED"
cd ~/fleet-mi300x && HIP_VISIBLE_DEVICES=0 ~/etx/build/ds/run --tokens 32 --context 1024 --repeat 2 2>&1 | grep -E "median|consecutive" | tr "\n" " "; echo " [ETX DeepSeek]"
for A in 1 0; do
  timeout 1800 $DOCKER -e VLLM_ROCM_USE_AITER=$A -v $HOME/models:/models -w $HOME/fleet-mi300x $IMG python3 bench/vllm_decode_timing.py --model /models/dsv2-lite-base \
    --golden build/golden_tokens.txt --json /tmp/vllm_ds_aiter$A.json 2>&1 | grep -E "decode:|tokens vs golden" | tr "\n" " "; echo " [vLLM DeepSeek AITER=$A]"
done
log "BENCH DONE"
