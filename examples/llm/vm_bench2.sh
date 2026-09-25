#!/usr/bin/env bash
# Round 2: sliced graph (one slice per XCD) vs per-operator graph vs unfused, plus DeepSeek unfused.
#   nohup bash ~/etx/examples/llm/vm_bench2.sh > ~/bench2.log 2>&1 &
set -uo pipefail
cd ~/etx
log() { printf '\n=== %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
MODELS=("Qwen/Qwen2.5-1.5B qwen2.5-1.5b" "Qwen/Qwen3-8B qwen3-8b" "Qwen/Qwen3-30B-A3B qwen3-30b-a3b")
NGPU=$(rocm-smi --showproductname 2>/dev/null | grep -c "Card Series")
G=golden_c1024.txt

log "envs + downloads + fleet bootstrap"
[ -d ~/fleet-mi300x ] && (cd ~/fleet-mi300x && HIP_VISIBLE_DEVICES=$([ "$NGPU" -ge 2 ] && echo 1 || echo 0) nohup bash scripts/hotaisle_bootstrap.sh > ~/bootstrap.log 2>&1 &)
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
for m in "${MODELS[@]}"; do set -- $m; tail -1 /tmp/dl_$2.log; done
[ "$NGPU" -ge 2 ] || { log "waiting for fleet's bootstrap (one GPU)"; until grep -qE "=== .* variants|BOOTSTRAP-DONE" ~/bootstrap.log 2>/dev/null; do sleep 20; done; pkill -f hotaisle_bootstrap.sh; sleep 1; pkill -f fleet_decode; }

for m in "${MODELS[@]}"; do set -- $m; name=$2
  log "prep + build $name"
  HIP_VISIBLE_DEVICES=0 ETX_LLM_CTX=2048 ETX_LLM_CHUNK=64 python3 examples/llm/prep.py --model "$1" --out ~/llm/$name --gen 32 --context 1024 --sliced > /tmp/prep_$name.log 2>&1 || { echo "PREP FAILED $name"; tail -5 /tmp/prep_$name.log; continue; }
  sed -i "s/^cfg CTX .*/cfg CTX 2048/; s/^cfg CHUNK .*/cfg CHUNK 64/" ~/llm/$name/llm.manifest
  U=32; [ $name = qwen3-30b-a3b ] && U=16
  ETX_OUT=$HOME/etx/build/r2/$name-op bash examples/llm/build.sh ~/llm/$name -DLLM_NT_WEIGHTS=1 -DLLM_GEMV_U=$U > /tmp/b_op_$name.log 2>&1 || { echo "BUILD FAILED $name op"; grep " error" /tmp/b_op_$name.log | head; }
  ETX_LLM_GRAPH=examples/llm/model_sliced.py ETX_OUT=$HOME/etx/build/r2/$name-sl bash examples/llm/build.sh ~/llm/$name -DLLM_NT_WEIGHTS=1 -DLLM_GEMV_U=$U -Rpass-analysis=kernel-resource-usage > /tmp/b_sl_$name.log 2>&1 || { echo "BUILD FAILED $name sliced"; grep " error" /tmp/b_sl_$name.log | head; }
  grep -E "tasks=|workers" /tmp/b_sl_$name.log | tr "\n" " "; grep -A14 "Function Name: etx_megakernel_d0" /tmp/b_sl_$name.log | grep -E "VGPRs:|ScratchSize" | sed "s/.*remark: *//; s/ \[-R.*//" | tr "\n" " "; echo
  for bin in "$name-sl" "$name-op"; do for mode in fused unfused; do
    flag=$([ "$mode" = unfused ] && echo --unfused || echo)
    HIP_VISIBLE_DEVICES=0 timeout 1800 build/r2/$bin/run ~/llm/$name --golden $G --repeat 2 $flag 2>&1 | grep -E "^run 1|^RESULT|^graph|rror" | tr "\n" " "; echo " [$bin $mode]"
  done; done
  HIP_VISIBLE_DEVICES=0 timeout 1800 build/r2/$name-sl/run ~/llm/$name --golden $G --repeat 3 --teacher 2>&1 | grep -E "^run" | sed "s/^/sliced teacher: /"
done

log "DeepSeek: fleet, ETX fused, ETX unfused (fleet's tiles)"
until grep -qE "=== .* variants|BOOTSTRAP-DONE" ~/bootstrap.log 2>/dev/null; do sleep 20; done
pkill -f hotaisle_bootstrap.sh; sleep 1; pkill -f fleet_decode; sleep 2
cd ~/fleet-mi300x
HIP_VISIBLE_DEVICES=0 ./build/fleet_decode_nt --graph build/taskgraph_d16.bin --repeat 2 2>&1 | grep -E "per-token|tokens:" | tail -2 | sed "s/^/fleet: /"
cd ~/etx && ETX_OUT=$HOME/etx/build/ds bash examples/dsv2lite/build.sh ~/fleet-mi300x > /tmp/ds.log 2>&1 || echo "DS BUILD FAILED"
cd ~/fleet-mi300x
for mode in "" --unfused; do HIP_VISIBLE_DEVICES=0 ~/etx/build/ds/run --tokens 32 --context 1024 --repeat 2 $mode 2>&1 | grep -E "median|consecutive" | tr "\n" " "; echo " [ETX DeepSeek ${mode:-fused}]"; done
log "BENCH2 DONE"
