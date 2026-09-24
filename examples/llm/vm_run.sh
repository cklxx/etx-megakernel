#!/usr/bin/env bash
# Fresh Hot Aisle MI300X VM -> three models prepared, built and run. Usage (on the VM, repo at ~/etx):
#   nohup bash ~/etx/examples/llm/vm_run.sh > ~/llm.log 2>&1 &
set -uo pipefail
cd ~/etx
log() { printf '\n=== %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
MODELS=("Qwen/Qwen2.5-1.5B qwen2.5-1.5b" "Qwen/Qwen3-8B qwen3-8b" "Qwen/Qwen3-30B-A3B qwen3-30b-a3b")
GEN=${GEN:-32}

log "python envs"
sudo apt-get update -qq > /tmp/apt.log 2>&1; sudo apt-get install -y -qq python3.12-venv >> /tmp/apt.log 2>&1
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/python -c "import yaml" 2>/dev/null || .venv/bin/pip install -q pyyaml
[ -d ~/llmenv ] || python3 -m venv ~/llmenv
. ~/llmenv/bin/activate
pip install -q --upgrade pip
pip install -q numpy pyyaml "huggingface_hub[hf_transfer]" hf_transfer safetensors
log "downloads (background, parallel)"
for m in "${MODELS[@]}"; do set -- $m
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$1', local_dir='$HOME/llm/$2/hf', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model'], max_workers=16)
print('DL-DONE $2')" > /tmp/dl_$2.log 2>&1 &
done
pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch 2>&1 | tail -1
pip install -q "transformers>=4.51,<5" accelerate 2>&1 | tail -1
python3 -c "import torch, transformers; print('torch', torch.__version__, torch.cuda.is_available(), 'transformers', transformers.__version__)"
wait
for m in "${MODELS[@]}"; do set -- $m; tail -1 /tmp/dl_$2.log; done

for m in "${MODELS[@]}"; do set -- $m
  log "prep $2"
  python3 examples/llm/prep.py --model "$1" --out ~/llm/$2 --gen $GEN 2>&1 | grep -vE "Loading checkpoint|it/s\]|^\s*$" | tail -4
  log "build $2"
  bash examples/llm/build.sh ~/llm/$2 > /tmp/build_$2.log 2>&1 || { echo "BUILD FAILED"; grep -E "error" /tmp/build_$2.log | head -20; continue; }
  grep -E "tasks=|workers" /tmp/build_$2.log
  log "run $2 (free-running)"
  timeout 1200 build/llm/$2/run ~/llm/$2 --repeat 2 2>&1 | tail -6
  log "run $2 (teacher-forced)"
  timeout 1200 build/llm/$2/run ~/llm/$2 --repeat 1 --teacher 2>&1 | tail -3
done
log "ALL DONE"
