#!/usr/bin/env bash
# Short diagnosis session: Qwen2.5-1.5B first (small download, fast build), every imported launch of layers 0-1
# checked against vLLM's original kernel, then the full runs if the check passes. Qwen3-8B downloads meanwhile.
#   nohup bash ~/etx/examples/vllm_llm/vm_check.sh > ~/chk.log 2>&1 &
set -uo pipefail
cd ~/etx
log() { printf '\n=== %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "envs and downloads"
(sudo docker pull -q ${VLLM_IMAGE:-rocm/vllm:latest} > /tmp/pull.log 2>&1; echo PULL-DONE >> /tmp/pull.log) &
sudo apt-get install -y -qq python3.12-venv > /tmp/apt.log 2>&1 || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv >> /tmp/apt.log 2>&1; }
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/python -c "import yaml" 2>/dev/null || .venv/bin/pip install -q pyyaml
[ -d ~/llmenv ] || python3 -m venv ~/llmenv
. ~/llmenv/bin/activate
pip install -q --upgrade pip; pip install -q numpy "huggingface_hub[hf_transfer]" hf_transfer safetensors
dl() { HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$1', local_dir='$HOME/llm/$2/hf', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model'], max_workers=16)
print('DL-DONE $2')" > /tmp/dl_$2.log 2>&1; }
dl Qwen/Qwen2.5-1.5B qwen2.5-1.5b &
dl Qwen/Qwen3-8B qwen3-8b &
pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch 2>&1 | tail -1
until grep -q DL-DONE /tmp/dl_qwen2.5-1.5b.log 2>/dev/null; do sleep 5; done
name=qwen2.5-1.5b
log "prep + check build $name"
cp examples/llm/results/golden_c1024_$name.txt ~/llm/$name/golden_c1024.txt
python3 examples/vllm_llm/prep.py --src ~/llm/$name/hf --out ~/llm/$name --maxlen 1072 2>&1 | tail -1
ETX_VL_CHECK=1 VLLM=~/vllm ETX_TORCH_HEADERS=~/rocm-headers/pytorch bash examples/vllm_llm/build.sh ~/llm/$name > /tmp/b_$name.log 2>&1 || { echo "BUILD FAILED"; grep -iE "error|failed" /tmp/b_$name.log | head -20; }
grep -E "Function Name|built" /tmp/b_$name.log
log "check $name (layers 0-1)"
B=build/vl/$name
timeout 600 stdbuf -oL $B/run ~/llm/$name $B/vl_plan.txt --check 2 2>&1 | tail -40
log "CHECK DONE"
