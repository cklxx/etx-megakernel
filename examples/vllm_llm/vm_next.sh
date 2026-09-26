#!/usr/bin/env bash
# One session, both models: per-kernel timing against vLLM's originals (--check), the synchronisation
# variants (chained launches, one block per workgroup, both), and vLLM's kernel trace. Full logs kept in
# ~/vlnext/ (nothing filtered away). Repo at ~/etx, vLLM sources at ~/vllm, torch headers at ~/rocm-headers.
#   nohup bash ~/etx/examples/vllm_llm/vm_next.sh > ~/next.log 2>&1 &
set -uo pipefail
cd ~/etx; mkdir -p ~/vlnext
log() { printf '\n=== %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
IMG=${VLLM_IMAGE:-rocm/vllm:latest}
DOCKER="sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 -v $HOME:$HOME -w $HOME/etx"
(sudo docker pull -q $IMG > /tmp/pull.log 2>&1; echo PULL-DONE >> /tmp/pull.log) &
log "envs and downloads"
sudo apt-get install -y -qq python3.12-venv > /tmp/apt.log 2>&1 || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv >> /tmp/apt.log 2>&1; }
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/python -c "import yaml" 2>/dev/null || .venv/bin/pip install -q pyyaml
[ -d ~/llmenv ] || python3 -m venv ~/llmenv
. ~/llmenv/bin/activate
pip install -q --upgrade pip; pip install -q numpy "huggingface_hub[hf_transfer]" hf_transfer safetensors
for m in Qwen/Qwen2.5-1.5B:qwen2.5-1.5b Qwen/Qwen3-8B:qwen3-8b; do hf=${m%%:*}; n=${m##*:}
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$hf', local_dir='$HOME/llm/$n/hf', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model'], max_workers=16)
print('DL-DONE $n')" > /tmp/dl_$n.log 2>&1 &
done
pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch 2>&1 | tail -1
wait
V="VLLM=$HOME/vllm ETX_TORCH_HEADERS=$HOME/rocm-headers/pytorch"
SLOTS1="attn_mfma4_g4=1,attn_mfma16_g6=1,attn_reduce=1,rms_norm_3d=1,reshape_and_cache=1"
for n in qwen2.5-1.5b qwen3-8b; do
  log "prep + builds $n (in parallel)"
  cp examples/llm/results/golden_c1024_$n.txt ~/llm/$n/golden_c1024.txt
  python3 examples/vllm_llm/prep.py --src ~/llm/$n/hf --out ~/llm/$n --maxlen 1072 2>&1 | tail -1
  env $V ETX_VL_CHECK=1 ETX_OUT=build/vl/$n bash examples/vllm_llm/build.sh ~/llm/$n > ~/vlnext/build_${n}.log 2>&1 &
  env $V ETX_VL_CHAIN=1 ETX_OUT=build/vl/${n}_chain bash examples/vllm_llm/build.sh ~/llm/$n > ~/vlnext/build_${n}_chain.log 2>&1 &
  env $V ETX_VL_SLOTS=$SLOTS1 ETX_OUT=build/vl/${n}_s1 bash examples/vllm_llm/build.sh ~/llm/$n > ~/vlnext/build_${n}_s1.log 2>&1 &
  env $V ETX_VL_CHAIN=1 ETX_VL_SLOTS=$SLOTS1 ETX_OUT=build/vl/${n}_chain_s1 bash examples/vllm_llm/build.sh ~/llm/$n > ~/vlnext/build_${n}_chain_s1.log 2>&1 &
  wait
  for v in "" _chain _s1 _chain_s1; do grep -E "Function Name|built|error|failed" ~/vlnext/build_${n}$v.log | head -3 | sed "s/^/[$n$v] /"; done
done
for n in qwen2.5-1.5b qwen3-8b; do
  log "per-kernel check and timing $n"
  B=build/vl/$n
  timeout 900 stdbuf -oL $B/run ~/llm/$n $B/vl_plan.txt --check 2 > ~/vlnext/check_$n.log 2>&1; echo "rc=$?"
  grep -E "HANG|DIFFERS|CHECK|original .* imported|ETX\)" ~/vlnext/check_$n.log | tail -16
  for v in "" _chain _s1 _chain_s1; do for mode in fused unfused; do
    [ "$mode" = unfused ] && [ "$v" != "" ] && [ "$v" != "_s1" ] && continue
    R=build/vl/$n$v; f=$([ $mode = unfused ] && echo --unfused)
    timeout 900 stdbuf -oL $R/run ~/llm/$n $R/vl_plan.txt --repeat 2 $f > ~/vlnext/run_${n}${v}_$mode.log 2>&1; rc=$?
    echo "[$n$v $mode] rc=$rc $(grep -E '^run 1|^RESULT|failed|rror' ~/vlnext/run_${n}${v}_$mode.log | tr '\n' ' ' | cut -c1-260)"
  done; done
done
log "vLLM kernel traces (custom_ops=all: the same kernels)"
until grep -q PULL-DONE /tmp/pull.log; do sleep 10; done
mkdir -p examples/vllm_llm/results; chmod 777 examples/vllm_llm/results ~/vlnext
for n in qwen2.5-1.5b qwen3-8b; do
  $DOCKER $IMG bash -c "rocprofv3 --kernel-trace --output-format csv -d $HOME/vlnext/prof_$n -- python3 examples/vllm_llm/vllm_profile.py run --model ~/llm/$n/hf --golden ~/llm/$n/golden_c1024.txt --custom-ops all > $HOME/vlnext/profrun_$n.log 2>&1; tail -3 $HOME/vlnext/profrun_$n.log; python3 examples/vllm_llm/vllm_profile.py analyze $HOME/vlnext/prof_$n --json examples/vllm_llm/results/prof_all_$n.json" 2>&1 | grep -vE "^\s*$|simple_timer" | head -34
done
log "ALL DONE"
