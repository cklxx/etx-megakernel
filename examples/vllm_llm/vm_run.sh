#!/usr/bin/env bash
# One MI300X VM: vLLM's kernels inside the ETX megakernel vs the same kernels launched one by one (ETX
# unfused, HIP graph) vs vLLM itself (default and custom_ops=all, i.e. with exactly these kernels), plus
# vLLM kernel traces (GPU span vs kernel busy time per token). Repo at ~/etx, vLLM sources at ~/vllm
# (csrc is enough), torch header-only headers at ~/rocm-headers/pytorch, golden files in the repo.
#   nohup bash ~/etx/examples/vllm_llm/vm_run.sh > ~/vl.log 2>&1 &
set -uo pipefail
cd ~/etx
log() { printf '\n=== %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
MODELS=(${VL_MODELS:-"Qwen/Qwen3-8B:qwen3-8b" "Qwen/Qwen2.5-1.5B:qwen2.5-1.5b"})
IMG=${VLLM_IMAGE:-rocm/vllm:latest}
DOCKER="sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 -v $HOME:$HOME -w $HOME/etx"

log "background: vLLM image"
command -v docker >/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq docker.io > /tmp/docker_apt.log 2>&1; }
(sudo docker pull -q $IMG > /tmp/pull.log 2>&1; echo PULL-DONE >> /tmp/pull.log) &

log "python envs and downloads"
sudo apt-get install -y -qq python3.12-venv > /tmp/apt.log 2>&1 || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv >> /tmp/apt.log 2>&1; }
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/python -c "import yaml" 2>/dev/null || .venv/bin/pip install -q pyyaml
[ -d ~/llmenv ] || python3 -m venv ~/llmenv
. ~/llmenv/bin/activate
pip install -q --upgrade pip; pip install -q numpy pyyaml "huggingface_hub[hf_transfer]" hf_transfer safetensors
for m in "${MODELS[@]}"; do hf=${m%%:*}; name=${m##*:}
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$hf', local_dir='$HOME/llm/$name/hf', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model'], max_workers=16)
print('DL-DONE $name')" > /tmp/dl_$name.log 2>&1 &
done
pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch 2>&1 | tail -1
wait
for m in "${MODELS[@]}"; do tail -1 /tmp/dl_${m##*:}.log; done; tail -1 /tmp/pull.log

for m in "${MODELS[@]}"; do name=${m##*:}
  log "prep + build $name"
  cp examples/llm/results/golden_c1024_$name.txt ~/llm/$name/golden_c1024.txt
  python3 examples/vllm_llm/prep.py --src ~/llm/$name/hf --out ~/llm/$name --maxlen 1072 > /tmp/vlprep_$name.log 2>&1 || { echo "PREP FAILED $name"; tail -5 /tmp/vlprep_$name.log; continue; }
  tail -1 /tmp/vlprep_$name.log
  VLLM=~/vllm ETX_TORCH_HEADERS=~/rocm-headers/pytorch bash examples/vllm_llm/build.sh ~/llm/$name > /tmp/vlbuild_$name.log 2>&1 || { echo "BUILD FAILED $name"; grep -iE "error|failed" /tmp/vlbuild_$name.log | head -20; continue; }
  grep -E "tasks=|Function Name|built" /tmp/vlbuild_$name.log
  B=build/vl/$name
  for mode in fused unfused fused unfused; do flag=$([ $mode = unfused ] && echo --unfused || echo)
    timeout 1800 $B/run ~/llm/$name $B/vl_plan.txt --repeat 2 $flag 2>&1 | grep -E "^run 1|^RESULT|rror|failed" | tr "\n" " "; echo " [$name vl $mode]"
  done
  timeout 1800 $B/run ~/llm/$name $B/vl_plan.txt --repeat 1 --teacher 2>&1 | grep -E "^run" | tr "\n" " "; echo " [$name vl teacher]"
  log "variant: small launches between qkv and o_proj pinned to one XCD (XCD-local events) $name"
  ETX_VL_PIN=1 ETX_OUT=build/vl/${name}_pin VLLM=~/vllm ETX_TORCH_HEADERS=~/rocm-headers/pytorch bash examples/vllm_llm/build.sh ~/llm/$name > /tmp/vlbuild_${name}_pin.log 2>&1 || { echo "BUILD FAILED ${name}_pin"; grep -iE "error|failed" /tmp/vlbuild_${name}_pin.log | head; continue; }
  P=build/vl/${name}_pin
  for r in 1 2; do timeout 1800 $P/run ~/llm/$name $P/vl_plan.txt --repeat 2 2>&1 | grep -E "^run 1|^RESULT|rror|failed" | tr "\n" " "; echo " [$name vl-pin fused]"; done
done

log "vLLM (same prompt ids): default and custom_ops=all; kernel traces"
mkdir -p examples/vllm_llm/results; chmod 777 examples/vllm_llm/results
until grep -q PULL-DONE /tmp/pull.log; do sleep 10; done
for m in "${MODELS[@]}"; do name=${m##*:}
  for ops in none all; do co=$([ $ops = all ] && echo "--custom-ops all" || echo)
    $DOCKER $IMG python3 examples/llm/vllm_bench.py --model ~/llm/$name/hf --golden ~/llm/$name/golden_c1024.txt --label $name $co \
      --json examples/vllm_llm/results/vllm_${ops}_$name.json 2>&1 | grep -E "^VLLM|Error|error" | tail -3
    $DOCKER $IMG bash -c "rm -rf /tmp/prof_${name}_$ops; rocprofv3 --kernel-trace --output-format csv -d /tmp/prof_${name}_$ops -- python3 examples/vllm_llm/vllm_profile.py run --model ~/llm/$name/hf --golden ~/llm/$name/golden_c1024.txt $co > /tmp/profrun.log 2>&1; tail -2 /tmp/profrun.log; python3 examples/vllm_llm/vllm_profile.py analyze /tmp/prof_${name}_$ops --json examples/vllm_llm/results/prof_${ops}_$name.json" 2>&1 | grep -vE "^\s*$" | head -24
  done
done
if [ "${VL_MOE:-1}" = 1 ]; then
  log "MoE capture for offline work: vLLM Qwen3-30B-A3B kernel trace + Triton cache (fused_moe IR and launch metadata)"
  . ~/llmenv/bin/activate
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-30B-A3B', local_dir='$HOME/llm/qwen3-30b-a3b/hf', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model'], max_workers=16)
print('DL-DONE')" > /tmp/dl_moe.log 2>&1; tail -1 /tmp/dl_moe.log
  cp examples/llm/results/golden_c1024_qwen3-30b-a3b.txt ~/llm/qwen3-30b-a3b/golden_c1024.txt
  mkdir -p ~/moe_capture && chmod 777 ~/moe_capture
  for ops in none all; do co=$([ $ops = all ] && echo "--custom-ops all" || echo)
    $DOCKER -e TRITON_CACHE_DIR=$HOME/moe_capture/triton_$ops $IMG bash -c "rocprofv3 --kernel-trace --output-format csv -d $HOME/moe_capture/prof_$ops -- python3 examples/vllm_llm/vllm_profile.py run --model ~/llm/qwen3-30b-a3b/hf --golden ~/llm/qwen3-30b-a3b/golden_c1024.txt $co > /tmp/profrun_moe.log 2>&1; tail -1 /tmp/profrun_moe.log; python3 examples/vllm_llm/vllm_profile.py analyze $HOME/moe_capture/prof_$ops --json examples/vllm_llm/results/prof_${ops}_qwen3-30b-a3b.json" 2>&1 | grep -vE "^\s*$" | head -30
  done
  sudo chown -R $USER ~/moe_capture; (cd ~ && tar czf moe_capture.tgz moe_capture/triton_* moe_capture/prof_*/*/*kernel_trace.csv 2>/dev/null); ls -la ~/moe_capture.tgz
fi
log "ALL DONE"
