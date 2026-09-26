#!/usr/bin/env bash
# The next GPU session, in the order CLAUDE.md prescribes: the deciding microbenchmarks first, then the ceiling
# (vLLM's kernel trace), then correctness, then the variants, the MoE capture last. Every step logs in full to
# ~/vlnext/ and prints the elapsed time; nothing is filtered away.
#   nohup bash ~/etx/examples/vllm_llm/vm_next.sh > ~/next.log 2>&1 &
#
# Gate (decided before the session): the uncached variants are built and run only if the microbenchmark shows
#   (1) uncached hand-off with DOMAIN fences: 0 stale words across XCDs, and
#   (2) its whole-GPU barrier is faster than today's cached + device-fence barrier.
# Budget: about 2 h of a 1x MI300X ($6); VL_MOE=0 skips the MoE capture (about 20 min).
set -uo pipefail
cd ~/etx; mkdir -p ~/vlnext; T0=$(date +%s)
log() { printf '\n=== %s (+%d min) %s\n' "$(date -u +%H:%M:%S)" $(( ($(date +%s) - T0) / 60 )) "$*"; }
IMG=${VLLM_IMAGE:-rocm/vllm:latest}
DOCKER="sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 -v $HOME:$HOME -w $HOME/etx"

log "background: vLLM image, python envs, model downloads"
(sudo docker pull -q $IMG > ~/vlnext/pull.log 2>&1; echo PULL-DONE >> ~/vlnext/pull.log) &
(
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
print('DL-DONE $n')" > ~/vlnext/dl_$n.log 2>&1 &
  done
  pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch > ~/vlnext/torch.log 2>&1
  wait; echo ENV-DONE > ~/vlnext/env.log
) > ~/vlnext/setup.log 2>&1 &

log "1. synchronisation microbenchmarks (the deciding measurement)"
hipcc -O3 -std=c++17 --offload-arch=gfx942 bench/sync/sync_bench.hip -o ~/vlnext/sync_bench > ~/vlnext/sync_build.log 2>&1 || { echo "sync_bench BUILD FAILED"; cat ~/vlnext/sync_build.log; }
for w in pingpong barrier boundary; do timeout 300 stdbuf -oL ~/vlnext/sync_bench $w > ~/vlnext/sync_$w.log 2>&1; echo "[$w] rc=$?"; grep -E "RESULT|HUNG|error" ~/vlnext/sync_$w.log; done
timeout 90 stdbuf -oL ~/vlnext/sync_bench gws > ~/vlnext/sync_gws.log 2>&1; echo "[gws] rc=$?"; grep -E "RESULT|error" ~/vlnext/sync_gws.log
UC_STALE=$(grep "RESULT pingpong mode=uc" ~/vlnext/sync_pingpong.log | grep -oE "stale words [0-9]+" | awk '{s+=$3} END {print s+0}')
UC_BAR=$(grep "RESULT barrier  mode=uc    relay=0 L2 load   0" ~/vlnext/sync_barrier.log | grep -oE "[0-9.]+ us per barrier" | awk '{print $1}')
DEV_BAR=$(grep "RESULT barrier  mode=dev   relay=0 L2 load   0" ~/vlnext/sync_barrier.log | grep -oE "[0-9.]+ us per barrier" | awk '{print $1}')
UC_GO=0; [ "${UC_STALE:-1}" = 0 ] && [ -n "$UC_BAR" ] && [ -n "$DEV_BAR" ] && awk -v u="$UC_BAR" -v d="$DEV_BAR" 'BEGIN {exit !(u < d)}' && UC_GO=1
log "GATE: uncached stale words ${UC_STALE:-?}, barrier uncached ${UC_BAR:-?} us vs device-fence ${DEV_BAR:-?} us -> UC_GO=$UC_GO"

log "2. ceiling: vLLM kernel traces (custom_ops=all: the kernels ETX imports)"
until grep -q ENV-DONE ~/vlnext/env.log 2>/dev/null && grep -q PULL-DONE ~/vlnext/pull.log; do sleep 10; done
for n in qwen2.5-1.5b qwen3-8b; do tail -1 ~/vlnext/dl_$n.log; done
mkdir -p examples/vllm_llm/results; chmod 777 examples/vllm_llm/results ~/vlnext
for n in qwen2.5-1.5b qwen3-8b; do
  cp examples/llm/results/golden_c1024_$n.txt ~/llm/$n/golden_c1024.txt
  timeout 900 $DOCKER $IMG bash -c "rocprofv3 --kernel-trace --output-format csv -d $HOME/vlnext/prof_$n -- python3 examples/vllm_llm/vllm_profile.py run --model ~/llm/$n/hf --golden ~/llm/$n/golden_c1024.txt --custom-ops all > $HOME/vlnext/profrun_$n.log 2>&1; tail -3 $HOME/vlnext/profrun_$n.log; python3 examples/vllm_llm/vllm_profile.py analyze $HOME/vlnext/prof_$n --json examples/vllm_llm/results/prof_all_$n.json" > ~/vlnext/prof_$n.out 2>&1
  echo "[prof $n] rc=$?"; grep -vE "^\s*$|simple_timer" ~/vlnext/prof_$n.out | head -34
done

log "3. prep and builds (parallel)"
. ~/llmenv/bin/activate
V="VLLM=$HOME/vllm ETX_TORCH_HEADERS=$HOME/rocm-headers/pytorch"
SLOTS1="attn_mfma4_g4=1,attn_mfma16_g6=1,attn_reduce=1,rms_norm_3d=1,reshape_and_cache=1"
VARIANTS="base xcd xcd_s1 chain"; [ $UC_GO = 1 ] && VARIANTS="$VARIANTS uc uc_s1"
for n in qwen2.5-1.5b qwen3-8b; do
  python3 examples/vllm_llm/prep.py --src ~/llm/$n/hf --out ~/llm/$n --maxlen 1072 > ~/vlnext/prep_$n.log 2>&1; tail -1 ~/vlnext/prep_$n.log
  for v in $VARIANTS; do
    case $v in
      base)   E="ETX_VL_CHECK=1";;
      xcd)    E="ETX_VL_XCD=1 ETX_VL_CHAIN=1";;
      xcd_s1) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_SLOTS=$SLOTS1";;
      chain)  E="ETX_VL_CHAIN=1";;
      uc)     E="ETX_VL_UC=1 ETX_VL_XCD=1 ETX_VL_CHAIN=1";;
      uc_s1)  E="ETX_VL_UC=1 ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_SLOTS=$SLOTS1";;
    esac
    env $V $E ETX_OUT=build/vl/${n}_$v bash examples/vllm_llm/build.sh ~/llm/$n > ~/vlnext/build_${n}_$v.log 2>&1 &
  done
  wait
  for v in $VARIANTS; do echo "[$n $v] $(grep -E 'Function Name: etx_megakernel|built|error|failed' ~/vlnext/build_${n}_$v.log | head -3 | tr '\n' ' ' | cut -c1-240)"; done
done

log "4. correctness and per-kernel timing against vLLM's originals (layers 0-1)"
for n in qwen2.5-1.5b qwen3-8b; do
  B=build/vl/${n}_base
  timeout 900 stdbuf -oL $B/run ~/llm/$n $B/vl_plan.txt --check 2 > ~/vlnext/check_$n.log 2>&1; echo "[check $n] rc=$?"
  grep -E "HANG|DIFFERS|CHECK|original .* imported .*%|total|ETX\)" ~/vlnext/check_$n.log | tail -16
done

log "5. full runs (32 tokens, 2 repeats each; full logs)"
for n in qwen2.5-1.5b qwen3-8b; do
  for v in $VARIANTS; do for mode in fused unfused; do
    [ $mode = unfused ] && [ $v != base ] && [ $v != xcd_s1 ] && continue
    R=build/vl/${n}_$v; f=$([ $mode = unfused ] && echo --unfused)
    [ -x $R/run ] || { echo "[$n $v $mode] not built"; continue; }
    timeout 900 stdbuf -oL $R/run ~/llm/$n $R/vl_plan.txt --repeat 2 $f > ~/vlnext/run_${n}_${v}_$mode.log 2>&1; rc=$?
    echo "[$n $v $mode] rc=$rc $(grep -E '^run |^RESULT|failed|rror' ~/vlnext/run_${n}_${v}_$mode.log | tr '\n' ' ' | cut -c1-300)"
  done; done
done

if [ "${VL_MOE:-1}" = 1 ]; then
  log "6. MoE capture: vLLM's Triton cache and kernel trace for Qwen3-30B-A3B"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-30B-A3B', local_dir='$HOME/llm/qwen3-30b-a3b/hf', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model'], max_workers=16)
print('DL-DONE')" > ~/vlnext/dl_moe.log 2>&1; tail -1 ~/vlnext/dl_moe.log
  cp examples/llm/results/golden_c1024_qwen3-30b-a3b.txt ~/llm/qwen3-30b-a3b/golden_c1024.txt
  mkdir -p ~/vlnext/moe && chmod 777 ~/vlnext/moe
  timeout 1200 $DOCKER -e TRITON_CACHE_DIR=$HOME/vlnext/moe/triton $IMG bash -c "rocprofv3 --kernel-trace --output-format csv -d $HOME/vlnext/moe/prof -- python3 examples/vllm_llm/vllm_profile.py run --model ~/llm/qwen3-30b-a3b/hf --golden ~/llm/qwen3-30b-a3b/golden_c1024.txt > $HOME/vlnext/moe/profrun.log 2>&1; tail -2 $HOME/vlnext/moe/profrun.log; python3 examples/vllm_llm/vllm_profile.py analyze $HOME/vlnext/moe/prof --json examples/vllm_llm/results/prof_none_qwen3-30b-a3b.json" > ~/vlnext/moe/prof.out 2>&1
  echo "[moe] rc=$?"; grep -vE "^\s*$|simple_timer" ~/vlnext/moe/prof.out | head -34
  sudo chown -R $USER ~/vlnext; find ~/vlnext/moe/triton -name "fused_moe_kernel*" | head
fi
sudo chown -R $USER ~/vlnext examples/vllm_llm/results 2>/dev/null; (cd ~ && tar czf vlnext.tgz --exclude='vlnext/prof_*' --exclude='*.o' vlnext 2>/dev/null); ls -la ~/vlnext.tgz
log "ALL DONE"
