#!/usr/bin/env bash
# The next GPU session, in the order CLAUDE.md prescribes: the deciding microbenchmarks first (synchronisation,
# overlap), then the ceiling (vLLM's kernel trace), then correctness, then the variants, then DeepSeek-V2-Lite (fleet's
# tiles under ETX: trace, schedule analysis, vLLM at batch 1/2/4), the MoE capture last. Every step logs in full to
# ~/vlnext/ and prints the elapsed time; nothing is filtered away.
#   rsync ~/code/etx and ~/code/fleet-mi300x to ~/etx and ~/fleet-mi300x on the VM (without .git / build / .venv), then
#   nohup bash ~/etx/examples/vllm_llm/vm_next.sh > ~/next.log 2>&1 &
#
# Gate (decided before the session): the uncached variants are built and run only if the microbenchmark shows
#   (1) uncached hand-off with DOMAIN fences: 0 stale words across XCDs, and
#   (2) its whole-GPU barrier is faster than today's cached + device-fence barrier.
# Budget: about 3 h of a 1x MI300X ($9); VL_MOE=0 skips the MoE capture (about 30 min), VL_DS=0 the DeepSeek step
# (about 40 min).
# Decisions this session feeds (written down before it, docs/PLAN-v3.md 4.4): the DeepSeek re-layout is built only if
# examples/dsv2lite/whatif.py, fed the overlap log, estimates >= 5% per token; two-request interleaving only if the fold
# slows by less than the batch2 break-even it prints AND it beats vLLM at batch 2 in tokens/s.
set -uo pipefail
cd ~/etx; mkdir -p ~/vlnext; T0=$(date +%s)
log() { printf '\n=== %s (+%d min) %s\n' "$(date -u +%H:%M:%S)" $(( ($(date +%s) - T0) / 60 )) "$*"; }
IMG=${VLLM_IMAGE:-rocm/vllm:latest}
DOCKER="sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 -v $HOME:$HOME -w $HOME/etx"

log "background: vLLM image, python envs, model downloads"
(sudo docker pull -q $IMG > ~/vlnext/pull.log 2>&1; echo PULL-DONE >> ~/vlnext/pull.log) &
(
  sudo apt-get install -y -qq python3.12-venv > /tmp/apt.log 2>&1 || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv >> /tmp/apt.log 2>&1; }
  .venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }   # a copied-in foreign venv is rebuilt
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
# fleet's environment (torch + transformers < 5 + the flash_attn stub, as its bootstrap) and the DeepSeek weights;
# waits for the venv package the first subshell installs (two apt-get at once broke a venv once)
(
  [ -d ~/fleet-mi300x ] || { echo "NO ~/fleet-mi300x: DeepSeek step will be skipped"; echo FLEET-ENV-DONE; exit 0; }
  until python3 -c "import ensurepip" 2>/dev/null; do sleep 5; done
  cd ~/fleet-mi300x && { .venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv .venv; }; } && . .venv/bin/activate
  pip install -q --upgrade pip; pip install -q numpy "huggingface_hub[hf_transfer]"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', local_dir='$HOME/models/dsv2-lite-base',
                  allow_patterns=['*.json', '*.safetensors', '*.py', 'tokenizer*'], max_workers=8)
print('DL-DONE dsv2-lite')" > ~/vlnext/dl_ds.log 2>&1 &
  pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch
  pip install -q "transformers>=4.39,<5" "tokenizers<0.20" accelerate safetensors
  SP=$(python3 -c "import site;print(site.getsitepackages()[0])")
  mkdir -p "$SP/flash_attn" && echo "# stub: satisfies transformers check_imports only" > "$SP/flash_attn/__init__.py"
  wait; echo FLEET-ENV-DONE
) > ~/vlnext/fleet_env.log 2>&1 &

log "1. synchronisation microbenchmarks (the deciding measurement)"
hipcc -O3 -std=c++17 --offload-arch=gfx942 bench/sync/sync_bench.hip -o ~/vlnext/sync_bench > ~/vlnext/sync_build.log 2>&1 || { echo "sync_bench BUILD FAILED"; cat ~/vlnext/sync_build.log; }
for w in pingpong barrier boundary; do timeout 300 stdbuf -oL ~/vlnext/sync_bench $w > ~/vlnext/sync_$w.log 2>&1; echo "[$w] rc=$?"; grep -E "RESULT|HUNG|error" ~/vlnext/sync_$w.log; done
UC_STALE=$(grep "RESULT pingpong mode=uc" ~/vlnext/sync_pingpong.log | grep -oE "stale words [0-9]+" | awk '{s+=$3} END {print s+0}')
UC_BAR=$(grep "RESULT barrier  mode=uc    relay=0 L2 load   0" ~/vlnext/sync_barrier.log | grep -oE "[0-9.]+ us per barrier" | awk '{print $1}')
DEV_BAR=$(grep "RESULT barrier  mode=dev   relay=0 L2 load   0" ~/vlnext/sync_barrier.log | grep -oE "[0-9.]+ us per barrier" | awk '{print $1}')
UC_BAD=$(grep -cE "mode=uc .*(NO PROGRESS|HUNG|NOT AS INTENDED)" ~/vlnext/sync_pingpong.log ~/vlnext/sync_barrier.log | awk -F: '{s+=$2} END {print s+0}')
UC_BAR_STALE=$(grep "RESULT barrier  mode=uc" ~/vlnext/sync_barrier.log | grep -oE "stale [0-9]+" | awk '{s+=$2} END {print s+0}')
UC_N=$(grep -c "RESULT pingpong mode=uc .*stale words" ~/vlnext/sync_pingpong.log); UC_NB=$(grep -c "RESULT barrier  mode=uc .*stale" ~/vlnext/sync_barrier.log)
# a missing line (crash, timeout) must not read as "0 stale": all 3 uncached hand-offs and all 4 uncached barriers must have reported
UC_GO=0; [ "$UC_N" = 3 ] && [ "$UC_NB" = 4 ] && [ "${UC_STALE:-1}" = 0 ] && [ "$UC_BAR_STALE" = 0 ] && [ "$UC_BAD" = 0 ] && [ -n "$UC_BAR" ] && [ -n "$DEV_BAR" ] && awk -v u="$UC_BAR" -v d="$DEV_BAR" 'BEGIN {exit !(u < d)}' && UC_GO=1
log "GATE: uncached results ${UC_N}/3 hand-off ${UC_NB}/4 barrier; stale words ${UC_STALE:-?} (hand-off) + ${UC_BAR_STALE:-?} (barrier), failed runs ${UC_BAD:-?}, barrier uncached ${UC_BAR:-?} us vs device-fence ${DEV_BAR:-?} us -> UC_GO=$UC_GO"

log "1b. overlap microbenchmarks: Infinity Cache read rate, and how much a latency-bound fold slows while the same CUs stream"
hipcc -O3 -std=c++17 --offload-arch=gfx942 bench/overlap/overlap_bench.hip -o ~/vlnext/overlap_bench > ~/vlnext/overlap_build.log 2>&1 || { echo "overlap_bench BUILD FAILED"; cat ~/vlnext/overlap_build.log; }
timeout 300 stdbuf -oL ~/vlnext/overlap_bench all > ~/vlnext/overlap.log 2>&1; echo "[overlap] rc=$?"; grep -E "RESULT|HUNG|error" ~/vlnext/overlap.log

log "2. ceiling: vLLM kernel traces (custom_ops=all: the kernels ETX imports)"
until grep -q ENV-DONE ~/vlnext/env.log 2>/dev/null && grep -q PULL-DONE ~/vlnext/pull.log; do sleep 10; done
for n in qwen2.5-1.5b qwen3-8b; do tail -1 ~/vlnext/dl_$n.log; done
mkdir -p examples/vllm_llm/results; chmod 777 examples/vllm_llm/results ~/vlnext
for n in qwen2.5-1.5b qwen3-8b; do
  cp examples/llm/results/golden_c1024_$n.txt ~/llm/$n/golden_c1024.txt
  timeout 900 $DOCKER $IMG bash -c "rocprofv3 --kernel-trace --output-format csv -d $HOME/vlnext/prof_$n -- python3 examples/vllm_llm/vllm_profile.py run --model $HOME/llm/$n/hf --golden $HOME/llm/$n/golden_c1024.txt --custom-ops all > $HOME/vlnext/profrun_$n.log 2>&1; tail -3 $HOME/vlnext/profrun_$n.log; python3 examples/vllm_llm/vllm_profile.py analyze $HOME/vlnext/prof_$n --json examples/vllm_llm/results/prof_all_$n.json" > ~/vlnext/prof_$n.out 2>&1
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
    timeout 600 stdbuf -oL $R/run ~/llm/$n $R/vl_plan.txt --repeat 2 $f > ~/vlnext/run_${n}_${v}_$mode.log 2>&1; rc=$?
    echo "[$n $v $mode] rc=$rc $(grep -E '^run |^RESULT|failed|rror' ~/vlnext/run_${n}_${v}_$mode.log | tr '\n' ' ' | cut -c1-300)"
  done; done
done

if [ "${VL_DS:-1}" = 1 ] && [ -d ~/fleet-mi300x ]; then
  log "6. DeepSeek-V2-Lite: fleet reference + weights, ETX build (fleet's tiles), correctness + timing, trace, schedule analysis"
  until grep -q FLEET-ENV-DONE ~/vlnext/fleet_env.log 2>/dev/null; do sleep 10; done
  tail -1 ~/vlnext/dl_ds.log
  ( cd ~/fleet-mi300x && . .venv/bin/activate && mkdir -p build \
    && timeout 1200 stdbuf -oL python3 src/host/reference_run.py --model $HOME/models/dsv2-lite-base --out build/golden.npz > ~/vlnext/ds_reference.log 2>&1 \
    && timeout 900 stdbuf -oL python3 src/host/pack_weights.py --model $HOME/models/dsv2-lite-base --out build/weights.bin > ~/vlnext/ds_pack.log 2>&1 )
  echo "[ds prep] rc=$?"; tail -2 ~/vlnext/ds_reference.log; tail -1 ~/vlnext/ds_pack.log
  ETX_OUT=$HOME/etx/build/ds bash examples/dsv2lite/build.sh ~/fleet-mi300x -Rpass-analysis=kernel-resource-usage > ~/vlnext/ds_build.log 2>&1 || echo "DS BUILD FAILED"
  grep -A12 "Function Name: etx_megakernel_d0" ~/vlnext/ds_build.log | grep -E "VGPRs|AGPRs|Scratch|Occupancy" | tr -s ' ' | head -4; cp build/ds/plan.json ~/vlnext/ds_plan.json 2>/dev/null
  (cd ~/fleet-mi300x && timeout 600 stdbuf -oL ~/etx/build/ds/run --tokens 32 --context 1024 --repeat 2 > ~/vlnext/ds_run.log 2>&1); echo "[ds run] rc=$?"
  grep -E "^per-token|^tokens:|consecutive layers" ~/vlnext/ds_run.log
  # trace of the last of 8 tokens (one row per task), then the schedule analysis: layer 5 and the whole token
  (cd ~/fleet-mi300x && ETX_TRACE=1 ETX_TRACE_RAW=$HOME/vlnext/ds_trace.bin timeout 600 stdbuf -oL ~/etx/build/ds/run --tokens 8 --context 1024 > ~/vlnext/ds_trace_run.log 2>&1); echo "[ds trace] rc=$?"
  .venv/bin/python -m examples.dsv2lite.bytes > ~/vlnext/ds_bytes.json
  .venv/bin/python -m etx.tools.timeline ~/vlnext/ds_plan.json ~/vlnext/ds_trace.bin --bytes ~/vlnext/ds_bytes.json --grids '_5$' \
      --html ~/vlnext/ds_L5.html --json ~/vlnext/ds_L5.json > ~/vlnext/ds_L5.txt 2>&1; cat ~/vlnext/ds_L5.txt
  .venv/bin/python -m etx.tools.timeline ~/vlnext/ds_plan.json ~/vlnext/ds_trace.bin --bytes ~/vlnext/ds_bytes.json \
      --html ~/vlnext/ds_token.html --json ~/vlnext/ds_token.json > ~/vlnext/ds_token.txt 2>&1; sed -n '/^per XCD/,/^$/p;/^critical path:/p;/^memory:/,$p' ~/vlnext/ds_token.txt
  .venv/bin/python -m examples.dsv2lite.whatif --overlap ~/vlnext/overlap.log > ~/vlnext/whatif.txt 2>&1; cat ~/vlnext/whatif.txt
  # vLLM on the same prompt at batch 1 / 2 / 4 (distinct prompts): AITER off and on at batch 1, the faster one for 2 and 4
  python3 -c "
import numpy as np; z = np.load('$HOME/fleet-mi300x/build/golden.npz')
print('prompt', *z['prompt_ids'].tolist()); print('gen', *z['token_ids'].tolist())" > ~/llm/ds_golden.txt
  best=0; bestms=1e9
  for A in 0 1; do
    timeout 900 $DOCKER -e VLLM_ROCM_USE_AITER=$A $IMG python3 examples/llm/vllm_bench.py --model $HOME/models/dsv2-lite-base --golden $HOME/llm/ds_golden.txt \
      --label dsv2-lite --batch 1 --json $HOME/vlnext/vllm_ds_b1_a$A.json > ~/vlnext/vllm_ds_b1_a$A.log 2>&1; echo "[vllm ds b1 aiter=$A] rc=$?"; grep -E "^VLLM|rror" ~/vlnext/vllm_ds_b1_a$A.log | tail -2
    ms=$(grep -oE "median_ms=[0-9.]+" ~/vlnext/vllm_ds_b1_a$A.log | cut -d= -f2); [ -n "$ms" ] && awk -v a="$ms" -v b="$bestms" 'BEGIN {exit !(a < b)}' && { best=$A; bestms=$ms; }
  done
  for B in 2 4; do
    timeout 900 $DOCKER -e VLLM_ROCM_USE_AITER=$best $IMG python3 examples/llm/vllm_bench.py --model $HOME/models/dsv2-lite-base --golden $HOME/llm/ds_golden.txt \
      --label dsv2-lite --batch $B --json $HOME/vlnext/vllm_ds_b$B.json > ~/vlnext/vllm_ds_b$B.log 2>&1; echo "[vllm ds b$B aiter=$best] rc=$?"; grep -E "^VLLM|rror" ~/vlnext/vllm_ds_b$B.log | tail -2
  done
fi

if [ "${VL_MOE:-1}" = 1 ]; then
  log "7. MoE capture: vLLM's Triton cache and kernel trace for Qwen3-30B-A3B, then vLLM at batch 1 / 2 / 4"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-30B-A3B', local_dir='$HOME/llm/qwen3-30b-a3b/hf', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model'], max_workers=16)
print('DL-DONE')" > ~/vlnext/dl_moe.log 2>&1; tail -1 ~/vlnext/dl_moe.log
  cp examples/llm/results/golden_c1024_qwen3-30b-a3b.txt ~/llm/qwen3-30b-a3b/golden_c1024.txt
  mkdir -p ~/vlnext/moe && chmod 777 ~/vlnext/moe
  timeout 1200 $DOCKER -e TRITON_CACHE_DIR=$HOME/vlnext/moe/triton $IMG bash -c "rocprofv3 --kernel-trace --output-format csv -d $HOME/vlnext/moe/prof -- python3 examples/vllm_llm/vllm_profile.py run --model $HOME/llm/qwen3-30b-a3b/hf --golden $HOME/llm/qwen3-30b-a3b/golden_c1024.txt > $HOME/vlnext/moe/profrun.log 2>&1; tail -2 $HOME/vlnext/moe/profrun.log; python3 examples/vllm_llm/vllm_profile.py analyze $HOME/vlnext/moe/prof --json examples/vllm_llm/results/prof_none_qwen3-30b-a3b.json" > ~/vlnext/moe/prof.out 2>&1
  echo "[moe] rc=$?"; grep -vE "^\s*$|simple_timer" ~/vlnext/moe/prof.out | head -34
  sudo chown -R $USER ~/vlnext; find ~/vlnext/moe/triton -name "fused_moe_kernel*" | head
  for B in 1 2 4; do
    timeout 900 $DOCKER $IMG python3 examples/llm/vllm_bench.py --model $HOME/llm/qwen3-30b-a3b/hf --golden $HOME/llm/qwen3-30b-a3b/golden_c1024.txt \
      --label qwen3-30b-a3b --batch $B --json $HOME/vlnext/vllm_moe_b$B.json > ~/vlnext/vllm_moe_b$B.log 2>&1; echo "[vllm moe b$B] rc=$?"; grep -E "^VLLM|rror" ~/vlnext/vllm_moe_b$B.log | tail -2
  done
fi
log "8. GWS hardware barrier (last: an unsupported GWS could leave the queue in a bad state)"
timeout 90 stdbuf -oL ~/vlnext/sync_bench gws > ~/vlnext/sync_gws.log 2>&1; echo "[gws] rc=$?"; grep -E "RESULT|error" ~/vlnext/sync_gws.log
sudo chown -R $USER ~/vlnext examples/vllm_llm/results 2>/dev/null; (cd ~ && tar czf vlnext.tgz --exclude='vlnext/prof_*' --exclude='*.o' vlnext 2>/dev/null); ls -la ~/vlnext.tgz
log "ALL DONE"
