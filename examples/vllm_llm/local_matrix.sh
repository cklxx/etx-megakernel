#!/usr/bin/env bash
# Local pre-flight for vm_next.sh (CLAUDE.md section 4): every model x variant it builds, compiled for gfx942 with the
# local LLVM and the ROCm device libraries; prints tasks/events, VGPRs, spills, LDS, the UC plan check. No GPU needed.
#   bash examples/vllm_llm/local_matrix.sh [outdir]
set -uo pipefail
# A matrix label describes a fixed variant, not the caller's last experiment.
# Preserve toolchain locations, but discard model and DSL experiment overrides.
for variable in ${!ETX_VL_@} ${!ETX_LLM_@} ${!ETX_DSL_@} ${!DSL_ABLATE_@}; do
  unset "$variable"
done
cd "$(dirname "$0")/../.."
S=${1:-/tmp/etx_matrix}; mkdir -p $S
R=${ETX_ROCM_HEADERS:-${ROCM_HEADERS:-$HOME/code/rocm-headers}}
DL=${ETX_DEVLIBS:-$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode}
PY=${ETX_PYTHON:-.venv/bin/python}; L=${ETX_LLVM:-${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}}
VLLM=${ETX_VLLM:-$HOME/code/vllm}
export ETX_LLVM="$L" ETX_ROCM_HEADERS="$R"
SLOTS1="attn_mfma4_g4=1,attn_mfma16_g6=1,attn_reduce=1,rms_norm_3d=1,reshape_and_cache=1"
one() {
  local n=$1 v=$2 O=$S/${1}_$2; mkdir -p $O
  local cfg=examples/llm/configs/$n.json E=""
  case $n in qwen3-8b) BLK="--block rms_norm_2d=512 --block reshape_and_cache=128"; EXP=wvsplitk_y2,attn_mfma4_g4,attn_reduce,rms_norm_2d,fused_add_rms_norm,rope_neox,reshape_and_cache,silu_and_mul,rms_norm_3d;;
             qwen2.5-1.5b) BLK="--block rms_norm_2d=192 --block reshape_and_cache=32"; EXP=wvsplitk_y2,attn_mfma16_g6,attn_reduce,rms_norm_2d,fused_add_rms_norm,rope_neox,reshape_and_cache,silu_and_mul;;
             qwen3-30b-a3b) BLK="--block rms_norm_2d=256 --block reshape_and_cache=64 --block silu_and_mul=96"
                            EXP=wvsplitk_y2,wvsplitk_y1,attn_mfma16_g8,attn_reduce,rms_norm_2d,fused_add_rms_norm,rope_neox,reshape_and_cache,silu_and_mul,rms_norm_3d,topk_softmax_e128,moe_sum_k8,fused_moe_up,fused_moe_down;;
             mixtral-8x7b) BLK="--block rms_norm_2d=512 --block reshape_and_cache=128 --block silu_and_mul=1024"
                            EXP=wvsplitk_y2,wvsplitk_y1,attn_mfma4_g4,attn_reduce,rms_norm_2d,fused_add_rms_norm,rope_neox,reshape_and_cache,silu_and_mul,topk_softmax_e8,moe_sum_k2,fused_moe_up_e8,fused_moe_down_e8;;
             *) echo "unsupported model: $n" >&2; return 2;; esac
  local MS=""; [ $n = qwen3-30b-a3b ] && MS="fused_moe_up=1,fused_moe_down=1,silu_and_mul=1"
  [ "$n" = mixtral-8x7b ] && MS="fused_moe_up_e8=1,fused_moe_down_e8=1,silu_and_mul=1"
  local SL="" UC=0
  case $v in base) E="";; xcd) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1";; xcd_s1) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1"; SL=$SLOTS1;; chain) E="ETX_VL_CHAIN=1";;
             uc) E="ETX_VL_UC=1 ETX_VL_PIN=1 ETX_VL_XCD=1 ETX_VL_CHAIN=1"; UC=1;;
             uc_s1) E="ETX_VL_UC=1 ETX_VL_PIN=1 ETX_VL_XCD=1 ETX_VL_CHAIN=1"; SL=$SLOTS1; UC=1;;
             *) echo "unsupported variant: $v" >&2; return 2;; esac
  env $E ETX_VL_IMPORTED=1 "$PY" -c '
import json, os, sys
with open(sys.argv[1], "w") as stream:
    json.dump({"model": sys.argv[2], "variant": sys.argv[3],
               "model_config": sys.argv[4], "overrides": {k: v for k, v in os.environ.items()
                if k.startswith(("ETX_VL_", "ETX_LLM_", "ETX_DSL_", "DSL_ABLATE_"))}},
              stream, indent=2, sort_keys=True)
    stream.write("\n")
' "$O/configuration.json" "$n" "$v" "$cfg" || return $?
  { $PY -m etx.importer.vllm_kernels --vllm "$VLLM" --out $O/imp --only $EXP $BLK --slots "${SL:+$SL,}$MS" \
    && $PY -m etx.importer.adapter $O/imp/imports.json --tiles $O/imp_tiles.hip --header $O/imports.h \
    && $PY -m etx.importer.bundle --imports $O/imp --exports $EXP --devlibs $DL -o $O/imports.bc \
    && env $E ETX_VL_IMPORTED=1 ETX_VL_IMPORTS=$O/imp/imports.json ETX_VL_ADAPTERS=$O/imp_tiles.hip ETX_VL_PLAN=$O/vl_plan.txt ETX_VL_CHAINS=$O/chain_tiles.hip ETX_LLM_CONFIG=$cfg \
         $PY -m etx compile examples/vllm_llm/model.py --arch gfx942 --out $O --inline-tiles > $O/compile.log 2>&1 \
    && { [ "$UC" != 1 ] || $PY examples/vllm_llm/check_uc_plan.py $O/plan.json; } \
    && ETX_DEVLIBS=$DL bash etx/importer/hip_ir_local.sh $O/megakernel_d0.hip $O/mk.ll -Ietx/runtime/include -I $O -I . -Xclang -mlink-builtin-bitcode -Xclang $O/imports.bc \
    && $L/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 $O/mk.ll -o $O/mk.s \
    && $L/clang++ -x hip --cuda-host-only -fsyntax-only -nogpuinc -nogpulib --offload-arch=gfx942 -std=c++17 -isystem $R/clang -isystem $R/inc \
         -include __clang_hip_runtime_wrapper.h -isystem etx/importer/shim/mac -D__HIP_PLATFORM_AMD__ $([ $v = base ] && echo -DETX_VL_CHECK) \
         -Ietx/runtime/include -I $O -I . examples/vllm_llm/host.hip; } > $O/matrix.log 2>&1
  local rc=$?
  local st=$($PY -c "
import re,sys
a=open('$O/mk.s').read() if $rc == 0 else ''
for blk in a.split('  - .agpr_count')[1:]:
    g=lambda k: re.search(rf'\.{k}:\s+(\d+)',blk).group(1)
    print(re.search(r'\.name:\s+(\S+)',blk).group(1).replace('etx_',''), 'vgpr', g('vgpr_count'), 'spill', g('vgpr_spill_count'), 'lds', g('group_segment_fixed_size'), end='; ')
" 2>/dev/null)
  echo "[$n $v] rc=$rc $(grep -E 'tasks=' $O/compile.log 2>/dev/null | head -1) $st $(grep -E 'UC plan|REJECTED' $O/matrix.log) $(grep -m2 -E 'error' $O/matrix.log | cut -c1-200)"
  return "$rc"
}
# LOCAL_MATRIX="model:variant ..." builds only those
pids=()
status=0
wait_all() {
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  pids=()
}
if [ -n "${LOCAL_MATRIX:-}" ]; then
  for mv in $LOCAL_MATRIX; do one "${mv%%:*}" "${mv##*:}" & pids+=("$!"); done
else
  for n in qwen3-8b qwen2.5-1.5b; do
    for v in base chain xcd xcd_s1 uc uc_s1; do one "$n" "$v" & pids+=("$!"); done
    wait_all
  done
  for v in base chain; do one qwen3-30b-a3b "$v" & pids+=("$!"); done
  wait_all
  for v in base chain; do one mixtral-8x7b "$v" & pids+=("$!"); done
fi
wait_all
exit "$status"
