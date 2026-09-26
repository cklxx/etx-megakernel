#!/usr/bin/env bash
# Local pre-flight for vm_next.sh (CLAUDE.md section 4): every model x variant it builds, compiled for gfx942 with the
# local LLVM and the ROCm device libraries; prints tasks/events, VGPRs, spills, LDS, the UC plan check. No GPU needed.
#   bash examples/vllm_llm/local_matrix.sh [outdir]
set -uo pipefail
cd "$(dirname "$0")/../.."
S=${1:-/tmp/etx_matrix}; mkdir -p $S
R=${ROCM_HEADERS:-$HOME/code/rocm-headers}; DL=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
PY=.venv/bin/python; L=${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}
SLOTS1="attn_mfma4_g4=1,attn_mfma16_g6=1,attn_reduce=1,rms_norm_3d=1,reshape_and_cache=1"
one() {
  local n=$1 v=$2 O=$S/${1}_$2; mkdir -p $O
  local cfg=examples/llm/configs/$n.json E=""
  case $n in qwen3-8b) BLK="--block rms_norm_2d=512 --block reshape_and_cache=128"; EXP=wvsplitk_y2,attn_mfma4_g4,attn_reduce,rms_norm_2d,fused_add_rms_norm,rope_neox,reshape_and_cache,silu_and_mul,rms_norm_3d;;
             qwen2.5-1.5b) BLK="--block rms_norm_2d=192 --block reshape_and_cache=32"; EXP=wvsplitk_y2,attn_mfma16_g6,attn_reduce,rms_norm_2d,fused_add_rms_norm,rope_neox,reshape_and_cache,silu_and_mul;; esac
  local SL="" UCF=""
  case $v in base) E="";; xcd) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1";; xcd_s1) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1"; SL=$SLOTS1;; chain) E="ETX_VL_CHAIN=1";;
             uc) E="ETX_VL_UC=1 ETX_VL_PIN=1 ETX_VL_XCD=1 ETX_VL_CHAIN=1"; UCF="-DETX_VL_UC -DETX_COHERENT_DEVICE_DATA";;
             uc_s1) E="ETX_VL_UC=1 ETX_VL_PIN=1 ETX_VL_XCD=1 ETX_VL_CHAIN=1"; SL=$SLOTS1; UCF="-DETX_VL_UC -DETX_COHERENT_DEVICE_DATA";; esac
  { $PY -m etx.importer.vllm_kernels --out $O/imp --only $EXP $BLK --slots "$SL" \
    && $PY -m etx.importer.adapter $O/imp/imports.json --tiles $O/imp_tiles.hip --header $O/imports.h \
    && $PY -m etx.importer.bundle --imports $O/imp --exports $EXP --devlibs $DL -o $O/imports.bc \
    && env $E ETX_VL_IMPORTS=$O/imp/imports.json ETX_VL_ADAPTERS=$O/imp_tiles.hip ETX_VL_PLAN=$O/vl_plan.txt ETX_VL_CHAINS=$O/chain_tiles.hip ETX_LLM_CONFIG=$cfg \
         $PY -m etx compile examples/vllm_llm/model.py --arch gfx942 --out $O --inline-tiles > $O/compile.log 2>&1 \
    && { [ -z "$UCF" ] || $PY examples/vllm_llm/check_uc_plan.py $O/plan.json; } \
    && ETX_DEVLIBS=$DL bash etx/importer/hip_ir_local.sh $O/megakernel_d0.hip $O/mk.ll $UCF -Ietx/runtime/include -I $O -I . -Xclang -mlink-builtin-bitcode -Xclang $O/imports.bc \
    && $L/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 $O/mk.ll -o $O/mk.s \
    && $L/clang++ -x hip --cuda-host-only -fsyntax-only -nogpuinc -nogpulib --offload-arch=gfx942 -std=c++17 -isystem $R/clang -isystem $R/inc \
         -include __clang_hip_runtime_wrapper.h -isystem etx/importer/shim/mac -D__HIP_PLATFORM_AMD__ $UCF $([ $v = base ] && echo -DETX_VL_CHECK) \
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
}
for n in qwen3-8b qwen2.5-1.5b; do
  for v in base chain xcd xcd_s1 uc uc_s1; do one $n $v & done
  wait
done
