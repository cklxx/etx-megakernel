#!/usr/bin/env bash
# The device half of examples/vllm_llm/build.sh with ROCm 7.2.4's own compiler (LLVM 22, what hipcc runs on the VM),
# on a Mac, in an x86 Linux container: import (the "hipcc" path of etx.importer.vllm_kernels), bundle, plan, and the
# megakernel's device code to assembly. Catches what the local LLVM 23 cannot: IR that LLVM 22 does not parse (Triton's
# is LLVM 23) and the VM's register allocation (VGPRs, spills).
#   [ETX_CFLAGS="-D..."] bash examples/vllm_llm/rocm_build_check.sh <model> [variant] [outdir]
#   model: qwen3-8b | qwen2.5-1.5b | qwen3-30b-a3b
# Needs: docker (OrbStack), the image etx-rocm22 (docker build --platform linux/amd64 -t etx-rocm22 -f
# examples/vllm_llm/rocm_build_check.Dockerfile .), and under
# ~/code/rocm-headers: rocm-llvm/ (the rocm-llvm 7.2.4 .deb unpacked), inc/ (HIP headers), devlibs/, pytorch/.
set -euo pipefail
M=${1:?model}; V=${2:-base}; OUT=${3:-$HOME/code/etx/build/rocm22/${M}_$V}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
case $V in base) E="";; chain) E="ETX_VL_CHAIN=1";; xcd) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1";;
  xcd2) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_REQUESTS=2"; ETX_CFLAGS="${ETX_CFLAGS:-} -DETX_VL_REQUESTS=2";;
  xcd_s1) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_SLOTS=attn_mfma16_g8=1,attn_mfma16_g6=1,attn_mfma4_g4=1,attn_reduce=1,rms_norm_3d=1,reshape_and_cache=1";;
  xcd_qknr) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_QKNR=1";;
  xcd_qknr_s1) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_QKNR=1 ETX_VL_SLOTS=qknorm_rope=1,attn_mfma16_g8=1,attn_reduce=1";;
  base_qknr) E="ETX_VL_QKNR=1";;
  xcd_qknr_grid) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_QKNR=grid";;
  xcd_dslgemv) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_QKNR=grid ETX_VL_DSL_GEMV=1";;
  *) echo "variant $V"; exit 1;; esac
mkdir -p "$OUT"
# paths as the container sees them: $HOME/code -> /code
C_ROOT=/code/${ROOT#$HOME/code/}; C_OUT=/code/${OUT#$HOME/code/}
case "$OUT" in "$HOME/code/"*) ;; *) echo "outdir must be under ~/code (mounted into the container)"; exit 1;; esac
docker run --rm --platform linux/amd64 -v "$HOME/code:/code" -v "$HOME/code:$HOME/code" -e M="$M" -e E="$E" -e ROOT="$C_ROOT" -e OUT="$C_OUT" -e CFLAGS="${ETX_CFLAGS:-}" etx-rocm22 bash -c '
set -euo pipefail
R=/code/rocm-headers; RL=$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm; DL=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
mkdir -p /tmp/bin
# hipcc stand-in: ROCm clang with the HIP headers, as hipcc invokes it for device-only compiles
cat > /tmp/bin/hipcc <<EOF
#!/usr/bin/env bash
exec $RL/bin/clang++ -nogpuinc -isystem $R/inc -include __clang_hip_runtime_wrapper.h "\$@"
EOF
chmod +x /tmp/bin/hipcc; export PATH=/tmp/bin:$PATH ETX_LLVM=$RL/bin ETX_TORCH_HEADERS=$R/pytorch
cd $ROOT; CFG=examples/llm/configs/$M.json
cfg() { python3 -c "import json;c=json.load(open(\"$CFG\"));print(c.get(\"$1\",0))"; }
H=$(cfg hidden_size); NKV=$(cfg num_key_value_heads); NH=$(cfg num_attention_heads); HD=$(cfg head_dim); [ "$HD" = 0 ] && HD=$((H / NH))
G=$((NH / NKV)); case $G in 4) ATTN=attn_mfma4_g4;; 6) ATTN=attn_mfma16_g6;; 8) ATTN=attn_mfma16_g8;; esac
EXP="wvsplitk_y2,$ATTN,attn_reduce,rms_norm_2d,fused_add_rms_norm,rope_neox,reshape_and_cache,silu_and_mul"
case $M in qwen3*) EXP="$EXP,rms_norm_3d";; esac
case " $E " in *ETX_VL_QKNR=1*|*ETX_VL_QKNR=grid*) EXP="$EXP,qknorm_rope";; esac   # as build.sh: any value but 0
BLK=(--block rms_norm_2d=$((H / 8)) --block reshape_and_cache=$((NKV * HD / 8))); SL=""
if [ "$(cfg num_experts)" != 0 ]; then
  EXP="$EXP,wvsplitk_y1,topk_softmax_e128,moe_sum_k8,fused_moe_up,fused_moe_down"
  BLK+=(--block silu_and_mul=$(( $(cfg moe_intermediate_size) / 8 ))); SL="fused_moe_up=1,fused_moe_down=1,silu_and_mul=1"
fi
echo "== import ($EXP)"
XS=$(echo "$E" | tr " " "\n" | sed -n "s/^ETX_VL_SLOTS=//p"); SL="${XS:+$XS,}$SL"; SL="${SL%,}"
python3 -m etx.importer.vllm_kernels --vllm /code/vllm --out $OUT/imp --compiler hipcc --only "$EXP" --slots "$SL" "${BLK[@]}"
DSLT=""; case " $E " in *ETX_VL_DSL_GEMV=1*) python3 -m etx.ops.gemv $OUT/imp/imports.json --tiles $OUT/dsl_tiles.hip; DSLT="ETX_VL_DSL_TILES=$OUT/dsl_tiles.hip";; esac
python3 -m etx.importer.adapter $OUT/imp/imports.json --tiles $OUT/imp_tiles.hip --header $OUT/imports.h
python3 -m etx.importer.bundle --imports $OUT/imp --exports "$EXP" --devlibs $DL -o $OUT/imports.bc
echo "== plan"
env $E $DSLT ETX_VL_IMPORTS=$OUT/imp/imports.json ETX_VL_ADAPTERS=$OUT/imp_tiles.hip ETX_VL_PLAN=$OUT/vl_plan.txt ETX_VL_CHAINS=$OUT/chain_tiles.hip ETX_LLM_CONFIG=$CFG \
  python3 -m etx compile examples/vllm_llm/model.py --arch gfx942 --out $OUT --inline-tiles > $OUT/compile.log 2>&1 || { tail -20 $OUT/compile.log; exit 1; }
grep -E "tasks=" $OUT/compile.log | head -2
echo "== device code (ROCm clang 22)"
$RL/bin/clang++ -x hip --cuda-device-only --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path=$DL \
  -isystem $R/inc -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -Ietx/runtime/include -I $OUT -I . \
  $CFLAGS -Xclang -mlink-builtin-bitcode -Xclang $OUT/imports.bc -Rpass-analysis=kernel-resource-usage -S $OUT/megakernel_d0.hip -o $OUT/mk.s 2> $OUT/resource.log \
  || { grep -m5 error $OUT/resource.log; exit 1; }
echo "== host.hip, device pass (etx/host.h is compiled in both passes)"
$RL/bin/clang++ -x hip --cuda-device-only --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path=$DL \
  -isystem $R/inc -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ $CFLAGS -DETX_VL_CHECK -Ietx/runtime/include -I $OUT -I . \
  -S examples/vllm_llm/host.hip -o /tmp/host_dev.s 2> $OUT/host_dev.log && echo "host.hip device pass ok" || { grep -m5 error $OUT/host_dev.log; exit 1; }
grep -E "Function Name|VGPRs:|AGPRs:|ScratchSize|VGPRs Spill|LDS Size|Occupancy" $OUT/resource.log | sed -E "s/.*remark: +//; s/ \[-Rpass.*//" | paste - - - - - - -
'
