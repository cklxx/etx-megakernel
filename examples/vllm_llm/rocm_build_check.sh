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
M=${1:?model}; V=${2:-base}; OUT=${3:-$HOME/code/etx/build/rocm22/${M}_${V//:/_}}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
case $V in base) E="";; chain) E="ETX_VL_CHAIN=1";; xcd) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1";;
  xcd2) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_REQUESTS=2";;
  xcd_s1) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_SLOTS=attn_mfma16_g8=1,attn_mfma16_g6=1,attn_mfma4_g4=1,attn_reduce=1,rms_norm_3d=1,reshape_and_cache=1";;
  xcd_qknr) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_QKNR=1";;
  xcd_qknr_s1) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_QKNR=1 ETX_VL_SLOTS=qknorm_rope=1,attn_mfma16_g8=1,attn_reduce=1";;
  base_qknr) E="ETX_VL_QKNR=1";;
  xcd_qknr_grid) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_QKNR=grid";;
  xcd_dslgemv) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_QKNR=grid ETX_VL_DSL_GEMV=1";;
  xcd_dsl) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_QKNR=grid ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1";;
  xcd_dsl2) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_SLOTS=attn_mfma16_g8=1";;
  xcd_dsl3) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_DSL_FOLD=1 ETX_VL_SLOTS=attn_mfma16_g8=1";;
  xcd_dsl3_b2) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1 ETX_VL_COH=1 ETX_VL_FOLD_TOPK=0 ETX_VL_CU=256 ETX_VL_REQUESTS=2 ETX_VL_BATCH=1 ETX_VL_NB=2 ETX_VL_SLOTS=attn_mfma16_g8=1";;
  xcd_dsl3x2) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_DSL_FOLD=1 ETX_VL_SLOTS=attn_mfma16_g8=1 ETX_VL_REQUESTS=2";;
  xcd_dsl3_touch) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1 ETX_VL_SLOTS=attn_mfma16_g8=1";;
  xcd_dsl3_coh) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1 ETX_VL_COH=1 ETX_VL_SLOTS=attn_mfma16_g8=1";;
  xcd_dsl3x2_coh) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1 ETX_VL_COH=1 ETX_VL_SLOTS=attn_mfma16_g8=1 ETX_VL_REQUESTS=2";;
  default) E="";;                                     # nothing passed: the derived defaults, i.e. what ships
  chk1) E="ETX_VL_CHECK=1 ETX_VL_XCD=0";;              # examples/sessions/nvref.py build_chk1: NR=1, so --check can run
  imported) E="ETX_VL_IMPORTED=1";;                   # round 24's all-imported comparison
  xcd_dsl3_all) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1 ETX_VL_COH=1 ETX_VL_ARGPF=1 ETX_VL_BAR2=1 ETX_VL_SLOTS=attn_mfma16_g8=1";;
  xcd_dsl3x2_all) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1 ETX_VL_COH=1 ETX_VL_ARGPF=1 ETX_VL_BAR2=1 ETX_VL_SLOTS=attn_mfma16_g8=1 ETX_VL_REQUESTS=2";;
  xcd_dsl3x2_touch) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1 ETX_VL_SLOTS=attn_mfma16_g8=1 ETX_VL_REQUESTS=2";;
  xcd_dsl2x2) E="ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_SLOTS=attn_mfma16_g8=1 ETX_VL_REQUESTS=2";;
  arm:*) A=${V#arm:}                                  # an arm of the registry (examples/vllm_llm/arms.py): what the session builds
    E="$(cd "$ROOT" && python3 -m examples.vllm_llm.arms env "$A")"
    ETX_CFLAGS="${ETX_CFLAGS:-} $(cd "$ROOT" && python3 -m examples.vllm_llm.arms cflags "$A")";;
  *) echo "variant $V"; exit 1;; esac
mkdir -p "$OUT"
# paths as the container sees them: $HOME/code -> /code
C_ROOT=/code/${ROOT#$HOME/code/}; C_OUT=/code/${OUT#$HOME/code/}
case "$OUT" in "$HOME/code/"*) ;; *) echo "outdir must be under ~/code (mounted into the container)"; exit 1;; esac
# Forward caller instructions as well as the named variant (the latter wins).
ENV_ARGS=()
for name in ${!ETX_VL_@} ${!ETX_DSL_@} ETX_DEV_RESET ETX_INBODY_EMIT ETX_COHERENCE; do   # ETX_COHERENCE: the build-wide coherence mode
  [ -z "${!name+x}" ] || ENV_ARGS+=(-e "$name=${!name}")
done
docker run --rm --pull=never --network=none ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} --platform linux/amd64 -v "$HOME/code:/code" -v "$HOME/code:$HOME/code" -e M="$M" -e E="$E" -e ROOT="$C_ROOT" -e OUT="$C_OUT" -e CFLAGS="${ETX_CFLAGS:-}" etx-rocm22 bash -c '
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
# Apply the exact same environment to selection, tile emission and compilation.
[ -z "$E" ] || export $E
if [ "${ETX_VL_BATCH:-0}" = 1 ]; then
  [ "${ETX_VL_REQUESTS:-1}" -gt 1 ] || { echo "ETX_VL_BATCH=1 needs ETX_VL_REQUESTS > 1" >&2; exit 1; }
  export ETX_VL_NB="${ETX_VL_REQUESTS}"
fi
if [ "${ETX_VL_UC:-0}" = 1 ]; then export ETX_VL_PIN=1; fi
SETTINGS="$(python3 -m examples.vllm_llm.plan_exports "$CFG" --shell)"
eval "$SETTINGS"
echo "== import ($EXPORTS)"
# Imports are merged by the capture/emitter tools. A check must validate only
# the current run closure, even when OUT contains captures from another build.
# Keep deliberate reuse in build.sh; discard only the old check manifest.
python3 -c "from pathlib import Path; Path(\"$OUT/imp/imports.json\").unlink(missing_ok=True)"
python3 -m etx.importer.vllm_kernels --vllm /code/vllm --out $OUT/imp --compiler hipcc --only "$EXPORTS" --slots "$SLOTS" "${BLK[@]}"
DSLT=""; if [ -n "$DSLOPS" ]; then python3 -m etx.ops.emit $OUT/imp/imports.json --tiles $OUT/dsl_tiles.hip --ops $DSLOPS; DSLT="ETX_VL_DSL_TILES=$OUT/dsl_tiles.hip"; fi
python3 -m etx.importer.adapter $OUT/imp/imports.json --tiles $OUT/imp_tiles.hip --header $OUT/imports.h
python3 -m etx.importer.bundle --imports $OUT/imp --exports "$EXPORTS" --devlibs $DL -o $OUT/imports.bc
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
