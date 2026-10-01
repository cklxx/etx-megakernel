#!/usr/bin/env bash
# Build the ETX megakernel of vLLM's kernels for one model packed by examples/vllm_llm/prep.py.
#   VLLM=~/vllm [ETX_HIPCC_FLAGS="-DETX_EV_STRIDE=1 ..."] bash examples/vllm_llm/build.sh <packed-dir>
# Steps: import the kernels this model needs (device IR from vLLM's sources with ROCm's hipcc), adapters,
# one bitcode bundle with the device-library functions they call, the ETX plan, the final hipcc link.
set -euo pipefail
DIR="$(cd "${1:?packed model dir}" && pwd)"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
NAME="$(basename "$DIR")"; OUT="${ETX_OUT:-$ROOT/build/vl/$NAME}"; mkdir -p "$OUT"
VLLM="${VLLM:-$HOME/vllm}"
PY="$ROOT/.venv/bin/python"
# model dimensions: from the packed manifest, or (before prep.py has run, so builds can start while weights pack) from
# the HF config through the same loader prep.py uses
if [ -f "$DIR/vl.manifest" ]; then
  cfg() { awk -v k="$1" '$1 == "cfg" && $2 == k {print $3}' "$DIR/vl.manifest"; }
else
  CFGJ="$DIR/hf/config.json"; [ -f "$CFGJ" ] || { echo "no $DIR/vl.manifest and no $CFGJ"; exit 1; }
  cfg() { (cd "$ROOT" && "$PY" -c "import sys; from examples.llm.model import load_dims; d = load_dims('$CFGJ')
v = {'H': d.H, 'NH': d.NH, 'NKV': d.NKV, 'HD': d.HD, 'qk_norm': int(d.qk_norm), 'moe': int(d.moe), 'MI': d.MI}['$1']; print(v)"); }
fi
H=$(cfg H); NH=$(cfg NH); NKV=$(cfg NKV); HD=$(cfg HD); QKN=$(cfg qk_norm); G=$((NH / NKV))
case $G in 4) ATTN=attn_mfma4_g4;; 6) ATTN=attn_mfma16_g6;; 8) ATTN=attn_mfma16_g8;; *) echo "no attention import for GQA $G"; exit 1;; esac
# Import only what the plan will actually instantiate: ETX_IMPORT_LDS is the MAX over every imported kernel's LDS, and
# the megakernel's workgroup carries that arena whether or not a task uses it. wvsplitk_y1/y2 declare 32768 B each --
# the largest of any tile -- and ETX_VL_DSL_GEMV=1 replaces them with etx.ops.gemv (21120 B), so importing them anyway
# cost 11648 B of LDS in every phase and capped occupancy at 1 workgroup per CU (measured 2026-10-01, round 16).
EXPORTS="$ATTN,attn_reduce,rms_norm_2d"
[ "${ETX_VL_DSL_GEMV:-0}" != 1 ] && EXPORTS="wvsplitk_y2,$EXPORTS"
[ "${ETX_VL_DSL_ATTN:-0}" != head ] && EXPORTS="$EXPORTS,fused_add_rms_norm,rope_neox,reshape_and_cache"
EXPORTS="$EXPORTS,silu_and_mul"
[ "$QKN" = 1 ] && [ "${ETX_VL_DSL_ATTN:-0}" != head ] && EXPORTS="$EXPORTS,rms_norm_3d"
# ETX_VL_QKNR=1: q/k norm + rope as vLLM's fused_qk_norm_rope (its enable_qk_norm_rope_fusion pass), one launch
[ "$QKN" = 1 ] && [ "${ETX_VL_QKNR:-0}" != 0 ] && EXPORTS="$EXPORTS,qknorm_rope"
# MoE (Qwen3-30B-A3B): router GEMV (YTILE 1), top-k, the two Triton fused_moe launches, moe_sum; silu over the 8 expert
# rows runs with block MI / 8, one row per task (as vLLM: 8 workgroups); one fused_moe program per workgroup (4 would
# need 80 KB of LDS and put 192 programs on 48 CUs)
MOEBLK=(); MOESLOTS=""
if [ "$(cfg moe)" = 1 ]; then
  [ "${ETX_VL_DSL_GEMV:-0}" != 1 ] && EXPORTS="$EXPORTS,wvsplitk_y1,fused_moe_up,fused_moe_down"
  [ "${ETX_VL_DSL_TOPK:-0}" != 1 ] && EXPORTS="$EXPORTS,topk_softmax_e128"
  [ "${ETX_VL_DSL_FOLD:-0}" != 1 ] && EXPORTS="$EXPORTS,moe_sum_k8"
  MOEBLK=(--block silu_and_mul=$(( $(cfg MI) / 8 ))); MOESLOTS="fused_moe_up=1,fused_moe_down=1,silu_and_mul=1"
fi
# The device bitcode moves between ROCm layouts (7.2: lib/llvm/lib/clang/<v>/lib/amdgcn/bitcode; 7.14:
# core-<v>/lib/llvm/amdgcn/bitcode), so search instead of assuming, and never let a failing `ls` kill the script
# (set -e + pipefail made the old one-liner exit 2 on any box without that exact path).
DEVLIBS="${ETX_DEVLIBS:-}"
if [ -z "$DEVLIBS" ]; then
  for d in /opt/rocm*/lib/llvm/lib/clang/*/lib/amdgcn/bitcode /opt/rocm*/lib/llvm/amdgcn/bitcode \
           /opt/rocm*/core-*/lib/llvm/amdgcn/bitcode /opt/rocm*/amdgcn/bitcode; do
    [ -f "$d/ocml.bc" ] && { DEVLIBS=$d; break; }
  done
fi
[ -f "$DEVLIBS/ocml.bc" ] || { echo "no ROCm device bitcode found (set ETX_DEVLIBS); looked under /opt/rocm*" >&2; exit 1; }
echo "devlibs: $DEVLIBS"
CONFIG="$DIR/hf/config.json"; [ -f "$CONFIG" ] || CONFIG="$DIR/config.json"

echo "== import ($EXPORTS)"
ORIG=(); CHK=(); [ "${ETX_VL_CHECK:-0}" = 1 ] && ORIG=(--orig) && CHK=(-DETX_VL_CHECK)
# ETX_VL_SLOTS="attn_mfma4_g4=1,attn_reduce=1": fewer of a kernel's blocks per workgroup (more CUs, idle waves)
SLOTS="${ETX_VL_SLOTS:-}"; [ -n "$MOESLOTS" ] && SLOTS="${SLOTS:+$SLOTS,}$MOESLOTS"
# ETX_VL_DSL_GEMV=1: the streaming GEMVs as ETX ops (etx.ops.gemv): their tiles and argument layouts join imports.json
# ETX_VL_DSL_TOPK=1: the router's softmax + top-k as etx.ops.topk
# ETX_VL_BATCH=1: the dense GEMV serves ETX_VL_REQUESTS activation columns per weight read; the tile emitter and the
# model both read ETX_VL_NB, so set it once here and export it to every step of the build.
if [ "${ETX_VL_BATCH:-0}" = 1 ]; then
  [ "${ETX_VL_REQUESTS:-1}" -gt 1 ] || { echo "ETX_VL_BATCH=1 needs ETX_VL_REQUESTS > 1" >&2; exit 1; }
  export ETX_VL_NB="${ETX_VL_REQUESTS}"
fi
DSLOPS=""; [ "${ETX_VL_DSL_GEMV:-0}" = 1 ] && DSLOPS="gemv"; [ "${ETX_VL_DSL_TOPK:-0}" = 1 ] && DSLOPS="${DSLOPS:+$DSLOPS,}topk"
# ETX_VL_DSL_ATTN=head: the attention block by kv head (etx.ops.attnprep; attention at one block per task)
if [ "${ETX_VL_DSL_ATTN:-0}" = head ]; then DSLOPS="${DSLOPS:+$DSLOPS,}attnprep"; SLOTS="${SLOTS:+$SLOTS,}$ATTN=1"; fi
"$PY" -m etx.importer.vllm_kernels --vllm "$VLLM" --out "$OUT/imp" --compiler hipcc --only "$EXPORTS" "${ORIG[@]}" --slots "$SLOTS" \
  --block rms_norm_2d=$((H / 8)) --block reshape_and_cache=$((NKV * HD / 8)) "${MOEBLK[@]}"
# the DSL ops' tiles and argument layouts join imports.json after the import, before the adapter
DSLT=(); if [ -n "$DSLOPS" ]; then "$PY" -m etx.ops.emit "$OUT/imp/imports.json" --tiles "$OUT/dsl_tiles.hip" --ops "$DSLOPS"; DSLT=(ETX_VL_DSL_TILES="$OUT/dsl_tiles.hip"); fi
"$PY" -m etx.importer.adapter "$OUT/imp/imports.json" --tiles "$OUT/imp_tiles.hip" --header "$OUT/imports.h"
"$PY" -m etx.importer.bundle --imports "$OUT/imp" --exports "$EXPORTS" --devlibs "$DEVLIBS" -o "$OUT/imports.bc"
# ETX_VL_UC=1: cross-XCD tensors in uncached memory and DOMAIN fences for DEVICE events (ETX_COHERENT_DEVICE_DATA);
# the KV cache stays cached, so its writer and readers must share an XCD: ETX_VL_PIN is forced on
UCF=()
if [ "${ETX_VL_UC:-0}" = 1 ]; then export ETX_VL_PIN=1; UCF=(-DETX_VL_UC -DETX_COHERENT_DEVICE_DATA); fi
# ETX_VL_COH=1: the DSL tiles' activation accesses are device-coherent (etx.ops.coh) and the plan drops their acquire
# invalidates (model.py coh_of); untested together with ETX_VL_UC's downgraded DEVICE fences, so the pair is refused
# ETX_VL_NT: weight loads in the DSL GEMV marked non-temporal (etx/ops/gemv.py dsl_wload). ON by default since
# 2026-10-01 round 16: weights are read once per token and have no reuse, so caching them evicts the MALL-resident
# KV cache and the activations. Measured -0.106 ms/token with L2 hit rate 31.9% -> 34.6%; ETX_VL_NT=0 restores it.
NTF=(); [ "${ETX_VL_NT:-1}" = 1 ] && NTF=(-DETX_DSL_NT)
COHF=()
# ETX_DEV_RESET=1: the kernel rewrites its own per-step state at the end of the step (a cooperative grid sync, then
# the grid copies the host-built init images), so the host enqueues only the ctrl memset instead of ~17 calls
[ "${ETX_DEV_RESET:-0}" = 1 ] && COHF+=(-DETX_DEV_RESET)
if [ "${ETX_VL_COH:-0}" = 1 ]; then
  [ "${ETX_VL_UC:-0}" = 1 ] && { echo "ETX_VL_COH=1 with ETX_VL_UC=1 is untested; pick one" >&2; exit 1; }
  COHF=(-DETX_DSL_COH)
fi
echo "== plan"
env "${DSLT[@]}" ETX_VL_IMPORTS="$OUT/imp/imports.json" ETX_VL_ADAPTERS="$OUT/imp_tiles.hip" ETX_VL_PLAN="$OUT/vl_plan.txt" ETX_LLM_CONFIG="$CONFIG" ETX_VL_CHAINS="$OUT/chain_tiles.hip" \
  "$PY" -m etx compile examples/vllm_llm/model.py --arch gfx942 --out "$OUT" --inline-tiles > "$OUT/compile.log" 2>&1 || { tail -20 "$OUT/compile.log"; exit 1; }
grep -E "tasks=|workers/domain" "$OUT/compile.log" || true
if [ "${ETX_VL_UC:-0}" = 1 ]; then "$PY" "$ROOT/examples/vllm_llm/check_uc_plan.py" "$OUT/plan.json" || exit 1; fi
echo "== hipcc"
# ETX_ROCM_HEADERS: a HIP include tree for ROCm installs that ship no headers (see etx/importer/vllm_kernels.py)
ROCMINC=(); [ -n "${ETX_ROCM_HEADERS:-}" ] && ROCMINC=(-isystem "$ETX_ROCM_HEADERS")
NBF=(-DETX_VL_NB="${ETX_VL_NB:-1}")      # the host refuses a plan whose nb exceeds what the tile was emitted for
# ETX_VL_ABLATE=<name>: a timing-only build with a piece of the DSL MoE tile removed (etx/ops/gemv.py's DSL_ABLATE_*).
# Every ablation but EPL4 produces WRONG output on purpose and must never ship; the correctness gates will fail.
ABL=(); [ -n "${ETX_VL_ABLATE:-}" ] && ABL=(-DDSL_ABLATE_$(echo "$ETX_VL_ABLATE" | tr '[:lower:]' '[:upper:]'))
# ETX_VL_FOLD_TOPK=0: keep the norm folds, restore the single top-k launch per layer (model.py FOLD_TOPK)
[ "${ETX_VL_FOLD_TOPK:-1}" = 0 ] && ABL+=(-DETX_VL_NO_TOPK_FOLD)
hipcc -O3 -std=c++17 --offload-arch=gfx942 "${ROCMINC[@]}" "${NBF[@]}" -I "$ROOT/etx/runtime/include" -I "$OUT" -I "$ROOT" \
  -Xclang -mlink-builtin-bitcode -Xclang "$OUT/imports.bc" -Rpass-analysis=kernel-resource-usage \
  "${CHK[@]}" "${UCF[@]}" "${COHF[@]}" "${NTF[@]}" "${ABL[@]}" -DETX_VL_REQUESTS=${ETX_VL_REQUESTS:-1} ${ETX_HIPCC_FLAGS:-} "$OUT/megakernel_d0.hip" "$ROOT/examples/vllm_llm/host.hip" $([ "${ETX_VL_CHECK:-0}" = 1 ] && echo "$OUT/imp/orig_table.hip -x none $OUT/imp/orig_*.o") -o "$OUT/run" 2>&1 \
  | grep -E "Function Name|VGPRs:|ScratchSize|VGPRs Spill|LDS Size|error" | sed -E "s/.*remark: +//; s/ \[-Rpass.*//" | paste - - - - - || true
[ -x "$OUT/run" ] || { echo "build failed: no $OUT/run"; exit 1; }
echo "built $OUT/run (plan $OUT/vl_plan.txt)"
