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
# ETX_VL_IMPORTED=1 restores the all-imported build (round 24's comparison: 6.435 ms against 4.937, and fusion still
# loses there). Otherwise the DSL tiles are the default, and the attention-prep tile applies only to a model with a
# q/k norm -- derived here exactly as examples/vllm_llm/model.py's _attn_head() derives it, so the two cannot disagree.
IMPORTED="${ETX_VL_IMPORTED:-0}"
D_GEMV="${ETX_VL_DSL_GEMV:-$([ "$IMPORTED" = 1 ] && echo 0 || echo 1)}"
D_TOPK="${ETX_VL_DSL_TOPK:-$([ "$IMPORTED" = 1 ] && echo 0 || echo 1)}"
D_FOLD="${ETX_VL_DSL_FOLD:-$([ "$IMPORTED" = 1 ] && echo 0 || echo 1)}"
D_ATTN="${ETX_VL_DSL_ATTN:-$([ "$IMPORTED" != 1 ] && [ "$QKN" = 1 ] && echo head || echo 0)}"
export ETX_VL_DSL_GEMV="$D_GEMV" ETX_VL_DSL_TOPK="$D_TOPK" ETX_VL_DSL_FOLD="$D_FOLD" ETX_VL_DSL_ATTN="$D_ATTN"
EXPORTS="$ATTN,attn_reduce,rms_norm_2d"
[ "$D_GEMV" != 1 ] && EXPORTS="wvsplitk_y2,$EXPORTS"
[ "$D_ATTN" != head ] && EXPORTS="$EXPORTS,fused_add_rms_norm,rope_neox,reshape_and_cache"
EXPORTS="$EXPORTS,silu_and_mul"
[ "$QKN" = 1 ] && [ "$D_ATTN" != head ] && EXPORTS="$EXPORTS,rms_norm_3d"
# ETX_VL_QKNR=1: q/k norm + rope as vLLM's fused_qk_norm_rope (its enable_qk_norm_rope_fusion pass), one launch
[ "$QKN" = 1 ] && [ "${ETX_VL_QKNR:-0}" != 0 ] && EXPORTS="$EXPORTS,qknorm_rope"
# MoE (Qwen3-30B-A3B): router GEMV (YTILE 1), top-k, the two Triton fused_moe launches, moe_sum; silu over the 8 expert
# rows runs with block MI / 8, one row per task (as vLLM: 8 workgroups); one fused_moe program per workgroup (4 would
# need 80 KB of LDS and put 192 programs on 48 CUs)
MOEBLK=(); MOESLOTS=""
if [ "$(cfg moe)" = 1 ]; then
  MOESFX=$([ "$(cfg E)" = 128 ] && echo "" || echo "_e$(cfg E)")   # shape-keyed fused_moe (round 39)
  [ "$D_GEMV" != 1 ] && EXPORTS="$EXPORTS,wvsplitk_y1,fused_moe_up$MOESFX,fused_moe_down$MOESFX"
  [ "$D_TOPK" != 1 ] && EXPORTS="$EXPORTS,topk_softmax_e$(cfg E)"
  [ "$D_FOLD" != 1 ] && EXPORTS="$EXPORTS,moe_sum_k$(cfg TOPK)"
  MOEBLK=(--block silu_and_mul=$(( $(cfg MI) / 8 > 1024 ? 1024 : $(cfg MI) / 8 ))); MOESLOTS="fused_moe_up$MOESFX=1,fused_moe_down$MOESFX=1,silu_and_mul=1"
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
DSLOPS=""; [ "$D_GEMV" = 1 ] && DSLOPS="gemv"; [ "$D_TOPK" = 1 ] && DSLOPS="${DSLOPS:+$DSLOPS,}topk"
# ETX_VL_DSL_ATTN=head: the attention block by kv head (etx.ops.attnprep; attention at one block per task)
# attnprep needs the attention at one block per task and the reduce at one block per kv head: slots = G. That G=8
# happened to equal attn_reduce's own default for Qwen3-30B-A3B, so it was never set; a G=4 model refused to plan.
if [ "$D_ATTN" = head ]; then DSLOPS="${DSLOPS:+$DSLOPS,}attnprep"; SLOTS="${SLOTS:+$SLOTS,}$ATTN=1,attn_reduce=$G"; fi
"$PY" -m etx.importer.vllm_kernels --vllm "$VLLM" --out "$OUT/imp" --compiler hipcc --only "$EXPORTS" "${ORIG[@]}" --slots "$SLOTS" \
  --block rms_norm_2d=$((H / 8)) --block reshape_and_cache=$((NKV * HD / 8)) "${MOEBLK[@]}"
# the DSL ops' tiles and argument layouts join imports.json after the import, before the adapter
# PhiMoE-class models (the manifest's ln_norm / bias_o / bias_lm, written by prep.py from the config): the DSL gemv
# layout grows two pointers (bias, LayerNorm bias). Exported for BOTH the tile emission and the plan, which asserts
# the layout and its own config derivation agree (model.py) -- the round-14/19/30 drift class dies loudly.
DSLEXT=$([ "$(cfg ln_norm)" = 1 ] || [ "$(cfg bias_o)" = 1 ] || [ "$(cfg bias_lm)" = 1 ] && echo 1 || echo 0)
export ETX_VL_DSL_EXT=$DSLEXT
DSLT=(); if [ -n "$DSLOPS" ]; then "$PY" -m etx.ops.emit "$OUT/imp/imports.json" --tiles "$OUT/dsl_tiles.hip" --ops "$DSLOPS"; DSLT=(ETX_VL_DSL_TILES="$OUT/dsl_tiles.hip"); fi
"$PY" -m etx.importer.adapter "$OUT/imp/imports.json" --tiles "$OUT/imp_tiles.hip" --header "$OUT/imports.h"
"$PY" -m etx.importer.bundle --imports "$OUT/imp" --exports "$EXPORTS" --devlibs "$DEVLIBS" -o "$OUT/imports.bc"
# ETX_VL_UC=1: cross-XCD tensors in uncached memory and DOMAIN fences for DEVICE events (ETX_COHERENT_DEVICE_DATA);
# the KV cache stays cached, so its writer and readers must share an XCD: ETX_VL_PIN is forced on
# ETX_VL_UC is published by the manifest (ETX_VL_UC, and the lowering reads the plan); only ETX_VL_PIN, which the
# plan itself reads, is exported here.
if [ "${ETX_VL_UC:-0}" = 1 ]; then export ETX_VL_PIN=1; fi
# ETX_VL_COH=1: the DSL tiles' activation accesses are device-coherent (etx.ops.coh) and the plan drops their acquire
# invalidates (model.py coh_of); untested together with ETX_VL_UC's downgraded DEVICE fences, so the pair is refused
# ETX_VL_NT: weight loads in the DSL GEMV marked non-temporal (etx/ops/gemv.py dsl_wload). ON by default since
# 2026-10-01 round 16: weights are read once per token and have no reuse, so caching them evicts the MALL-resident
# KV cache and the activations. Measured -0.106 ms/token with L2 hit rate 31.9% -> 34.6%; ETX_VL_NT=0 restores it.
NTF=(); [ "${ETX_VL_NT:-1}" = 1 ] && NTF=(-DETX_DSL_NT)
# ETX_VL_VECNORM: the folded norm stages 8 elements per thread with 16 B coherent accesses instead of one 2 B request
# per element (etx/ops/gemv.py dsl_stage_norm_at). Own array, never reassigned -- a flag appended to an array that a
# later `if` overwrites is how ETX_DEV_RESET silently never compiled (round 19/20).
VNF=(); [ "${ETX_VL_VECNORM:-1}" = 1 ] && VNF=(-DETX_DSL_VECNORM)   # ON by default: -0.105 ms, round 23
# ETX_DEV_RESET=1 is read by model.py at PLAN time and emitted into plan_data.h as ETX_EMIT_DEV_RESET, so it needs
# no define here. It had one until round 19, and this `if` overwrote the array that carried it whenever ETX_VL_COH
# was also set -- the host then believed in a reset tail that was never compiled: 0/32 tokens, no diagnostic.
# ETX_VL_PREP_EARLY=1: dsl_attnprep issues positions[0] / slot_mapping[0] with the row load instead of after the
# wave reduction, wave-uniform (one request per wave, not one per lane). Round 22.
PREPF=(); [ "${ETX_VL_PREP_EARLY:-0}" = 1 ] && PREPF=(-DETX_DSL_PREP_EARLY -DETX_DSL_UNIFORM)
# ETX_VL_COH reaches the tiles through the feature manifest (etx/codegen/manifest.py): model.py decides it at plan
# time and both generated artefacts define ETX_DSL_COH from that decision. It had a -D here until round 31, derived
# by an expression that had to be hand-matched against model.py's -- the structure that produced rounds 19/20/30.
if [ "${ETX_VL_COH:-1}" = 1 ] && [ "${ETX_VL_UC:-0}" = 1 ]; then
  echo "ETX_VL_COH=1 with ETX_VL_UC=1 is untested; pick one" >&2; exit 1
fi
echo "== plan"
env "${DSLT[@]}" ETX_VL_IMPORTS="$OUT/imp/imports.json" ETX_VL_ADAPTERS="$OUT/imp_tiles.hip" ETX_VL_PLAN="$OUT/vl_plan.txt" ETX_LLM_CONFIG="$CONFIG" ETX_VL_CHAINS="$OUT/chain_tiles.hip" \
  "$PY" -m etx compile examples/vllm_llm/model.py --arch gfx942 --out "$OUT" --inline-tiles > "$OUT/compile.log" 2>&1 || { tail -20 "$OUT/compile.log"; exit 1; }
grep -E "tasks=|workers/domain" "$OUT/compile.log" || true
if [ "${ETX_VL_UC:-0}" = 1 ]; then "$PY" "$ROOT/examples/vllm_llm/check_uc_plan.py" "$OUT/plan.json" || exit 1; fi
echo "== hipcc"
# ETX_ROCM_HEADERS: a HIP include tree for ROCm installs that ship no headers (see etx/importer/vllm_kernels.py)
ROCMINC=(); [ -n "${ETX_ROCM_HEADERS:-}" ] && ROCMINC=(-isystem "$ETX_ROCM_HEADERS")
# ETX_VL_NB and ETX_VL_REQUESTS are published by the manifest; the host reads the plan's values, not a -D it could
# disagree with (round 14: the tile emitted without NB while the plan batched, extra requests reading column 0).
# ETX_VL_ABLATE=<name>: a timing-only build with a piece of the DSL MoE tile removed (etx/ops/gemv.py's DSL_ABLATE_*).
# Every ablation but EPL4 produces WRONG output on purpose and must never ship; the correctness gates will fail.
ABL=(); [ -n "${ETX_VL_ABLATE:-}" ] && ABL=(-DDSL_ABLATE_$(echo "$ETX_VL_ABLATE" | tr '[:lower:]' '[:upper:]'))
# ETX_VL_FOLD_TOPK forces model.py's derived fold decision; it reaches the tile through plan_data.h's
# ETX_EMIT_FOLD_TOPK, not a -D here -- the -D defaulted to "fold" and silently contradicted the plan (round 30).
hipcc -O3 -std=c++17 --offload-arch=gfx942 "${ROCMINC[@]}" -I "$ROOT/etx/runtime/include" -I "$OUT" -I "$ROOT" \
  -Xclang -mlink-builtin-bitcode -Xclang "$OUT/imports.bc" -Rpass-analysis=kernel-resource-usage \
  "${CHK[@]}" "${NTF[@]}" "${VNF[@]}" "${ABL[@]}" "${PREPF[@]}" ${ETX_HIPCC_FLAGS:-} "$OUT/megakernel_d0.hip" "$ROOT/examples/vllm_llm/host.hip" $([ "${ETX_VL_CHECK:-0}" = 1 ] && echo "$OUT/imp/orig_table.hip -x none $OUT/imp/orig_*.o") -o "$OUT/run" 2>&1 \
  | grep -E "Function Name|VGPRs:|ScratchSize|VGPRs Spill|LDS Size|error" | sed -E "s/.*remark: +//; s/ \[-Rpass.*//" | paste - - - - - || true
[ -x "$OUT/run" ] || { echo "build failed: no $OUT/run"; exit 1; }
echo "built $OUT/run (plan $OUT/vl_plan.txt)"
