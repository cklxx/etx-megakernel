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
EXPORTS="wvsplitk_y2,$ATTN,attn_reduce,rms_norm_2d,fused_add_rms_norm,rope_neox,reshape_and_cache,silu_and_mul"
[ "$QKN" = 1 ] && EXPORTS="$EXPORTS,rms_norm_3d"
# ETX_VL_QKNR=1: q/k norm + rope as vLLM's fused_qk_norm_rope (its enable_qk_norm_rope_fusion pass), one launch
[ "$QKN" = 1 ] && [ "${ETX_VL_QKNR:-0}" != 0 ] && EXPORTS="$EXPORTS,qknorm_rope"
# MoE (Qwen3-30B-A3B): router GEMV (YTILE 1), top-k, the two Triton fused_moe launches, moe_sum; silu over the 8 expert
# rows runs with block MI / 8, one row per task (as vLLM: 8 workgroups); one fused_moe program per workgroup (4 would
# need 80 KB of LDS and put 192 programs on 48 CUs)
MOEBLK=(); MOESLOTS=""
if [ "$(cfg moe)" = 1 ]; then
  EXPORTS="$EXPORTS,wvsplitk_y1,topk_softmax_e128,moe_sum_k8,fused_moe_up,fused_moe_down"
  MOEBLK=(--block silu_and_mul=$(( $(cfg MI) / 8 ))); MOESLOTS="fused_moe_up=1,fused_moe_down=1,silu_and_mul=1"
fi
DEVLIBS="${ETX_DEVLIBS:-$(ls -d /opt/rocm/lib/llvm/lib/clang/*/lib/amdgcn/bitcode 2>/dev/null | head -1)}"
[ -d "$DEVLIBS" ] || DEVLIBS=/opt/rocm/amdgcn/bitcode
CONFIG="$DIR/hf/config.json"; [ -f "$CONFIG" ] || CONFIG="$DIR/config.json"

echo "== import ($EXPORTS)"
ORIG=(); CHK=(); [ "${ETX_VL_CHECK:-0}" = 1 ] && ORIG=(--orig) && CHK=(-DETX_VL_CHECK)
# ETX_VL_SLOTS="attn_mfma4_g4=1,attn_reduce=1": fewer of a kernel's blocks per workgroup (more CUs, idle waves)
SLOTS="${ETX_VL_SLOTS:-}"; [ -n "$MOESLOTS" ] && SLOTS="${SLOTS:+$SLOTS,}$MOESLOTS"
# ETX_VL_DSL_GEMV=1: the streaming GEMVs as ETX ops (etx.ops.gemv): their tiles and argument layouts join imports.json
# ETX_VL_DSL_TOPK=1: the router's softmax + top-k as etx.ops.topk
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
echo "== plan"
env "${DSLT[@]}" ETX_VL_IMPORTS="$OUT/imp/imports.json" ETX_VL_ADAPTERS="$OUT/imp_tiles.hip" ETX_VL_PLAN="$OUT/vl_plan.txt" ETX_LLM_CONFIG="$CONFIG" ETX_VL_CHAINS="$OUT/chain_tiles.hip" \
  "$PY" -m etx compile examples/vllm_llm/model.py --arch gfx942 --out "$OUT" --inline-tiles > "$OUT/compile.log" 2>&1 || { tail -20 "$OUT/compile.log"; exit 1; }
grep -E "tasks=|workers/domain" "$OUT/compile.log" || true
if [ "${ETX_VL_UC:-0}" = 1 ]; then "$PY" "$ROOT/examples/vllm_llm/check_uc_plan.py" "$OUT/plan.json" || exit 1; fi
echo "== hipcc"
hipcc -O3 -std=c++17 --offload-arch=gfx942 -I "$ROOT/etx/runtime/include" -I "$OUT" -I "$ROOT" \
  -Xclang -mlink-builtin-bitcode -Xclang "$OUT/imports.bc" -Rpass-analysis=kernel-resource-usage \
  "${CHK[@]}" "${UCF[@]}" -DETX_VL_REQUESTS=${ETX_VL_REQUESTS:-1} ${ETX_HIPCC_FLAGS:-} "$OUT/megakernel_d0.hip" "$ROOT/examples/vllm_llm/host.hip" $([ "${ETX_VL_CHECK:-0}" = 1 ] && echo "$OUT/imp/orig_table.hip -x none $OUT/imp/orig_*.o") -o "$OUT/run" 2>&1 \
  | grep -E "Function Name|VGPRs:|ScratchSize|VGPRs Spill|LDS Size|error" | sed -E "s/.*remark: +//; s/ \[-Rpass.*//" | paste - - - - - || true
[ -x "$OUT/run" ] || { echo "build failed: no $OUT/run"; exit 1; }
echo "built $OUT/run (plan $OUT/vl_plan.txt)"
