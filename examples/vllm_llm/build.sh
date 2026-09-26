#!/usr/bin/env bash
# Build the ETX megakernel of vLLM's kernels for one model packed by examples/vllm_llm/prep.py.
#   VLLM=~/vllm bash examples/vllm_llm/build.sh <packed-dir>
# Steps: import the kernels this model needs (device IR from vLLM's sources with ROCm's hipcc), adapters,
# one bitcode bundle with the device-library functions they call, the ETX plan, the final hipcc link.
set -euo pipefail
DIR="$(cd "${1:?packed model dir}" && pwd)"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
NAME="$(basename "$DIR")"; OUT="${ETX_OUT:-$ROOT/build/vl/$NAME}"; mkdir -p "$OUT"
VLLM="${VLLM:-$HOME/vllm}"
PY="$ROOT/.venv/bin/python"
cfg() { awk -v k="$1" '$1 == "cfg" && $2 == k {print $3}' "$DIR/vl.manifest"; }
H=$(cfg H); NH=$(cfg NH); NKV=$(cfg NKV); HD=$(cfg HD); QKN=$(cfg qk_norm); G=$((NH / NKV))
case $G in 4) ATTN=attn_mfma4_g4;; 6) ATTN=attn_mfma16_g6;; 8) ATTN=attn_mfma16_g8;; *) echo "no attention import for GQA $G"; exit 1;; esac
EXPORTS="wvsplitk_y2,$ATTN,attn_reduce,rms_norm_2d,fused_add_rms_norm,rope_neox,reshape_and_cache,silu_and_mul"
[ "$QKN" = 1 ] && EXPORTS="$EXPORTS,rms_norm_3d"
DEVLIBS="${ETX_DEVLIBS:-$(ls -d /opt/rocm/lib/llvm/lib/clang/*/lib/amdgcn/bitcode 2>/dev/null | head -1)}"
[ -d "$DEVLIBS" ] || DEVLIBS=/opt/rocm/amdgcn/bitcode
CONFIG="$DIR/hf/config.json"; [ -f "$CONFIG" ] || CONFIG="$DIR/config.json"

echo "== import ($EXPORTS)"
ORIG=(); CHK=(); [ "${ETX_VL_CHECK:-0}" = 1 ] && ORIG=(--orig) && CHK=(-DETX_VL_CHECK)
# ETX_VL_SLOTS="attn_mfma4_g4=1,attn_reduce=1": fewer of a kernel's blocks per workgroup (more CUs, idle waves)
"$PY" -m etx.importer.vllm_kernels --vllm "$VLLM" --out "$OUT/imp" --compiler hipcc --only "$EXPORTS" "${ORIG[@]}" --slots "${ETX_VL_SLOTS:-}" \
  --block rms_norm_2d=$((H / 8)) --block reshape_and_cache=$((NKV * HD / 8))
"$PY" -m etx.importer.adapter "$OUT/imp/imports.json" --tiles "$OUT/imp_tiles.hip" --header "$OUT/imports.h"
"$PY" -m etx.importer.bundle --imports "$OUT/imp" --exports "$EXPORTS" --devlibs "$DEVLIBS" -o "$OUT/imports.bc"
# ETX_VL_UC=1: cross-XCD tensors in uncached memory and DOMAIN fences for DEVICE events (ETX_COHERENT_DEVICE_DATA);
# the KV cache stays cached, so its writer and readers must share an XCD: ETX_VL_PIN is forced on
UCF=()
if [ "${ETX_VL_UC:-0}" = 1 ]; then export ETX_VL_PIN=1; UCF=(-DETX_VL_UC -DETX_COHERENT_DEVICE_DATA); fi
echo "== plan"
ETX_VL_IMPORTS="$OUT/imp/imports.json" ETX_VL_ADAPTERS="$OUT/imp_tiles.hip" ETX_VL_PLAN="$OUT/vl_plan.txt" ETX_LLM_CONFIG="$CONFIG" ETX_VL_CHAINS="$OUT/chain_tiles.hip" \
  "$PY" -m etx compile examples/vllm_llm/model.py --arch gfx942 --out "$OUT" --inline-tiles > "$OUT/compile.log" 2>&1 || { tail -20 "$OUT/compile.log"; exit 1; }
grep -E "tasks=|workers/domain" "$OUT/compile.log" || true
echo "== hipcc"
hipcc -O3 -std=c++17 --offload-arch=gfx942 -I "$ROOT/etx/runtime/include" -I "$OUT" -I "$ROOT" \
  -Xclang -mlink-builtin-bitcode -Xclang "$OUT/imports.bc" -Rpass-analysis=kernel-resource-usage \
  "${CHK[@]}" "${UCF[@]}" "$OUT/megakernel_d0.hip" "$ROOT/examples/vllm_llm/host.hip" $([ "${ETX_VL_CHECK:-0}" = 1 ] && echo "$OUT/imp/orig_table.hip -x none $OUT/imp/orig_*.o") -o "$OUT/run" 2>&1 \
  | grep -E "Function Name|VGPRs:|ScratchSize|VGPRs Spill|LDS Size|error" | sed -E "s/.*remark: +//; s/ \[-Rpass.*//" | paste - - - - - || true
[ -x "$OUT/run" ] || { echo "build failed: no $OUT/run"; exit 1; }
echo "built $OUT/run (plan $OUT/vl_plan.txt)"
