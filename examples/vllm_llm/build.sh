#!/usr/bin/env bash
# Build the ETX megakernel of vLLM's kernels for one model packed by examples/vllm_llm/prep.py.
#   VLLM=~/vllm [ETX_HIPCC_FLAGS="-DETX_EV_STRIDE=1 ..."] bash examples/vllm_llm/build.sh <packed-dir>
# Steps: import the kernels this model needs (device IR from vLLM's sources with ROCm's hipcc), adapters,
# one bitcode bundle with the device-library functions they call, the ETX plan, the final hipcc link.
# Bash 3.2 treats empty arrays as unset under nounset. Guard optional array
# expansions with ${array[@]+"${array[@]}"}; empty arrays still pass zero args.
set -euo pipefail
DIR="$(cd "${1:?packed model dir}" && pwd)"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
NAME="$(basename "$DIR")"; OUT="${ETX_OUT:-$ROOT/build/vl/$NAME}"; mkdir -p "$OUT"
VLLM="${VLLM:-$HOME/vllm}"
PY="$ROOT/.venv/bin/python"
CONFIG="$DIR/hf/config.json"; [ -f "$CONFIG" ] || CONFIG="$DIR/config.json"
# Independent diagnostic builds only. Ordinary source/ABI remain unchanged.
IBF=()
if [ "${ETX_INBODY_BUILD:-0}" = 1 ]; then
  export ETX_INBODY_EMIT=1
  IBF=(-DETX_INBODY=1)
else
  unset ETX_INBODY_EMIT
fi
[ "${ETX_BF16_CAPTURE_BUILD:-0}" = 1 ] && IBF+=(-DETX_BF16_CAPTURE)
# Set plan-affecting environment before deriving imports or emitting tiles.
if [ "${ETX_VL_BATCH:-0}" = 1 ]; then
  [ "${ETX_VL_REQUESTS:-1}" -gt 1 ] || { echo "ETX_VL_BATCH=1 needs ETX_VL_REQUESTS > 1" >&2; exit 1; }
  export ETX_VL_NB="${ETX_VL_REQUESTS}"
fi
if [ "${ETX_VL_UC:-0}" = 1 ]; then export ETX_VL_PIN=1; fi
# One source for imports, launch settings and DSL emitters, also used by the
# ROCm device check. Capture first so a failed helper cannot be hidden by eval.
SETTINGS="$("$PY" -m examples.vllm_llm.plan_exports "$CONFIG" --shell)"
eval "$SETTINGS"
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
echo "== import ($EXPORTS)"
ORIG=(); CHK=(); [ "${ETX_VL_CHECK:-0}" = 1 ] && ORIG=(--orig) && CHK=(-DETX_VL_CHECK)
"$PY" -m etx.importer.vllm_kernels --vllm "$VLLM" --out "$OUT/imp" --compiler hipcc --only "$EXPORTS" ${ORIG[@]+"${ORIG[@]}"} --slots "$SLOTS" \
  ${BLK[@]+"${BLK[@]}"}
# the DSL ops' tiles and argument layouts join imports.json after the import, before the adapter
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
env ${DSLT[@]+"${DSLT[@]}"} ETX_VL_IMPORTS="$OUT/imp/imports.json" ETX_VL_ADAPTERS="$OUT/imp_tiles.hip" ETX_VL_PLAN="$OUT/vl_plan.txt" ETX_LLM_CONFIG="$CONFIG" ETX_VL_CHAINS="$OUT/chain_tiles.hip" \
  "$PY" -m etx compile examples/vllm_llm/model.py --arch gfx942 --out "$OUT" --inline-tiles > "$OUT/compile.log" 2>&1 || { tail -20 "$OUT/compile.log"; exit 1; }
grep -E "tasks=|workers/domain" "$OUT/compile.log" || true
if [ "${ETX_VL_UC:-0}" = 1 ]; then "$PY" "$ROOT/examples/vllm_llm/check_uc_plan.py" "$OUT/plan.json" || exit 1; fi
echo "== hipcc"
hipcc --version 2>&1 | tee "$OUT/compiler_version.txt"
# ETX_ROCM_HEADERS: a HIP include tree for ROCm installs that ship no headers (see etx/importer/vllm_kernels.py)
ROCMINC=(); [ -n "${ETX_ROCM_HEADERS:-}" ] && ROCMINC=(-isystem "$ETX_ROCM_HEADERS")
# ETX_VL_NB and ETX_VL_REQUESTS are published by the manifest; the host reads the plan's values, not a -D it could
# disagree with (round 14: the tile emitted without NB while the plan batched, extra requests reading column 0).
# ETX_VL_ABLATE=<name>: a timing-only build with a piece of the DSL MoE tile removed (etx/ops/gemv.py's DSL_ABLATE_*).
# Every ablation but EPL4 produces WRONG output on purpose and must never ship; the correctness gates will fail.
ABL=(); [ -n "${ETX_VL_ABLATE:-}" ] && ABL=(-DDSL_ABLATE_$(echo "$ETX_VL_ABLATE" | tr '[:lower:]' '[:upper:]'))
# ETX_TOPK_KEY32=1: the softmax top-k selects on one 32-bit key per expert with a DPP wave max (etx/tile_util.h,
# docs/TOPK-KEY32-2026-10-08.md). Device code only; the plan does not change. Default off.
[ "${ETX_TOPK_KEY32:-0}" = 1 ] && ABL+=(-DETX_TOPK_KEY32)
# ETX_VL_FOLD_TOPK forces model.py's derived fold decision; it reaches the tile through plan_data.h's
# ETX_EMIT_FOLD_TOPK, not a -D here -- the -D defaulted to "fold" and silently contradicted the plan (round 30).
hipcc -O3 -std=c++17 --offload-arch=gfx942 ${ROCMINC[@]+"${ROCMINC[@]}"} -I "$ROOT/etx/runtime/include" -I "$OUT" -I "$ROOT" \
  -Xclang -mlink-builtin-bitcode -Xclang "$OUT/imports.bc" -Rpass-analysis=kernel-resource-usage \
  ${CHK[@]+"${CHK[@]}"} ${NTF[@]+"${NTF[@]}"} ${VNF[@]+"${VNF[@]}"} ${ABL[@]+"${ABL[@]}"} ${PREPF[@]+"${PREPF[@]}"} ${IBF[@]+"${IBF[@]}"} ${ETX_HIPCC_FLAGS:-} "$OUT/megakernel_d0.hip" "$ROOT/examples/vllm_llm/host.hip" $([ "${ETX_VL_CHECK:-0}" = 1 ] && echo "$OUT/imp/orig_table.hip -x none $OUT/imp/orig_*.o") -o "$OUT/run" 2>&1 \
  | tee "$OUT/hipcc.log"
[ -x "$OUT/run" ] || { echo "build failed: no $OUT/run"; exit 1; }
# CHECK builds require packed inputs and a golden before certification. Ordinary
# pre-prep builds remain supported; they cannot supply a certificate until rebuilt.
if [ "${ETX_VL_CHECK:-0}" = 1 ] || { [ -f "$DIR/vl.manifest" ] && [ -f "$DIR/golden_c1024.txt" ]; }; then
  "$PY" -m examples.vllm_llm.import_certificate --write-build "$OUT" --model "$DIR"
fi
echo "built $OUT/run (plan $OUT/vl_plan.txt)"
