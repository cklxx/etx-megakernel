#!/usr/bin/env bash
# Real capture-site arithmetic and native host-object checks, no GPU/runtime.
set -euo pipefail
ROOT=${1:?root}; OUT=${2:?out}; DEPS=${HOME}/.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/link-deps/root
REF=${HOME}/code/etx/results_2026-10-01/box_final/b14_b2
mkdir -p "$OUT/compiler-tmp"
docker run --rm --pull=never --network=none --platform linux/amd64 \
 -v "$HOME/code:/code" -v "$HOME/code:$HOME/code:ro" -v "$OUT:$OUT" -v "$DEPS:/link-deps:ro" \
 -e LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu -e ROOT="$ROOT" -e OUT="$OUT" -e REF="$REF" -e TMPDIR="$OUT/compiler-tmp" etx-rocm22 bash -c '
set -euo pipefail
RL=/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
COMMON=(-x hip --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path="$DL" -isystem /code/rocm-headers/inc -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -I"$ROOT/etx/runtime/include" -I"$ROOT" -I"$OUT/g13-ready-final")
for REV in b5eaa67 ff9ef92; do
 P="$OUT/historical-capture-ready-$REV"
 "$RL/bin/clang++" "${COMMON[@]}" -I"$P" --cuda-device-only -Rpass-analysis=kernel-resource-usage -S "$P/capture-probe.hip" -o "$P/capture.s" > "$P/device.stdout.log" 2> "$P/resource.log"
 "$RL/bin/llvm-mc" -triple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj "$P/capture.s" -o "$P/capture.o" > "$P/assemble.log" 2>&1
 "$RL/bin/ld.lld" -shared "$P/capture.o" -o "$P/capture.hsaco" > "$P/link.log" 2>&1
 "$RL/bin/clang++" "${COMMON[@]}" -I"$REF" --cuda-host-only -c "$P/host.hip" -o "$P/host.o" > "$P/host.log" 2>&1
 sha256sum "$P/dsl_tiles.hip" "$P/host.hip" "$P/capture.hsaco" "$P/host.o" "$REF/plan_data.h" "$REF/imports.h" > "$P/SHA256SUMS"
done
' > "$OUT/capture-docker.log" 2>&1
