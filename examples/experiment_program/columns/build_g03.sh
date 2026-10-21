#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?root}; OUT=${2:?out}; DEPS=${HOME}/.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/link-deps/root
mkdir -p "$OUT/compiler-tmp"
docker run --rm --pull=never --network=none --platform linux/amd64 \
 -v "$HOME/code:/code" -v "$HOME/code:$HOME/code:ro" -v "$OUT:$OUT" -v "$DEPS:/link-deps:ro" \
 -e LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu -e ROOT="$ROOT" -e OUT="$OUT" -e TMPDIR="$OUT/compiler-tmp" etx-rocm22 bash -c '
set -euo pipefail
RL=/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
cp "$OUT/historical_b5eaa67.h" "$OUT/g03-ready-final/historical_stream.h"
COMMON=(-x hip --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path="$DL" -isystem /code/rocm-headers/inc -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -I"$ROOT/etx/runtime/include" -I"$ROOT" -I"$OUT/g03-ready-final")
"$RL/bin/clang++" "${COMMON[@]}" --cuda-device-only -Rpass-analysis=kernel-resource-usage -S "$OUT/g03-ready-final/megakernel_d0.hip" -o "$OUT/g03-ready-final/g03.s" > "$OUT/g03-ready-final/device.stdout.log" 2> "$OUT/g03-ready-final/resource.log"
"$RL/bin/llvm-mc" -triple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj "$OUT/g03-ready-final/g03.s" -o "$OUT/g03-ready-final/g03.o" > "$OUT/g03-ready-final/assemble.log" 2>&1
"$RL/bin/ld.lld" -shared "$OUT/g03-ready-final/g03.o" -o "$OUT/g03-ready-final/g03.hsaco" > "$OUT/g03-ready-final/link.log" 2>&1
ln -s g03.s "$OUT/g03-ready-final/mk.s"
ln -s g03.hsaco "$OUT/g03-ready-final/model.hsaco"
"$RL/bin/clang++" -x c++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ -DETX_G03_DRIVER -I/code/rocm-headers/inc -I"$ROOT" -I"$ROOT/etx/runtime/include" -I"$OUT/g03-ready-final" "$ROOT/examples/experiment_program/columns/driver.cpp" -fuse-ld=lld -ldl -o "$OUT/g03-ready-run" > "$OUT/g03-ready-final/host.log" 2>&1
sha256sum "$OUT/g03-ready-final/g03.hsaco" "$OUT/g03-ready-final/g03.s" "$OUT/g03-ready-run" "$OUT/g03-ready-final/plan.json" "$OUT/g03-ready-final/plan_data.h" "$ROOT/examples/experiment_program/columns/g03_tiles.hip" "$ROOT/examples/experiment_program/columns/driver.cpp" "$ROOT/examples/experiment_program/columns/g13_runtime.h" > "$OUT/g03-ready-final/SHA256SUMS"
' > "$OUT/g03-ready-final-docker.log" 2>&1
