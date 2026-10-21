#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?root}; OUT=${2:?out}; FLEET_SRC=${ETX_COLUMN_FLEET_SRC:?accepted cooked Fleet source required}
DEPS=${HOME}/.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/link-deps/root
mkdir -p "$OUT/compiler-tmp"
docker run --rm --pull=never --network=none --platform linux/amd64 \
 -v "$HOME/code:/code" -v "$HOME/code:$HOME/code:ro" -v "$OUT:$OUT" -v "$FLEET_SRC:/fleet/src:ro" -v "$DEPS:/link-deps:ro" \
 -e LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu -e ROOT="$ROOT" -e OUT="$OUT" -e TMPDIR="$OUT/compiler-tmp" etx-rocm22 bash -c '
set -euo pipefail
RL=/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
COMMON=(-x hip --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path="$DL" -isystem /code/rocm-headers/inc -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -DFLEET_NT_WEIGHTS=1 -DETX_SLOT_BY_CU -I/fleet/src -I"$ROOT/etx/runtime/include" -I"$ROOT")
for P in "$OUT"/native-B*; do
 [ -d "$P" ] || continue
 B=2;case "$P" in *B4*) B=4;;esac
 "$RL/bin/clang++" "${COMMON[@]}" -DETX_COLUMN_NATIVE_BATCH="$B" -I"$P" --cuda-device-only -Rpass-analysis=kernel-resource-usage -S "$P/megakernel_d0.hip" -o "$P/micro.s" > "$P/device.stdout.log" 2> "$P/resource.log"
 "$RL/bin/llvm-mc" -triple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj "$P/micro.s" -o "$P/micro.o" > "$P/assemble.log" 2>&1
 "$RL/bin/ld.lld" -shared "$P/micro.o" -o "$P/micro.hsaco" > "$P/link.log" 2>&1
 "$RL/bin/clang++" -x c++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ -DETX_COLUMN_DRIVER -I/code/rocm-headers/inc -I"$ROOT" -I"$ROOT/etx/runtime/include" -I"$P" "$ROOT/examples/experiment_program/columns/driver.cpp" -fuse-ld=lld -ldl -o "$P/columns-run" > "$P/host.log" 2>&1
 "$RL/bin/llvm-readobj" --notes "$P/micro.hsaco" > "$P/metadata.txt"
 ln -s micro.s "$P/mk.s"
 ln -s micro.hsaco "$P/model.hsaco"
 sha256sum "$P/micro.hsaco" "$P/micro.s" "$P/columns-run" "$P/plan.json" "$P/plan_data.h" "$P/queue-proof.json" "$ROOT/examples/experiment_program/columns/micro_tiles.hip" "$ROOT/examples/experiment_program/columns/micro_state.h" "$ROOT/examples/experiment_program/columns/micro_runtime.h" "$ROOT/examples/experiment_program/columns/driver.cpp" > "$P/SHA256SUMS"
done
' > "$OUT/micro-docker.log" 2>&1
