#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?root}; FLEET=${2:?sealed fleet export}; OUT=${3:?output}
FLEET_SRC=${ETX_COLUMN_FLEET_SRC:?accepted cooked Fleet source required}
HEADERS=${ETX_ROCM_HEADERS:-$HOME/code/rocm-headers}
docker image inspect etx-rocm22 --format '{{.Id}}' > "$OUT/image.txt"
docker run --rm --pull=never --network=none --platform linux/amd64 \
 -v "$ROOT:/etx:ro" -v "$FLEET:/fleet:ro" -v "$FLEET_SRC:/fleet/src:ro" -v "$OUT:/out" -v "$HEADERS:/rocm:ro" \
 -v "$HOME/.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/link-deps/root:/link-deps:ro" \
 -e LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu -w /etx etx-rocm22 bash -c '
set -euo pipefail
RL=/rocm/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=/rocm/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
CXX=$RL/bin/clang++
mkdir -p /out/compiler-tmp
export TMPDIR=/out/compiler-tmp
"$CXX" --version > /out/compiler.txt
sha256sum /link-deps/usr/lib/x86_64-linux-gnu/libxml2.so.2* > /out/link_deps.sha256
COMMON=(-x hip --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path="$DL" -isystem /rocm/inc -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -DFLEET_NT_WEIGHTS=1 -I/fleet/src -I/etx/etx/runtime/include -I/out -I/etx)
declare -p COMMON > /out/compiler_args.txt
SRC=examples/experiment_program/columns
sha256sum "$SRC/kernels.hip" "$SRC/historical.hip" "$SRC/driver.cpp" /etx/examples/experiment_program/hip_dynamic.h > /out/candidate_sources.sha256
for name in columns b5eaa67 ff9ef92; do
 if [ "$name" = columns ]; then input=$SRC/kernels.hip; EXTRA=(); else
  input=$SRC/historical.hip
  cp "/out/historical_$name.h" /out/historical_stream.h
  EXTRA=(); if [ "$name" = ff9ef92 ]; then EXTRA=(-DHISTORICAL_FF); fi
 fi
 "$CXX" "${COMMON[@]}" "${EXTRA[@]}" --cuda-device-only -Rpass-analysis=kernel-resource-usage -S "$input" -o "/out/$name.s" > "/out/$name.stdout.log" 2> "/out/$name.resource.log"
 "$CXX" -target amdgcn-amd-amdhsa -mcpu=gfx942 -c "/out/$name.s" -o "/out/$name.o" > "/out/$name.object.log" 2>&1
 "$RL/bin/ld.lld" -shared "/out/$name.o" -o "/out/$name.hsaco" > "/out/$name.link.log" 2>&1
 "$RL/bin/llvm-readobj" --notes "/out/$name.hsaco" > "/out/$name.metadata.txt"
 "$RL/bin/llvm-objdump" -d --mcpu=gfx942 "/out/$name.hsaco" > "/out/$name.isa.txt"
done
"$CXX" -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -I/rocm/inc -I/etx "$SRC/driver.cpp" -fuse-ld=lld -ldl -o /out/columns-run > /out/host.log 2>&1
"$CXX" -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -I/rocm/inc -I/etx "$SRC/capture_compile.cpp" -fuse-ld=lld -ldl -o /out/capture-hook-check > /out/capture_host.log 2>&1
sha256sum /out/*.hsaco /out/columns-run /out/*.s /out/compiler_args.txt /out/sources.json /out/candidate_sources.sha256 /out/link_deps.sha256 > /out/SHA256SUMS
' > "$OUT/docker.log" 2>&1
cat "$OUT/docker.log"
