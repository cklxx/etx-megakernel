#!/usr/bin/env bash
# CPU-only compile/link. Use an existing ROCm directory or local Docker image.
# bash tools/cpu_device_check.sh <codegen-dir> <rocm-directory-or-image>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(cd "${1:?codegen directory}" && pwd)"
TOOLCHAIN=${2:?ROCm directory or local image}
if [[ ! -d "$TOOLCHAIN" ]]; then
  docker image inspect "$TOOLCHAIN" --format '{{.Id}} {{json .RepoDigests}}' > "$OUT/compiler-image.txt"
  exec docker run --rm --network none --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$ROOT,dst=/src,readonly" \
    --mount "type=bind,src=$OUT,dst=/out" --workdir /src "$TOOLCHAIN" \
    bash /src/tools/cpu_device_check.sh /out /opt/rocm
fi
TOOLCHAIN="$(cd "$TOOLCHAIN" && pwd)"
export ROCM_PATH="$TOOLCHAIN" HIP_PATH="$TOOLCHAIN" HIP_CLANG_PATH="$TOOLCHAIN/llvm/bin"
export HCC_AMDGPU_TARGET=gfx942
export LD_LIBRARY_PATH="$TOOLCHAIN/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$TOOLCHAIN/bin/hipcc" --version > "$OUT/compiler-version.txt"
sha256sum "$TOOLCHAIN/llvm/bin/clang" > "$OUT/compiler-sha256.txt"
cd "$ROOT"
for ll in "$OUT"/imports/*.ll; do
  echo "Verify imported IR $ll"
  "$TOOLCHAIN/llvm/bin/llvm-link" "$ll" -o "${ll%.ll}.bc"
  "$TOOLCHAIN/llvm/bin/opt" -passes=verify -disable-output "${ll%.ll}.bc"
done
for pair in splitk_sum:splitk moe_layer:moe gemm_rs:gemm_rs; do
  graph=${pair%%:*}; tile=${pair##*:}
  echo "Compile/link $graph (no execution)"
  "$TOOLCHAIN/bin/hipcc" -O2 --offload-arch=gfx942 -fgpu-rdc \
    -I etx/runtime/include -I "$OUT/$graph" \
    "$OUT/$graph"/megakernel_d*.hip "examples/tiles/$tile.hip" "examples/hosts/$tile.hip" \
    -o "$OUT/$graph/run"
done
