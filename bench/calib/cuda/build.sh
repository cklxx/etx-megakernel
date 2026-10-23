#!/usr/bin/env bash
# Compile only. No benchmark or device discovery is run here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NVCC="${NVCC:-nvcc}"
OUT="${OUT:-$ROOT/build/calib_cuda}"
ARCH="${CUDA_ARCH:-sm_90}"
mkdir -p "$OUT"
for name in sync_bench hw_bench atomic_pingpong queue_contention step_overhead p2p_sync; do
  echo "Compiling $name ($ARCH)"
  "$NVCC" -std=c++17 -O2 -arch="$ARCH" -I "$ROOT/bench/calib/cuda" \
    "$ROOT/bench/calib/cuda/$name.cu" -o "$OUT/$name"
done
