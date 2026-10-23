#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
# Compile and run in sequence; never overlap GPU benchmark processes.
for name in argmax topk_row spec_round markov_rows copy_rows dflash_conv dflash_select gdn_advance line_copy peer_barrier peer_collective weight_prep; do
  /opt/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 "$root/drivers/$name.cu" -o "$build/$name"
  if [[ ${BENCH_COMPILE_ONLY:-0} != 1 ]]; then
    timeout 60s "$build/$name"
  fi
done
