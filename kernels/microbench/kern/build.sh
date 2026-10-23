#!/usr/bin/env bash
# Compile every copied kernel to a cubin.
# Usage: ARCH=sm_89 bash build.sh [outdir]
# Kernels that need Hopper features (griddepcontrol, cluster_group) are retried on sm_90.
# Runs on the desktop inside the CUDA devel image; no GPU is needed to compile.
set -uo pipefail
ARCH=${ARCH:-sm_89}
FALLBACK=${FALLBACK:-sm_90}
OUT=${1:-cubins}
IMAGE=${IMAGE:-docker.m.daocloud.io/nvidia/cuda:12.4.1-devel-ubuntu22.04}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT"
fail=0
for f in "$HERE"/src/*.cu; do
  name=$(basename "$f" .cu)
  printf '%-18s ' "$name"
  if out=$(docker run --rm -v "$HERE":/w -w /w "$IMAGE" \
        nvcc -cubin -arch="$ARCH" -O3 -lineinfo -o "/w/$OUT/$name.cubin" "/w/src/$name.cu" 2>&1); then
    printf 'OK      %s  (sm_89)\n' "$(ls -l "$OUT/$name.cubin" | awk '{print $5}') B"
  elif out2=$(docker run --rm -v "$HERE":/w -w /w "$IMAGE" \
        nvcc -cubin -arch="$FALLBACK" -O3 -lineinfo -o "/w/$OUT/$name.$FALLBACK.cubin" "/w/src/$name.cu" 2>&1); then
    printf 'OK      %s  (%s only, needs Hopper ISA)\n' "$(ls -l "$OUT/$name.$FALLBACK.cubin" | awk '{print $5}') B" "$FALLBACK"
  else
    printf 'FAIL\n'
    echo "$out2" | grep -E 'error' | head -2 | sed 's/^/    /'
    fail=$((fail+1))
  fi
done
echo "---"
echo "cubins=$(ls "$OUT"/*.cubin 2>/dev/null | wc -l | tr -d ' ') failed=$fail arch=$ARCH fallback=$FALLBACK"
