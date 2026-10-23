#!/usr/bin/env bash
# Run one command inside the ROCm 7.2.4 GLM build/run image, so the binary is compiled, linked and
# executed against the same toolchain and HIP runtime (ROCm 6.4.1 cannot keep 2 waves/SIMD; a binary
# linked here needs libamdhip64.so.7 at run time).
#
#   bash examples/glm53/rocm724.sh <dir>... -- <command...>
#
# The repository and every <dir> are mounted at their own absolute paths; the command runs from the
# repository root. GPUs (/dev/kfd, /dev/dri) are passed through when present. The image is built once
# from $ETX_ROCM_BASE (default rocm/dev-ubuntu-22.04:7.2.4) plus pyyaml and numpy; it is the only step
# that needs the network. Generation under its Python 3.10 is byte-identical to Python 3.14 (verified
# on pp2-b1, 13 generated files).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
BASE=${ETX_ROCM_BASE:-rocm/dev-ubuntu-22.04:7.2.4}
IMAGE=${ETX_ROCM_IMAGE:-etx-glm-rocm724}
DOCKER=${DOCKER:-docker}
mounts=(-v "$ROOT:$ROOT")
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  dir=$(cd "$1" && pwd); mounts+=(-v "$dir:$dir"); shift
done
[ "${1:-}" = -- ] || { echo "usage: $0 <dir>... -- <command...>" >&2; exit 2; }
shift
if ! $DOCKER image inspect "$IMAGE" > /dev/null 2>&1; then
  printf 'FROM %s\nRUN pip3 install --no-cache-dir pyyaml numpy\n' "$BASE" | $DOCKER build -t "$IMAGE" -
fi
gpu=()
if [ -e /dev/kfd ]; then
  gpu=(--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --security-opt seccomp=unconfined)
fi
envs=()
for name in ETX_GLM_BATCH ETX_GLM_GATE ETX_GLM_FIXTURE ETX_GLM_OPTIONS ETX_GLM_CUS HIP_VISIBLE_DEVICES; do
  if [ -n "${!name:-}" ]; then envs+=(-e "$name=${!name}"); fi
done
exec $DOCKER run --rm --pull=never "${gpu[@]}" "${mounts[@]}" "${envs[@]}" -e ETX_PYTHON=python3 \
  -e PYTHONPATH="$ROOT" -w "$ROOT" "$IMAGE" bash -lc "$(printf '%q ' "$@")"
