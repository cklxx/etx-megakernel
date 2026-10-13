#!/usr/bin/env bash
# Stage a fresh box for examples/sessions/ds_inplace.py from a clone of this repository.
#   examples/dsv2lite/stage_inplace.sh <ip>        (ETX_SESS_USER: the login, default root)
# Self-contained: pushes THIS clone (no other directory layout assumed), the pinned fleet export
# (examples/dsv2lite/pinned.py: git archive 3fb67a6, content-hash checked) and ~/etx/SOURCE_IDENTITY (the box has no
# .git, so this file is the only record of what ran). Refuses a dirty tree. Uses ONE multiplexed SSH connection:
# boxes that rate-limit new SSH connections (ufw limit) refused a burst of rsyncs on 2026-10-13. Checks that the box's
# hipcc finds HIP headers; installs none itself (see REPRODUCE.md step 2).
set -euo pipefail
IP=${1:?ip}; H=${ETX_SESS_USER:-root}@$IP
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "refusing: $ROOT has uncommitted changes; commit them so SOURCE_IDENTITY names what runs" >&2; exit 2
fi
EXPORT=$(cd "$ROOT" && .venv/bin/python -m examples.dsv2lite.pinned export) || { echo "refusing: no verified fleet export" >&2; exit 2; }
echo "fleet export $EXPORT (verified against the pinned 3fb67a6 hash)"
COMMIT=$(git -C "$ROOT" rev-parse HEAD); BRANCH=$(git -C "$ROOT" branch --show-current)

CM_DIR=$(mktemp -d /tmp/etxssh.XXXXXX)              # short path: macOS limits the control socket path to 104 bytes
OPTS=(-o ControlMaster=auto -o "ControlPath=$CM_DIR/cm" -o ControlPersist=600 -o ConnectTimeout=15
      -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
ssh "${OPTS[@]}" -fN "$H"
trap 'ssh -o "ControlPath=$CM_DIR/cm" -O exit "$H" >/dev/null 2>&1; rm -rf "$CM_DIR"' EXIT
RSH="ssh ${OPTS[*]}"

rsync -az -e "$RSH" --delete --exclude .git --exclude .venv --exclude build --exclude __pycache__ \
  --exclude 'results_*' "$ROOT/" "$H:etx/"
rsync -az -e "$RSH" --delete "$EXPORT/" "$H:fleet-3fb67a6/"
printf 'commit %s\nbranch %s\ndirty 0\nstaged %s\n' "$COMMIT" "$BRANCH" "$(date -u +%FT%TZ)" | ssh "${OPTS[@]}" "$H" 'cat > ~/etx/SOURCE_IDENTITY'
ssh "${OPTS[@]}" "$H" 'cat ~/etx/SOURCE_IDENTITY; test -f ~/fleet-3fb67a6/src/host/reference_run.py && echo FLEET-EXPORT-OK
  hipcc --version 2>/dev/null | head -1
  printf "#include <hip/hip_runtime.h>\n" | hipcc -x hip --offload-arch=gfx942 -fsyntax-only - >/dev/null 2>&1 \
    && echo HIP-HEADERS-OK || echo "HIP-HEADERS-MISSING: see REPRODUCE.md step 2"
  rocm-smi --showproductname 2>/dev/null | grep -m1 "Card Series"; nproc | sed "s/^/cores: /"; df -h ~ | tail -1'
echo "staged. next: examples/dsv2lite/reproduce.sh run $IP"
