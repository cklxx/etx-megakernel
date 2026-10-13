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

# EXACTLY the tracked files (review 07, R08: ignored .pyc and other untracked files rode along while `git status` was
# clean). Anything else in ~/etx is deleted (the box's own .venv and build/ excepted), and the box recomputes the tree
# hash -- sha256 over `sha256  path` lines of the tracked files in `git ls-files` order -- which must equal ours.
FILES=$(mktemp); git -C "$ROOT" ls-files -z > "$FILES"
TREE=$(cd "$ROOT" && xargs -0 shasum -a 256 < "$FILES" | shasum -a 256 | cut -c1-64)
rsync -az -e "$RSH" --from0 --files-from="$FILES" "$ROOT/" "$H:etx/"
BOX_TREE=$(tr '\0' '\n' < "$FILES" | ssh "${OPTS[@]}" "$H" 'set -e; cd ~/etx; cat > SOURCE_FILES
  find . -type f ! -path "./.venv/*" ! -path "./build/*" ! -name SOURCE_FILES ! -name SOURCE_IDENTITY | sed "s#^\./##" | sort > /tmp/etx_have
  sort SOURCE_FILES > /tmp/etx_want; comm -23 /tmp/etx_have /tmp/etx_want | xargs -r -d "\n" rm -f
  find . -name __pycache__ ! -path "./.venv/*" -prune -exec rm -rf {} +
  tr "\n" "\0" < SOURCE_FILES | xargs -0 sha256sum | sha256sum | cut -c1-64')
rm -f "$FILES"
[ "$BOX_TREE" = "$TREE" ] || { echo "refusing: the box tree $BOX_TREE differs from the commit's $TREE" >&2; exit 2; }
rsync -az -e "$RSH" --delete "$EXPORT/" "$H:fleet-3fb67a6/"
printf 'commit %s\nbranch %s\ndirty 0\ntree %s\nstaged %s\n' "$COMMIT" "$BRANCH" "$TREE" "$(date -u +%FT%TZ)" | ssh "${OPTS[@]}" "$H" 'cat > ~/etx/SOURCE_IDENTITY'
ssh "${OPTS[@]}" "$H" 'cat ~/etx/SOURCE_IDENTITY; test -f ~/fleet-3fb67a6/src/host/reference_run.py && echo FLEET-EXPORT-OK
  hipcc --version 2>/dev/null | head -1
  printf "#include <hip/hip_runtime.h>\n" | hipcc -x hip --offload-arch=gfx942 -fsyntax-only - >/dev/null 2>&1 \
    && echo HIP-HEADERS-OK || echo "HIP-HEADERS-MISSING: see REPRODUCE.md step 2"
  rocm-smi --showproductname 2>/dev/null | grep -m1 "Card Series"; nproc | sed "s/^/cores: /"; df -h ~ | tail -1'
echo "staged. next: examples/dsv2lite/reproduce.sh run $IP"
