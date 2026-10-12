#!/usr/bin/env bash
# Stage a fresh box for examples/sessions/ds_inplace.py, from the laptop.
#   examples/dsv2lite/stage_inplace.sh <ip>        (ETX_SESS_USER overrides the login, as in tools/session.sh)
# Refuses a dirty ETX tree (the box has no .git, so SOURCE_IDENTITY is the only record of what ran), recreates and
# verifies the pinned fleet export (examples/dsv2lite/pinned.py: git archive 3fb67a6), pushes with tools/session.sh,
# then copies the export to ~/fleet-3fb67a6 and writes ~/etx/SOURCE_IDENTITY. No GPU, no download on the laptop.
set -euo pipefail
IP=${1:?ip}; H=${ETX_SESS_USER:-hotaisle}@$IP
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $H"

if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "refusing: $ROOT has uncommitted changes; commit them so SOURCE_IDENTITY names what runs" >&2; exit 2
fi
# The pinned export (git archive 3fb67a6, content-hash checked) from examples/dsv2lite/pinned.py; never the working tree.
EXPORT=$(cd "$ROOT" && .venv/bin/python -m examples.dsv2lite.pinned export) || { echo "refusing: no verified fleet export" >&2; exit 2; }
echo "fleet export $EXPORT (verified against the pinned 3fb67a6 hash)"
COMMIT=$(git -C "$ROOT" rev-parse HEAD); BRANCH=$(git -C "$ROOT" branch --show-current)

bash "$ROOT/tools/session.sh" push "$IP"
rsync -az --delete "$EXPORT/" "$H:~/fleet-3fb67a6/"
printf 'commit %s\nbranch %s\ndirty 0\nstaged %s\n' "$COMMIT" "$BRANCH" "$(date -u +%FT%TZ)" | $SSH 'cat > ~/etx/SOURCE_IDENTITY'
$SSH 'cat ~/etx/SOURCE_IDENTITY; test -f ~/fleet-3fb67a6/src/host/reference_run.py && echo FLEET-EXPORT-OK'
echo "staged. next: ETX_SESS_MAX_SECONDS=7200 tools/session.sh start $IP examples/sessions/ds_inplace.py 0"
