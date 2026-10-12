#!/usr/bin/env bash
# Stage a fresh box for examples/sessions/ds_inplace.py, from the laptop.
#   examples/dsv2lite/stage_inplace.sh <ip>        (ETX_SESS_USER overrides the login, as in tools/session.sh)
# Refuses a dirty ETX tree (the box has no .git, so SOURCE_IDENTITY is the only record of what ran), recreates and
# verifies the pinned fleet export (git archive 3fb67a6, never fleet's working tree), pushes with tools/session.sh,
# then copies the export to ~/fleet-3fb67a6 and writes ~/etx/SOURCE_IDENTITY. No GPU, no download on the laptop.
set -euo pipefail
IP=${1:?ip}; H=${ETX_SESS_USER:-hotaisle}@$IP
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FLEET_REPO=${ETX_DSV2_FLEET_REPO:-$HOME/code/fleet-mi300x}
EXPORT=${ETX_DSV2_FLEET_EXPORT:-/tmp/etx-dsv2/fleet-3fb67a6}
SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $H"

if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "refusing: $ROOT has uncommitted changes; commit them so SOURCE_IDENTITY names what runs" >&2; exit 2
fi
if [ ! -d "$EXPORT/src" ]; then
  mkdir -p "$EXPORT" && git -C "$FLEET_REPO" archive 3fb67a6 | tar -x -C "$EXPORT"
fi
if [ -e "$EXPORT/.git" ]; then echo "refusing: $EXPORT is a checkout, not an export" >&2; exit 2; fi
(cd "$ROOT" && .venv/bin/python - "$EXPORT" <<'PY'
import sys
from pathlib import Path
from examples.dsv2lite.inplace_session import PINNED_SRC_SHA256, source_sha
got = source_sha(Path(sys.argv[1]) / "src")
print(f"fleet export {got[:16]} (pinned {PINNED_SRC_SHA256[:16]})")
sys.exit(0 if got == PINNED_SRC_SHA256 else "refusing: fleet export is not the pinned 3fb67a6 source")
PY
)
COMMIT=$(git -C "$ROOT" rev-parse HEAD); BRANCH=$(git -C "$ROOT" branch --show-current)

bash "$ROOT/tools/session.sh" push "$IP"
rsync -az --delete "$EXPORT/" "$H:~/fleet-3fb67a6/"
printf 'commit %s\nbranch %s\ndirty 0\nstaged %s\n' "$COMMIT" "$BRANCH" "$(date -u +%FT%TZ)" | $SSH 'cat > ~/etx/SOURCE_IDENTITY'
$SSH 'cat ~/etx/SOURCE_IDENTITY; test -f ~/fleet-3fb67a6/src/host/reference_run.py && echo FLEET-EXPORT-OK'
echo "staged. next: ETX_SESS_MAX_SECONDS=7200 tools/session.sh start $IP examples/sessions/ds_inplace.py 0"
