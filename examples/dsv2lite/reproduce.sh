#!/usr/bin/env bash
# One entry point from a fresh clone to the 2026-10-13 ds_inplace numbers. See examples/dsv2lite/REPRODUCE.md.
#   reproduce.sh setup                 create .venv (python >= 3.11) with the dev extra
#   reproduce.sh check                 local, no GPU: every step of the measured chain, expected outputs asserted
#   reproduce.sh box-setup <ip>        give the box's hipcc its HIP headers if they are missing (ROCm 7.14 boxes)
#   reproduce.sh stage <ip>            push this clone + pinned fleet export + SOURCE_IDENTITY (stage_inplace.sh)
#   reproduce.sh run <ip>              start examples/sessions/ds_inplace.py on GPU 0, hard stop 7200 s
#   reproduce.sh status <ip>           runner state
#   reproduce.sh fetch <ip> <dest>     pull ~/sess (logs, JSON, traces, stamps; no binaries) + SHA256SUMS
#   reproduce.sh rederive <dir>        re-derive every verdict of a fetched session from its raw logs
# ETX_SESS_USER is the box login (default root). ETX_CACHE (default ~/code/etx-cache) holds the pinned fleet export and
# the archived data sets; ETX_FLEET_REPO (default ~/code/fleet-mi300x) is a clone of github.com/cklxx/fleet-mi300x.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python
MEASURED_COMMIT=7f3dd89f8cc92648b4fa35f9000028059658cfee      # SOURCE_IDENTITY of the 2026-10-13 session
SESSION_DIR=examples/dsv2lite/results/2026-10-13-ds_inplace
export ETX_SESS_USER=${ETX_SESS_USER:-root}

step() { printf '\n== %s\n' "$*"; }
ok() { printf 'OK   %s\n' "$*"; }
die() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

mux() {   # one multiplexed SSH connection for the rest of this script (boxes may rate-limit new connections)
  CM_DIR=$(mktemp -d /tmp/etxssh.XXXXXX)
  OPTS=(-o ControlMaster=auto -o "ControlPath=$CM_DIR/cm" -o ControlPersist=600 -o ConnectTimeout=15
        -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
  ssh "${OPTS[@]}" -fN "$ETX_SESS_USER@$1"
  trap 'ssh -o "ControlPath=$CM_DIR/cm" -O exit "$ETX_SESS_USER@'"$1"'" >/dev/null 2>&1; rm -rf "$CM_DIR"' EXIT
}

case "${1:-}" in
setup)
  python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version' || die "python >= 3.11 required"
  python3 -m venv .venv && $PY -m pip install -q -e '.[dev]' && ok ".venv ready: $($PY --version)"
  ;;
check)
  [ -x $PY ] || die "run: $0 setup"
  step "1. box-side code is the measured commit (${MEASURED_COMMIT:0:7}); later commits may only ADD files"
  git cat-file -e "$MEASURED_COMMIT^{commit}" || die "commit $MEASURED_COMMIT not in this clone"
  git diff --quiet --diff-filter=MDR "$MEASURED_COMMIT" HEAD -- etx examples/dsv2lite examples/sessions/ds_inplace.py \
    ':!examples/dsv2lite/stage_inplace.sh' ':!examples/dsv2lite/results' \
    || die "box-side code differs from $MEASURED_COMMIT: $(git diff --name-only --diff-filter=MDR "$MEASURED_COMMIT" HEAD -- etx examples/dsv2lite examples/sessions/ds_inplace.py ':!examples/dsv2lite/stage_inplace.sh' ':!examples/dsv2lite/results' | tr '\n' ' ')"
  ok "no box-side file modified or removed since ${MEASURED_COMMIT:0:7}"
  step "2. pinned fleet source (git archive 3fb67a6)"
  out=$($PY -m examples.dsv2lite.pinned export) || die "fleet export"
  ok "$out = 3fb67a6, src sha256 28f5cbb4f39524df..."
  step "3. archived data sets (cklxx/etx-data release data-2026-09-27, tarball + per-file SHA256)"
  out=$($PY -m examples.dsv2lite.pinned data) || die "data sets"
  ok "$out"
  step "4. CPU gates (tests/test_dsv2_*.py)"
  out=$($PY -m pytest -q -rs tests/test_dsv2_*.py 2>&1 | tail -1)
  echo "$out" | grep -q -E '^[0-9]+ passed in' || die "CPU gates: $out"
  ok "$out"
  step "5. dry run: the same 33-job table as the measured session"
  logdir=$(mktemp -d /tmp/etxdry.XXXXXX)
  $PY -m etx.tools.gpu_jobs examples/sessions/ds_inplace.py --gpus 0 --logdir "$logdir" --dry-run > "$logdir/table.txt" 2>&1
  diff -u "$SESSION_DIR/DRYRUN.txt" "$logdir/table.txt" || die "dry-run table differs from $SESSION_DIR/DRYRUN.txt"
  ok "$(grep -c 'timeout' "$logdir/table.txt") jobs, identical to $SESSION_DIR/DRYRUN.txt"
  step "6. expected per-stage values regenerate byte for byte (stage_table.py: trace partition + replay)"
  $PY -m examples.dsv2lite.stage_table > /dev/null
  git diff --exit-code --stat -- examples/dsv2lite/results/stages || die "stage table regenerated differently"
  ok "$(grep '^\*\*Summary, arithmetic' examples/dsv2lite/results/stages/stage_table.md | tr -d '*')"
  step "7. every verdict re-derived from the session's raw logs (incl. the combined arm's 92 us)"
  $PY -m examples.dsv2lite.rederive "$SESSION_DIR" | tee "$logdir/rederive.txt"
  grep -q '^  all      vs default median    92.0 spread  18.0 every_positive=1 GO=True' "$logdir/rederive.txt" \
    || die "the combined arm's verdict did not re-derive"
  ok "all vs default: median 92.0 us, spread 18.0, every pair faster, GO"
  ;;
box-setup)
  IP=${2:?ip}; mux "$IP"
  ssh "${OPTS[@]}" "$ETX_SESS_USER@$IP" 'set -e
    if printf "#include <hip/hip_runtime.h>\n" | hipcc -x hip --offload-arch=gfx942 -fsyntax-only - >/dev/null 2>&1; then
      echo HIP-HEADERS-OK
    else
      v=$(hipcc --version | sed -n "s/^HIP version: \([0-9]*\.[0-9]*\).*/\1/p")
      echo "installing amdrocm-runtime-dev$v (the HIP headers of the installed ROCm)"
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "amdrocm-runtime-dev$v" >/dev/null
      printf "#include <hip/hip_runtime.h>\n" | hipcc -x hip --offload-arch=gfx942 -fsyntax-only - && echo HIP-HEADERS-OK
    fi
    hipcc --version | head -2'
  ;;
stage)
  examples/dsv2lite/stage_inplace.sh "${2:?ip}"
  ;;
run)
  ETX_SESS_MAX_SECONDS=7200 tools/session.sh start "${2:?ip}" examples/sessions/ds_inplace.py 0
  ;;
status)
  tools/session.sh status "${2:?ip}"
  ;;
fetch)
  IP=${2:?ip}; DEST=${3:?dest}
  [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ] && die "$DEST is not empty: fetch into a new directory (review 08, R12)"
  mux "$IP"; mkdir -p "$DEST"
  rsync -az -e "ssh ${OPTS[*]}" --exclude 'b_*/run' --exclude 'b_*/fleet_src' --exclude '*.o' \
    "$ETX_SESS_USER@$IP:sess/" "$DEST/"
  # the box's SHA256SUMS is the provenance record: verify the copy against it, never overwrite it (review 07, R04)
  if [ -f "$DEST/SHA256SUMS" ]; then
    $PY etx/tools/verify_sums.py "$DEST" --excluded 'b_*/run' 'b_*/fleet_src/*' '*.o' || die "fetch does not match the box's SHA256SUMS: keep the box"
  else
    (cd "$DEST" && find . -type f ! -name 'SHA256SUMS*' | sort | xargs shasum -a 256 > SHA256SUMS.local)
    echo "no box-side SHA256SUMS: only SHA256SUMS.local (hashes of the copy) exists"
  fi
  ok "fetched to $DEST; next: $0 rederive $DEST (ds_inplace) or examples/dsv2lite/verify_hour.py $DEST (ds_hour)"
  ;;
rederive)
  DIR=${2:?session dir}
  $PY -m examples.dsv2lite.rederive "$DIR"
  # A new session is the same experiment only with the measured inputs and sources (the HF model revision is not
  # pinned by the download; the weights' hash is what identifies it).
  $PY - "$DIR" "$SESSION_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
new, ref = (Path(p) for p in sys.argv[1:3])
same = True
for f, key in (("inputs.json", "sha256"), ("source.json", "fleet_src_sha256")):
    a, b = json.loads((new / f).read_text())[key], json.loads((ref / f).read_text())[key]
    print(f"{f}: {'SAME as the measured session' if a == b else 'DIFFERENT from the measured session'}")
    same &= a == b
sys.exit(0 if same else 3)
PYEOF
  ;;
*)
  sed -n '2,13p' "$0"; exit 2
  ;;
esac
