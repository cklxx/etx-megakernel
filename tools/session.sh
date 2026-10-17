#!/usr/bin/env bash
# One GPU session on one or two Hot Aisle VMs, driven from the laptop.  Each VM runs its own etx.tools.gpu_jobs runner
# (jobs tagged machine="A" / "B" in the session file), fully detached from ssh, logging to ~/sess.
#
#   tools/session.sh stage  <ip>                     this repository's HEAD, as an archive, in a NEW box directory; ~/etx -> it
#   tools/session.sh push   <ip>                     stage, then rsync fleet-mi300x, vLLM csrc, pytorch headers
#   tools/session.sh preflight <ip>                check pushed tree, ROCm, dry run and Docker (no pull)
#   tools/session.sh postmortem <ip> <dest>          fetch and report failed/skipped jobs and runner minutes
#   tools/session.sh start  <ip> <session.py> [gpus] [machine]   start the runner (gpus default: all the VM has)
#   tools/session.sh status <ip>                     runner alive?, last 25 lines of its log
#   tools/session.sh fetch  <ip> <dest> [id]         copy the box's newest (or named) session record into <dest>/<id>
#                                                    and verify it against its own SEAL (etx/session/record.py)
set -uo pipefail
# ETX_SESS_USER overrides the login (Hot Aisle VMs use hotaisle; other providers may give root)
CMD=${1:?stage|push|preflight|start|status|fetch|postmortem}; IP=${2:?ip}; H=${ETX_SESS_USER:-hotaisle}@$IP
SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $H"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
case $CMD in
  stage)
    # identity by construction: the box extracts the archive whose hash it checked into an empty directory, so
    # nothing untracked, ignored or left by an earlier session can be in the tree that runs (review 07/08, R08)
    [ -z "$(git -C "$ROOT" status --porcelain)" ] || { echo "refusing: $ROOT has uncommitted changes"; exit 2; }
    TAR=$(mktemp); git -C "$ROOT" archive --format=tar.gz HEAD > "$TAR" || exit 1
    SUM=$(shasum -a 256 < "$TAR" | cut -c1-64)
    $SSH "mkdir -p ~/src && cat > ~/src/.incoming.$SUM" < "$TAR" || exit 1
    rm -f "$TAR"
    $SSH bash -s -- "$SUM" "$(git -C "$ROOT" rev-parse HEAD)" "$(git -C "$ROOT" branch --show-current)" <<'REMOTE'
set -e; SUM=$1; IN=~/src/.incoming.$SUM; D=~/src/$SUM-$(date -u +%Y%m%dT%H%M%SZ)
[ "$(sha256sum < "$IN" | cut -c1-64)" = "$SUM" ] || { echo "refusing: the box received an archive that is not $SUM"; exit 2; }
mkdir "$D"; tar -xz -C "$D" -f "$IN"; rm -f "$IN"
printf 'commit %s\nbranch %s\narchive %s\nstaged %s\n' "$2" "$3" "$SUM" "$(date -u +%FT%TZ)" > "$D/SOURCE_IDENTITY"
if [ -e ~/etx ] && [ ! -L ~/etx ]; then mv ~/etx ~/src/unstaged-"$(date -u +%s)"; fi
ln -sfn "$D" ~/etx; echo "STAGED ~/etx -> $D"; cat "$D/SOURCE_IDENTITY"
REMOTE
    ;;
  push)
    ssh-keygen -R "$IP" >/dev/null 2>&1
    until $SSH true 2>/dev/null; do sleep 5; done
    bash "$0" stage "$IP" || exit 1
    cd ~/code || exit 1
    rsync -az --exclude .git --exclude .venv --exclude build --exclude __pycache__ fleet-mi300x/ $H:~/fleet-mi300x/ &&
    rsync -az vllm/csrc $H:~/vllm/ && rsync -az --exclude .git rocm-headers/pytorch $H:~/rocm-headers/ &&
    rsync -azL rocm-headers/inc $H:~/rocm-headers/ &&
    { [ ! -d ~/code/etx-cache/fleet_ref ] || rsync -az ~/code/etx-cache/fleet_ref/ $H:~/fleet-mi300x/; } &&   # kept reference outputs
    $SSH 'rocm-smi --showproductname | grep -c "Card Series" | sed "s/^/GPUs: /"; nproc | sed "s/^/cores: /"; df -h ~ | tail -1' ;;
  preflight)
    $SSH bash -s <<'REMOTE'
failed=0
check() {
    if ! eval "$2"; then
        printf 'PREFLIGHT FAIL %s: %s\n' "$1" "$2"
        failed=1
    fi
}
check pushed-tree 'test -f ~/rocm-headers/inc/hip/hip_runtime.h && test -d ~/rocm-headers/pytorch && test -f ~/etx/etx/tools/gpu_jobs.py && echo PUSH-OK'
version_cmd='cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm/core-*/.info/version 2>/dev/null || hipconfig --version 2>/dev/null'
version=$(eval "$version_cmd")
if [ -z "$version" ]; then
    echo 'PREFLIGHT WAIT rocm-version: retry in 300 s'
    sleep 300
    version=$(eval "$version_cmd")
fi
if [ -z "$version" ]; then
    printf 'PREFLIGHT FAIL rocm-version: %s\n' "$version_cmd"
    failed=1
fi
check dry-run 'cd ~/etx && python3 -m examples.sessions.vlbest --dry-run'
check container 'sudo docker info >/dev/null && echo DOCKER-OK'
[ "$failed" = 0 ] || exit 1
printf 'PREFLIGHT OK\nROCM-VERSION %s\n' "$version"
REMOTE
    ;;
  postmortem)
    DEST=${3:?dest}
    # Always inspect partial captures, but never hide a failed transfer. Every record has its own directory, so a
    # re-fetch cannot mix sessions; a box with no record (the runner never started) shows its startup log.
    fetch_rc=0
    bash "$0" fetch "$IP" "$DEST" || fetch_rc=$?
    [ "$fetch_rc" != 3 ] || fetch_rc=0                 # captured; the record says what is incomplete
    if [ "$fetch_rc" != 0 ]; then
      printf 'UNCAPTURED %s (fetch failed)\n' "$DEST"
      $SSH 'for f in ~/sess/startup.*; do [ -f "$f" ] && { echo "STARTUP $f"; tail -20 "$f"; }; done' 2>/dev/null
    fi
    python3 "$(dirname "$0")/session_postmortem.py" "$DEST"
    report_rc=$?
    [ "$fetch_rc" = 0 ] && [ "$report_rc" = 0 ]
    ;;
  start)
    SESS=${3:?session file (path inside the etx repo)}; GPUS=${4:-}; M=${5:-A}
    REMOTE_BASH=bash
    if [ "${ETX_SESS_MAX_SECONDS+x}" = x ]; then
      CAP=$(printf '%s' "$ETX_SESS_MAX_SECONDS" | sed "s/'/'\\\\''/g")
      REMOTE_BASH="ETX_SESS_MAX_SECONDS='$CAP' bash"
      printf 'Passing session cap: ETX_SESS_MAX_SECONDS=%s\n' "$ETX_SESS_MAX_SECONDS"
    fi
    if [ "${ETX_QWEN_G6_CONFIG+x}" = x ]; then
      QWEN_CONFIG=$(printf '%s' "$ETX_QWEN_G6_CONFIG" | sed "s/'/'\\\\''/g")
      REMOTE_BASH="ETX_QWEN_G6_CONFIG='$QWEN_CONFIG' $REMOTE_BASH"
      printf 'Passing Qwen config: ETX_QWEN_G6_CONFIG=%s\n' "$ETX_QWEN_G6_CONFIG"
    fi
    # Session inputs that live on the box: the packed Qwen model, the GLM stage, its HF reference and batch.
    for name in ETX_QWEN_BATCH_MODEL ETX_GLM_STAGE ETX_GLM_REFERENCE ETX_GLM_BATCH; do
      [ "${!name+x}" = x ] || continue
      value=$(printf '%s' "${!name}" | sed "s/'/'\\\\''/g")
      REMOTE_BASH="$name='$value' $REMOTE_BASH"
      printf 'Passing %s=%s\n' "$name" "${!name}"
    done
    $SSH "$REMOTE_BASH" -s -- "$SESS" "${GPUS:-auto}" "$M" <<'REMOTE'
SESS=$1; G=$2; M=$3; [ "$G" = auto ] && G=""
cd ~/etx || exit 1
[ -n "$G" ] || G=$(rocm-smi --showproductname | grep -c "Card Series" | awk '{for (i = 0; i < $1; i++) printf "%s%d", (i ? "," : ""), i}')
mkdir -p ~/sess
# The runner creates a new record ~/sess/<session id> per start (etx/session/record.py); startup.* stays outside it.
# Keep startup errors too, without truncating any earlier session's output.
STARTUP=$(mktemp "$HOME/sess/startup.XXXXXX") || exit 1
setsid nohup python3 -m etx.tools.gpu_jobs "$SESS" --gpus "$G" --logdir ~/sess --machine "$M" > "$STARTUP" 2>&1 < /dev/null &
sleep 2; echo "runner started: machine $M, GPUs $G; startup log: $STARTUP"; head -5 "$STARTUP"
REMOTE
    ;;
  status)
    $SSH 'pgrep -f etx.tools.gpu_jobs >/dev/null && echo RUNNING || echo "runner not running"; find ~/sess -name runner.log -print -exec tail -25 {} \;' ;;
  fetch)
    DEST=${3:?dest}; ID=${4:-$($SSH 'ls -1t ~/sess 2>/dev/null | grep -v "^startup\." | head -1')}
    [ -n "$ID" ] || { echo "NO RECORD on the box (~/sess holds no session): nothing to fetch"; exit 4; }
    # the destination is the session id itself: a fetch can never land on another session's files (R12)
    [ ! -e "$DEST/$ID" ] || { echo "refusing: $DEST/$ID exists; it is already fetched"; exit 2; }
    mkdir -p "$DEST"; rm -rf "$DEST/$ID.partial"
    rsync -az $H:"sess/$ID/SEAL" "$DEST/$ID.partial/" 2>/dev/null
    EXCL=()
    if [ -f "$DEST/$ID.partial/SEAL" ]; then
      while IFS= read -r g; do EXCL+=(--exclude "$g"); done < <(python3 "$ROOT/etx/session/record.py" local-only "$DEST/$ID.partial")
    fi
    rsync -az ${EXCL[@]+"${EXCL[@]}"} $H:"sess/$ID/" "$DEST/$ID.partial/" || exit 1
    mv "$DEST/$ID.partial" "$DEST/$ID"
    ls "$DEST/$ID" | head -50
    python3 "$ROOT/etx/session/record.py" verify "$DEST/$ID"; rc=$?      # 0 complete, 3 incomplete, 2 not verified
    [ "$rc" != 2 ] || { echo "FETCH NOT VERIFIED: keep the box"; exit 2; }
    # keep the fleet reference outputs (goldens, KV caches) for the next VM: no reference re-run (docs/DATA.md)
    $SSH 'test -d ~/fleet-mi300x/build' 2>/dev/null || exit "$rc"
    mkdir -p ~/code/etx-cache/fleet_ref/build ~/code/etx-cache/fleet_ref/build_r1
    capture_rc=0
    rsync -az $H:'fleet-mi300x/build/golden*' $H:fleet-mi300x/build/fleet_cache.bin ~/code/etx-cache/fleet_ref/build/ || {
      echo "UNCAPTURED $H:fleet-mi300x/build/golden* or fleet_cache.bin"; capture_rc=1;
    }
    rsync -az $H:fleet-mi300x/build_r1/ ~/code/etx-cache/fleet_ref/build_r1/ || {
      echo "UNCAPTURED $H:fleet-mi300x/build_r1/"; capture_rc=1;
    }
    du -sh ~/code/etx-cache/fleet_ref
    [ "$capture_rc" = 0 ] || exit 1
    exit "$rc" ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
