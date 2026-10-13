#!/usr/bin/env bash
# One GPU session on one or two Hot Aisle VMs, driven from the laptop.  Each VM runs its own etx.tools.gpu_jobs runner
# (jobs tagged machine="A" / "B" in the session file), fully detached from ssh, logging to ~/sess.
#
#   tools/session.sh push   <ip>                     rsync etx, fleet-mi300x, vLLM csrc, pytorch headers (no .git/.venv/build)
#   tools/session.sh preflight <ip>                check pushed tree, ROCm, dry run and Docker (no pull)
#   tools/session.sh postmortem <ip> <dest>          fetch and report failed/skipped jobs and runner minutes
#   tools/session.sh start  <ip> <session.py> [gpus] [machine]   start the runner (gpus default: all the VM has)
#   tools/session.sh status <ip>                     runner alive?, last 25 lines of its log
#   tools/session.sh fetch  <ip> <dest>              pull ~/sess (small files; *.bin traces too) into <dest>
set -uo pipefail
# ETX_SESS_USER overrides the login (Hot Aisle VMs use hotaisle; other providers may give root)
CMD=${1:?push|preflight|start|status|fetch|postmortem}; IP=${2:?ip}; H=${ETX_SESS_USER:-hotaisle}@$IP
SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $H"
case $CMD in
  push)
    ssh-keygen -R "$IP" >/dev/null 2>&1
    until $SSH true 2>/dev/null; do sleep 5; done
    cd ~/code || exit 1
    rsync -az --exclude build --exclude .venv --exclude .git --exclude __pycache__ --exclude 'results_*' etx/ $H:~/etx/ &&
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
    # Always inspect partial captures, but never hide a failed transfer.
    fetch_rc=0
    ETX_FETCH_REUSE=1 bash "$0" fetch "$IP" "$DEST" || fetch_rc=$?
    if [ "$fetch_rc" != 0 ]; then
      printf 'UNCAPTURED %s (fetch failed)\n' "$DEST"
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
    $SSH "$REMOTE_BASH" -s -- "$SESS" "${GPUS:-auto}" "$M" <<'REMOTE'
SESS=$1; G=$2; M=$3; [ "$G" = auto ] && G=""
cd ~/etx || exit 1
[ -n "$G" ] || G=$(rocm-smi --showproductname | grep -c "Card Series" | awk '{for (i = 0; i < $1; i++) printf "%s%d", (i ? "," : ""), i}')
mkdir -p ~/sess
# The Python runner reserves runner.log and selects a fresh directory on reuse.
# Keep startup errors too, without truncating any earlier session's output.
STARTUP=$(mktemp "$HOME/sess/startup.XXXXXX") || exit 1
setsid nohup python3 -m etx.tools.gpu_jobs "$SESS" --gpus "$G" --logdir ~/sess --machine "$M" > "$STARTUP" 2>&1 < /dev/null &
sleep 2; echo "runner started: machine $M, GPUs $G; startup log: $STARTUP"; head -5 "$STARTUP"
REMOTE
    ;;
  status)
    $SSH 'pgrep -f etx.tools.gpu_jobs >/dev/null && echo RUNNING || echo "runner not running"; find ~/sess -name runner.log -print -exec tail -25 {} \;' ;;
  fetch)
    DEST=${3:?dest}
    # a reused destination would mix another session's files into this one (review 08, R12)
    # (postmortem re-fetches over a partial capture on purpose: ETX_FETCH_REUSE=1; the verifier still flags any file
    # the box did not list)
    if [ "${ETX_FETCH_REUSE:-0}" != 1 ] && [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
      echo "refusing: $DEST is not empty; fetch into a new directory"; exit 2; fi
    mkdir -p "$DEST"
    # the provenance of an earlier capture goes before anything is transferred: if this fetch fails half-way, the
    # directory has no box manifest and cannot verify as this box's capture (review 09, R12) -- at any depth: the
    # runner nests a session-*/ directory when its log directory is taken (review 10, R12)
    find "$DEST" \( -name SHA256SUMS -o -name SHA256SUMS.local \) ! -type d -exec rm -f {} +
    # build records (plans, resources, generated kernels) are part of the result; only the binaries and patched
    # source copies stay on the box (review 07, R03). The box's SHA256SUMS is verified, never overwritten (R04).
    # an exact mirror of the box's session directory (minus the excluded): nothing from an earlier capture -- an old
    # SHA256SUMS included -- can survive a re-fetch into the same place (review 09, R12)
    rsync -az --delete --delete-excluded --exclude 'b_*/run' --exclude 'b_*/fleet_src/' --exclude '*.o' $H:~/sess/ "$DEST/" || exit 1
    ls "$DEST" | head -50
    if [ -f "$DEST/SHA256SUMS" ]; then
      python3 "$(dirname "$0")/../etx/tools/verify_sums.py" "$DEST" --excluded 'b_*/run' 'b_*/fleet_src/*' '*.o' || { echo "FETCH NOT VERIFIED: keep the box"; exit 2; }
    else
      echo "no box-side SHA256SUMS in $DEST: provenance missing (the session had no hash job)"
    fi
    # keep the fleet reference outputs (goldens, KV caches) for the next VM: no reference re-run (docs/DATA.md)
    mkdir -p ~/code/etx-cache/fleet_ref/build ~/code/etx-cache/fleet_ref/build_r1
    capture_rc=0
    rsync -az $H:'fleet-mi300x/build/golden*' $H:fleet-mi300x/build/fleet_cache.bin ~/code/etx-cache/fleet_ref/build/ || {
      echo "UNCAPTURED $H:fleet-mi300x/build/golden* or fleet_cache.bin"; capture_rc=1;
    }
    rsync -az $H:fleet-mi300x/build_r1/ ~/code/etx-cache/fleet_ref/build_r1/ || {
      echo "UNCAPTURED $H:fleet-mi300x/build_r1/"; capture_rc=1;
    }
    du -sh ~/code/etx-cache/fleet_ref
    exit "$capture_rc" ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
