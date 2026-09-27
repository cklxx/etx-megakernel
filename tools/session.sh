#!/usr/bin/env bash
# One GPU session on one or two Hot Aisle VMs, driven from the laptop.  Each VM runs its own etx.tools.gpu_jobs runner
# (jobs tagged machine="A" / "B" in the session file), fully detached from ssh, logging to ~/sess.
#
#   tools/session.sh push   <ip>                     rsync etx, fleet-mi300x, vLLM csrc, pytorch headers (no .git/.venv/build)
#   tools/session.sh start  <ip> <session.py> [gpus] [machine]   start the runner (gpus default: all the VM has)
#   tools/session.sh status <ip>                     runner alive?, last 25 lines of its log
#   tools/session.sh fetch  <ip> <dest>              pull ~/sess (small files; *.bin traces too) into <dest>
set -uo pipefail
CMD=${1:?push|start|status|fetch}; IP=${2:?ip}; H=hotaisle@$IP
SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $H"
case $CMD in
  push)
    ssh-keygen -R "$IP" >/dev/null 2>&1
    until $SSH true 2>/dev/null; do sleep 5; done
    cd ~/code || exit 1
    rsync -az --exclude build --exclude .venv --exclude .git --exclude __pycache__ --exclude 'results_*' etx/ $H:~/etx/ &&
    rsync -az --exclude .git --exclude .venv --exclude build --exclude __pycache__ fleet-mi300x/ $H:~/fleet-mi300x/ &&
    rsync -az vllm/csrc $H:~/vllm/ && rsync -az --exclude .git rocm-headers/pytorch $H:~/rocm-headers/ &&
    { [ ! -d ~/code/etx-cache/fleet_ref ] || rsync -az ~/code/etx-cache/fleet_ref/ $H:~/fleet-mi300x/; } &&   # kept reference outputs
    $SSH 'rocm-smi --showproductname | grep -c "Card Series" | sed "s/^/GPUs: /"; nproc | sed "s/^/cores: /"; df -h ~ | tail -1' ;;
  start)
    SESS=${3:?session file (path inside the etx repo)}; GPUS=${4:-}; M=${5:-A}
    $SSH bash -s -- "$SESS" "${GPUS:-auto}" "$M" <<'REMOTE'
SESS=$1; G=$2; M=$3; [ "$G" = auto ] && G=""
cd ~/etx || exit 1
[ -n "$G" ] || G=$(rocm-smi --showproductname | grep -c "Card Series" | awk '{for (i = 0; i < $1; i++) printf "%s%d", (i ? "," : ""), i}')
mkdir -p ~/sess
setsid nohup python3 -m etx.tools.gpu_jobs "$SESS" --gpus "$G" --logdir ~/sess --machine "$M" > ~/sess/runner.log 2>&1 < /dev/null &
sleep 2; echo "runner started: machine $M, GPUs $G"; head -5 ~/sess/runner.log
REMOTE
    ;;
  status)
    $SSH 'pgrep -f etx.tools.gpu_jobs >/dev/null && echo RUNNING || echo "runner not running"; tail -25 ~/sess/runner.log' ;;
  fetch)
    DEST=${3:?dest}; mkdir -p "$DEST"
    rsync -az --exclude 'b_*' $H:~/sess/ "$DEST/" && ls "$DEST" | head -50
    # keep the fleet reference outputs (goldens, KV caches) for the next VM: no reference re-run (docs/DATA.md)
    mkdir -p ~/code/etx-cache/fleet_ref/build ~/code/etx-cache/fleet_ref/build_r1
    rsync -az $H:'fleet-mi300x/build/golden*' $H:fleet-mi300x/build/fleet_cache.bin ~/code/etx-cache/fleet_ref/build/ 2>/dev/null
    rsync -az $H:fleet-mi300x/build_r1/ ~/code/etx-cache/fleet_ref/build_r1/ 2>/dev/null; du -sh ~/code/etx-cache/fleet_ref ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
