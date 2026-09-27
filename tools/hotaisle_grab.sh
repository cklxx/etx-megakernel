#!/usr/bin/env bash
# Provision a Hot Aisle MI300X VM through the account TUI (ssh admin.hotaisle.app, driven in tmux), polling until one of
# the wanted types is listed.  Usage: tools/hotaisle_grab.sh [1x|2x|any] [poll seconds, default 120]
#   prints the success box (VM name, IP, ssh line) and exits 0.  Billing starts at once; the 2x type has a 1 h minimum.
#   Notes (memory: hotaisle-vm-billing-and-roles): dialogs answer to `y`, not Enter; a reused IP changes host key.
set -uo pipefail
WANT=${1:-any}; POLL=${2:-120}
case $WANT in 1x) PAT="1x MI300X VM";; 2x) PAT="2x MI300X VM";; any) PAT="[12]x MI300X VM";; *) echo "usage: $0 [1x|2x|any]"; exit 2;; esac
S=ha_grab
while true; do
  tmux kill-session -t $S 2>/dev/null
  tmux new-session -d -s $S -x 200 -y 50 'ssh admin.hotaisle.app'
  for i in $(seq 1 30); do tmux capture-pane -p -t $S | grep -q "Account Balance" && break; sleep 1; done
  tmux send-keys -t $S n
  for i in $(seq 1 20); do tmux capture-pane -p -t $S | grep -q "Available Resources\|pretty full" && break; sleep 1; done
  sleep 2; tmux send-keys -t $S '?'; sleep 1
  if tmux capture-pane -p -t $S | grep -qE "$PAT"; then
    [ "$WANT" = any ] && PAT="2x MI300X VM" && ! tmux capture-pane -p -t $S | grep -q "$PAT" && PAT="1x MI300X VM"   # prefer 2x (two GPUs)
    for i in $(seq 1 10); do
      tmux capture-pane -p -t $S | grep "│" | head -1 | grep -q "$PAT" && break
      tmux send-keys -t $S j; sleep 0.5
    done
    tmux send-keys -t $S Enter
    for i in $(seq 1 20); do tmux capture-pane -p -t $S | grep -q "Are you sure" && break; sleep 1; done
    tmux send-keys -t $S y
    for i in $(seq 1 90); do tmux capture-pane -p -t $S | grep -q "ssh hotaisle@" && break; sleep 1; done
    date -u; tmux capture-pane -p -t $S | grep -E "VM Name|IP Address|ssh hotaisle@|rror"
    tmux send-keys -t $S Escape; sleep 1; tmux kill-session -t $S
    exit 0
  fi
  tmux kill-session -t $S
  echo "$(date -u +%H:%M:%S) no $WANT VM available; next look in ${POLL}s"
  sleep "$POLL"
done
