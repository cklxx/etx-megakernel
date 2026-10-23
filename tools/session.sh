#!/usr/bin/env bash
# One GPU session on one or two Hot Aisle VMs, driven from the laptop.  Each VM runs its own etx.tools.gpu_jobs runner
# (jobs tagged machine="A" / "B" in the session file), fully detached from ssh, logging to ~/sess.
#
#   tools/session.sh stage  <ip>                     this repository's HEAD, as an archive, in a NEW box directory; ~/etx -> it
#   tools/session.sh push   <ip>                     stage, then rsync fleet-mi300x, vLLM csrc, pytorch headers
#   tools/session.sh preflight <ip>                check pushed tree, ROCm, Qwen batch + GLM dry runs, Docker (no pull)
#   tools/session.sh postmortem <ip> <dest>          fetch and report failed/skipped jobs and runner minutes
#   tools/session.sh start  <ip> <session.py> [gpus] [machine]   start the runner (gpus default: all the VM has)
#   tools/session.sh status <ip>                     runner alive?, last 25 lines of its log
#   tools/session.sh fetch  <ip> <dest> [id]         copy the box's newest (or named) session record into <dest>/<id>
#                                                    and verify it against its own SEAL (etx/session/record.py)
#   ETX_SESS_USER overrides the login (Hot Aisle VMs use hotaisle; other providers may give root)
#
# Implementation: etx/session/transport.sh behind etx.external.run_session_transport. Interpreter: ETX_PYTHON,
# default this repository's .venv (the system python3 may be too old for etx).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${ETX_PYTHON:-$ROOT/.venv/bin/python}" -m etx.tools.session --root "$ROOT" "$@"
