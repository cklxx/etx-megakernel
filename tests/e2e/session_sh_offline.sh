#!/bin/sh
# Offline regression checks; optional argument permits mutation checks on a copy.
set -eu
SCRIPT=${1:-tools/session.sh}
# Source checks read the transport implementation; behaviour checks drive the caller-facing script.
IMPL=${2:-etx/session/transport.sh}
bash -n "$SCRIPT"
bash -n "$IMPL"
grep -F 'rsync -azL rocm-headers/inc $H:~/rocm-headers/ &&' "$IMPL" >/dev/null
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT HUP INT TERM
mkdir "$TMP/bin"
cat > "$TMP/bin/ssh" <<'MOCK'
#!/bin/sh
set -eu
shift 5 # -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new user@host
printf '%s\n' "$*" > "$COMMAND_LOG"
cat >/dev/null
# Discard the real runner script; test the remote command with a harmless probe.
unset ETX_SESS_MAX_SECONDS
printf '%s\n' 'printf "%s" "${ETX_SESS_MAX_SECONDS-unset}"' | sh -c "$*" > "$CAP_LOG"
MOCK
chmod +x "$TMP/bin/ssh"
PATH="$TMP/bin:$PATH"
COMMAND_LOG="$TMP/command" CAP_LOG="$TMP/cap"
export PATH COMMAND_LOG CAP_LOG
unset ETX_SESS_MAX_SECONDS
bash "$SCRIPT" start offline session.py > "$TMP/output"
[ "$(cat "$CAP_LOG")" = unset ]
[ "$(cat "$COMMAND_LOG")" = 'bash -s -- session.py auto A' ]
[ ! -s "$TMP/output" ]
for cap in 1200 '' "quote'\$(false); literal"; do
  ETX_SESS_MAX_SECONDS=$cap
  export ETX_SESS_MAX_SECONDS
  bash "$SCRIPT" start offline session.py > "$TMP/output"
  [ "$(cat "$CAP_LOG")" = "$cap" ]
  grep -F 'ETX_SESS_MAX_SECONDS=' "$COMMAND_LOG" >/dev/null
  grep -F 'Passing session cap: ETX_SESS_MAX_SECONDS=' "$TMP/output" >/dev/null
done
unset ETX_SESS_MAX_SECONDS
# The two-GPU session inputs reach the box, quoted for spaces and apostrophes.
ETX_QWEN_BATCH_MODEL="/models/qwen 3" ETX_GLM_STAGE="/staged/glm'53" \
ETX_GLM_REFERENCE=/staged/glm53-reference.json ETX_GLM_BATCH=2 ETX_GLM_GATE=module \
  bash "$SCRIPT" start offline session.py > "$TMP/output"
[ "$(cat "$CAP_LOG")" = unset ]
cat > "$TMP/expected" <<'EOF'
ETX_GLM_GATE='module' ETX_GLM_BATCH='2' ETX_GLM_REFERENCE='/staged/glm53-reference.json' ETX_GLM_STAGE='/staged/glm'\''53' ETX_QWEN_BATCH_MODEL='/models/qwen 3' bash -s -- session.py auto A
EOF
cmp "$TMP/expected" "$COMMAND_LOG"
grep -F 'Passing ETX_GLM_BATCH=2' "$TMP/output" >/dev/null
# Preflight dry-runs the two sessions it packages, not vlbest.
grep -F 'python3 -m etx.tools.gpu_jobs examples/sessions/qwen_batch_correctness.py --gpus 0 --dry-run' "$IMPL" >/dev/null
grep -F 'python3 -m etx.tools.gpu_jobs examples/sessions/glm53_model.py --gpus 0,1 --dry-run' "$IMPL" >/dev/null
! grep -F 'examples.sessions.vlbest --dry-run' "$IMPL" >/dev/null
# status runs its remote command under `sh -c` like sshd: it must not count that shell (or awk)
# as the runner, and must see a real `python3 -m etx.tools.gpu_jobs` process.
mkdir -p "$TMP/home/sess" "$TMP/fake/etx/tools"
: > "$TMP/fake/etx/__init__.py"; : > "$TMP/fake/etx/tools/__init__.py"
printf 'import time\ntime.sleep(60)\n' > "$TMP/fake/etx/tools/gpu_jobs.py"
HOME="$TMP/home" bash "$SCRIPT" status offline < /dev/null > /dev/null
[ "$(head -1 "$CAP_LOG")" = 'runner not running' ]
(cd "$TMP/fake" && exec python3 -m etx.tools.gpu_jobs) & RUNNER=$!
trap 'kill "$RUNNER" 2>/dev/null || true; rm -rf "$TMP"' EXIT HUP INT TERM
sleep 1
HOME="$TMP/home" bash "$SCRIPT" status offline < /dev/null > /dev/null
[ "$(head -1 "$CAP_LOG")" = RUNNING ]
kill "$RUNNER"; wait "$RUNNER" 2>/dev/null || true
HOME="$TMP/home" bash "$SCRIPT" status offline < /dev/null > /dev/null
[ "$(head -1 "$CAP_LOG")" = 'runner not running' ]
printf '%s\n' 'session.sh offline checks passed'
