#!/bin/sh
# Offline regression checks; optional argument permits mutation checks on a copy.
set -eu
SCRIPT=${1:-tools/session.sh}
bash -n "$SCRIPT"
grep -F 'rsync -azL rocm-headers/inc $H:~/rocm-headers/ &&' "$SCRIPT" >/dev/null
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
ETX_GLM_REFERENCE=/staged/glm53-reference.json ETX_GLM_BATCH=2 \
  bash "$SCRIPT" start offline session.py > "$TMP/output"
[ "$(cat "$CAP_LOG")" = unset ]
cat > "$TMP/expected" <<'EOF'
ETX_GLM_BATCH='2' ETX_GLM_REFERENCE='/staged/glm53-reference.json' ETX_GLM_STAGE='/staged/glm'\''53' ETX_QWEN_BATCH_MODEL='/models/qwen 3' bash -s -- session.py auto A
EOF
cmp "$TMP/expected" "$COMMAND_LOG"
grep -F 'Passing ETX_GLM_BATCH=2' "$TMP/output" >/dev/null
printf '%s\n' 'session.sh offline checks passed'
