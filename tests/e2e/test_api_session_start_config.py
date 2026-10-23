"""tools/session.sh start transporting the Qwen config, moved from tests/test_dsv2_batch.py, through external APIs only."""
import json
import os
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("config", [None, "", "/models/qwen config.json", "/models/qwen'$(touch $ETX_N6_MARKER);config.json"])
def test_session_start_transports_qwen_config_without_shell_expansion(tmp_path, config):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ssh = bindir / "ssh"
    ssh.write_text("#!/usr/bin/env python3\n"
                   "import os, subprocess, sys\n"
                   "sys.stdin.read()\n"
                   "env = dict(os.environ)\n"
                   "env.pop('ETX_QWEN_G6_CONFIG', None)\n"
                   "env.pop('ETX_SESS_MAX_SECONDS', None)\n"
                   "probe = \"python3 -c 'import json,os; print(json.dumps({k: os.environ.get(k) for k in "
                   "[\\\"ETX_QWEN_G6_CONFIG\\\", \\\"ETX_SESS_MAX_SECONDS\\\"]}))' > \\\"$TRANSPORT_CAPTURE\\\"\\n\"\n"
                   "raise SystemExit(subprocess.run(['bash', '-c', ' '.join(sys.argv[6:])], "
                   "input=probe, text=True, env=env).returncode)\n")
    ssh.chmod(0o755)
    capture = tmp_path / "transport.json"
    marker = tmp_path / "expanded"
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
               TRANSPORT_CAPTURE=str(capture), ETX_N6_MARKER=str(marker), ETX_SESS_MAX_SECONDS="1200",
               ETX_PYTHON=sys.executable)
    env.pop("ETX_QWEN_G6_CONFIG", None)
    if config is not None:
        env["ETX_QWEN_G6_CONFIG"] = config
    result = run_command(["bash", str(ROOT / "tools/session.sh"), "start", "offline", "session.py"],
                         cwd=ROOT, env=env, timeout=10)
    assert result.returncode == 0, result.stdout
    assert json.loads(capture.read_text()) == {"ETX_QWEN_G6_CONFIG": config, "ETX_SESS_MAX_SECONDS": "1200"}
    assert not marker.exists()
