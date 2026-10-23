"""The DeepSeek in-place session preflight refusing unmeasured inputs, moved from tests/test_dsv2_inplace_session.py."""
import hashlib
import os
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_preflight_refuses_inputs_that_are_not_the_measured_ones(tmp_path):
    """review 07, R09: a consistent but different input set (even with its own SHA256SUMS) must not pass preflight."""
    data = tmp_path / "data"
    data.mkdir()
    contents = {"weights.bin": b"w" * 64, "weights.bin.manifest": b"m", "fleet_cache.bin": b"c",
                "golden_tokens.txt": " ".join(["1"] * 33).encode(), "golden_hidden.bin": b"\0" * (27 * 2048 * 4)}
    for name, body in contents.items():
        (data / name).write_bytes(body)
    (data / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in contents.items()))
    env = dict(os.environ, ETX_SESS_LOGDIR=str(tmp_path / "s"), ETX_DSV2_FLEET=str(tmp_path / "f"), ETX_DSV2_DATA=str(data))
    run = run_command([sys.executable, "-m", "examples.dsv2lite.inplace_session", "preflight"], cwd=ROOT, env=env)
    assert run.returncode != 0 and "is not the measured one" in run.stdout, run.stdout[-2000:]
