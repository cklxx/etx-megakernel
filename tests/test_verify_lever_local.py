"""verify_lever.py --compiler (docker | 7.2.4 | 7.14) and tools/rocm_build_local.sh: CPU only, no compiler needed."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from examples.dsv2lite import verify_lever as vl

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/rocm_build_local.sh"
FLEET = Path("/nonexistent/fleet")


def test_build_cmd_docker_is_unchanged():
    assert vl.build_cmd("topk_rank", Path("/o"), "docker", fleet=FLEET) == [
        "bash", "examples/dsv2lite/rocm_build_check.sh", str(FLEET), "topk_rank", "/o"]
    assert vl.build_cmd("d4", Path("/o"), "docker", registered=True, fleet=FLEET)[3:5] == ["--arm", "d4"]


@pytest.mark.parametrize("tc", ["7.2.4", "7.14"])
def test_build_cmd_local(tc):
    assert vl.build_cmd("topk_rank", Path("/o"), tc, fleet=FLEET) == ["bash", vl.LOCAL_SCRIPT, tc, str(FLEET), "topk_rank", "/o"]
    assert vl.build_cmd("alld4s", Path("/o"), tc, registered=True, fleet=FLEET) == [
        "bash", vl.LOCAL_SCRIPT, tc, str(FLEET), "--arm", "alld4s", "/o"]


def test_unknown_compiler_is_refused():
    with pytest.raises(ValueError, match="compiler"):
        vl.build_cmd("default", Path("/o"), "7.0", fleet=FLEET)


def test_compiler_record_local_and_docker(tmp_path):
    d = tmp_path / "default"
    d.mkdir()
    local = {"mode": "local", "toolchain": "7.14", "compiler_path": "/x/clang++", "rocm_root": "/x",
             "clang_version": "AMD clang version 23", "clang_commit": "a" * 40}
    (d / "compiler.json").write_text(json.dumps(local))
    assert vl.compiler_record(d, "7.14") == local
    (d / "compiler.txt").write_text("AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project "
                                    "roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)\nTarget: x\n")
    (tmp_path / "image.txt").write_text("sha256:abc\n")
    rec = vl.compiler_record(d, "docker")
    assert rec["mode"] == "docker" and rec["toolchain"] == "7.2.4" and rec["image"] == "sha256:abc"
    assert rec["clang_commit"] == "f58b06dce1f9c15707c5f808fd002e18c2accf7e"
    assert set(rec) >= {"mode", "toolchain", "compiler_path", "clang_version", "clang_commit"}


def test_local_script_syntax_and_refusals(tmp_path):
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    fleet = tmp_path / "fleet"
    (fleet / "src").mkdir(parents=True)
    bad = subprocess.run(["bash", str(SCRIPT), "6.0", str(fleet), "default", str(tmp_path / "o")], capture_output=True, text=True)
    assert bad.returncode == 2 and "unknown toolchain" in bad.stderr
    missing = subprocess.run(["bash", str(SCRIPT), "7.2.4", str(fleet), "default", str(tmp_path / "o")], capture_output=True,
                             text=True, env={"PATH": "/usr/bin:/bin", "ETX_ROCM_724": str(tmp_path / "none")})
    assert missing.returncode == 2 and "incomplete" in missing.stderr
    (fleet / ".git").mkdir()
    co = subprocess.run(["bash", str(SCRIPT), "7.2.4", str(fleet), "default"], capture_output=True, text=True)
    assert co.returncode == 2 and "checkout" in co.stderr


def test_local_script_writes_the_compiler_record_fields():
    text = SCRIPT.read_text()
    for field in ('"mode": "local"', '"toolchain"', '"compiler_path"', '"clang_version"', '"clang_commit"', '"rocm_root"'):
        assert field in text
    assert "docker" not in text.split("set -euo pipefail", 1)[1]


def test_cli_needs_exactly_one_arm(capsys):
    with pytest.raises(SystemExit):
        vl.main(["default", "--registered", "d4"])
    with pytest.raises(SystemExit):
        vl.main([])


def test_committed_d4_records_have_the_report_fields():
    for name in ("d4", "d4s"):
        rec = json.loads((ROOT / f"examples/dsv2lite/results/levers/{name}.json").read_text())
        assert {"resources", "static"} <= set(rec["arm_build"]) and "summary" in rec["cpu_gates"]
        assert rec["compiler"]["mode"] in ("docker", "local") and rec["compiler"]["clang_commit"]
        assert rec["verdict"] == "PASS" and "prior_record" in rec
