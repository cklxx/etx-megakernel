"""The byte-ledger command (examples/vllm_llm/byte_ledger.py) through the external API."""
from pathlib import Path
import sys

from etx.testing import packed_manifest_text, run_command

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/llm/configs"


def ledger(*argv):
    return run_command([sys.executable, "-m", "examples.vllm_llm.byte_ledger", *map(str, argv)], cwd=ROOT)


def test_cli_prints_breakdown_total_and_floor():
    run = ledger("--config", CONFIGS / "phi-3.5-moe.json", "--batch", "1", "--context", "1024")
    assert run.returncode == 0
    out = run.stdout
    assert "not a packed manifest" in out
    assert out.count("\n   31 ") == 1 and "TOTAL 13.1537 GB" in out and "FLOOR 3.059 ms/token" in out


def test_packed_manifest_file(tmp_path):
    path = tmp_path / "vl.manifest"
    path.write_text(packed_manifest_text(CONFIGS / "phi-3.5-moe.json") + "cfgf eps 1e-05\n")
    run = ledger("--manifest", path, "--batch", "1", "--context", "1024")
    assert run.returncode == 0 and "(packed manifest)" in run.stdout


def test_cli_refusal_names_input_and_prints_no_floor(tmp_path):
    run = ledger("--manifest", tmp_path / "absent", "--batch", "1", "--context", "1024")
    assert run.returncode == 2
    assert "REFUSED: missing input: manifest" in run.stdout and "FLOOR" not in run.stdout
    path = tmp_path / "vl.manifest"
    path.write_text("\n".join(ln for ln in packed_manifest_text(CONFIGS / "phi-3.5-moe.json").splitlines()
                              if not ln.startswith("cfg NKV ")))
    run = ledger("--manifest", path, "--batch", "1", "--context", "1024")
    assert run.returncode == 2
    assert "REFUSED: missing input: cfg NKV" in run.stdout and "FLOOR" not in run.stdout
