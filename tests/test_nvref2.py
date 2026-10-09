"""The follow-up changes only the primary vLLM estimator's repeat count."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nvref2_only_primary_vllm_repeats_change():
    old_path = ROOT / "examples/sessions/nvref.py"
    new_path = ROOT / "examples/sessions/nvref2.py"
    old, new = old_path.read_bytes(), new_path.read_bytes()
    # Even comments, deadlines and the optional AITER arm must stay unchanged.
    assert old.count(b"--repeats 1") == 1
    assert new.count(b"--repeats {1 if aiter else 5}") == 1
    assert new.replace(b"--repeats {1 if aiter else 5}", b"--repeats 1") == old

    job_lists = []
    for source, path in ((old, old_path), (new, new_path)):
        namespace = {"__file__": str(path), "__name__": path.stem}
        # Parse the bytes themselves; evaluation only builds command strings.
        exec(compile(ast.parse(source, filename=str(path)), str(path), "exec"), namespace)
        job_lists.append(namespace["JOBS"])
    previous, following = job_lists
    primary = {f"vllm_r{rnd}" for rnd in range(1, 6)}
    assert {j["name"] for j in following if j["name"] in primary} == primary
    expected_commands = [
        j["cmd"].replace("--repeats 1", "--repeats 5")
        if j["name"] in primary else j["cmd"] for j in previous
    ]
    assert [j["cmd"] for j in following] == expected_commands
    assert [{k: v for k, v in j.items() if k != "cmd"} for j in following] == [
        {k: v for k, v in j.items() if k != "cmd"} for j in previous
    ]
