"""The follow-up changes only the primary vLLM estimator's repeat count."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def without_module_docstring(source):
    module = ast.parse(source)
    if not module.body:
        return source
    first = module.body[0]
    if not (isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return source
    lines = source.splitlines(keepends=True)
    # AST columns are byte offsets; retain everything outside the string,
    # including comments or statements on its opening/closing lines.
    start = sum(map(len, lines[:first.lineno - 1])) + first.col_offset
    end = sum(map(len, lines[:first.end_lineno - 1])) + first.end_col_offset
    return source[:start] + source[end:]


def test_nvref2_only_primary_vllm_repeats_change():
    old_path = ROOT / "examples/sessions/nvref.py"
    new_path = ROOT / "examples/sessions/nvref2.py"
    old, new = old_path.read_bytes(), new_path.read_bytes()
    # Keep the original byte-for-byte comparison on the complete existing prefix.
    prefix, separator, tail = new.partition(b"\n# Optional model diagnostics")
    assert separator and tail
    new = prefix
    # Even comments, deadlines and the optional AITER arm must stay unchanged.
    assert old.count(b"--repeats 1") == 1
    assert new.count(b"--repeats {1 if aiter else 5}") == 1

    job_lists = []
    metadata = []
    for source, path in ((old, old_path), (new, new_path)):
        namespace = {"__file__": str(path), "__name__": path.stem}
        # Parse the bytes themselves; evaluation only builds command strings.
        exec(compile(ast.parse(source, filename=str(path)), str(path), "exec"), namespace)
        job_lists.append(namespace["JOBS"])
        # GPU assignments and job names are checked in the job parameters below.
        metadata.append({name: namespace[name] for name in (
            "LLM", "PY", "V", "RES", "M", "HF", "S", "IMAGE", "ALL", "ENV", "DEPS",
        )})
    assert metadata[1] == metadata[0]
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

    # Only module documentation may differ beyond the repeat token: even job
    # comments, deadlines and the optional AITER arm must stay byte-identical.
    assert without_module_docstring(new).replace(
        b"--repeats {1 if aiter else 5}", b"--repeats 1",
    ) == without_module_docstring(old)
    # Evaluate the full follow-up too: every pre-existing job, including report,
    # must remain identical to the prefix checked above. No GPU command is run.
    namespace = {"__file__": str(new_path), "__name__": new_path.stem}
    exec(compile(ast.parse(new_path.read_bytes()), str(new_path), "exec"), namespace)
    all_jobs = namespace["JOBS"]
    assert all_jobs[:-3] == following
    assert following[-1]["name"] == "report"
    diagnostics = all_jobs[-3:]
    assert [j["name"] for j in diagnostics] == ["trace1", "floors_trace", "evx"]
    for j, predecessor, lane in zip(
        diagnostics, ("report", "trace1", "floors_trace"), ("gpu", "cpu", "gpu")
    ):
        assert j["reads"] == predecessor
        assert j["timeout"] == 300
        assert j.get("lane", "gpu") == lane
        if lane == "gpu":
            assert j["gpu"] == 0
