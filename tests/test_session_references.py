"""Check session entry points without running job commands or importing GPU code."""
import re
import shlex
import sys
from pathlib import Path

import pytest

from etx.tools.gpu_jobs import load_jobs


ROOT = Path(__file__).resolve().parents[1]
SESSIONS = sorted((ROOT / "examples/sessions").rglob("*.py"))


def local_python_references(command, cwd="~/etx"):
    """Track the explicit cd commands used to return from fleet to ETX."""
    local_dirs = {"~/etx", "$HOME/etx", ".", str(ROOT)}
    for parts in _shell_commands(command):
        if parts[0] == "cd" and len(parts) == 2:
            cwd = parts[1]
        elif cwd in local_dirs:
            yield from _references(parts)
        # Commands in fleet/VM directories refer to another checkout; skip
        # until an explicit `cd ~/etx` brings subsequent commands back here.


def _shell_commands(command):
    """Split shell lists without interpreting quoted strings as commands."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    parts = []
    for token in lexer:
        if token and all(c in ";&|()\n" for c in token):
            if parts:
                yield parts
            parts = []
        else:
            parts.append(token)
    if parts:
        yield parts


def _references(parts):
    while parts and (parts[0] in {"{", "}", "then", "do", "!"}
                     or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", parts[0], re.S)):
        parts = parts[1:]
    if not parts or not re.fullmatch(r"(?:.*/)?python(?:3(?:\.\d+)?)?", parts[0]):
        # Docker, shell built-ins, ssh, and other executables do not invoke
        # Python in this checkout; quoted Python text inside them is not a job.
        return
    args = parts[1:]
    while args and args[0] in {"-u", "-B", "-E", "-s", "-S", "-I"}:
        args = args[1:]
    if not args or args[0] in {"-c", "-"}:
        # Inline code and stdin/heredocs have no file entry point to resolve.
        return
    if args[0] == "-m":
        assert len(args) > 1, "Python -m is missing a module name"
        module = args[1]
        if module.split(".")[0] in sys.stdlib_module_names:
            # Environment setup uses `python3 -m venv`, supplied by Python.
            return
        yield Path(module.replace(".", "/") + ".py")
    else:
        script = args[0]
        if script.startswith(("/", "~", "$")) or ":" in script:
            # Absolute/expanded/remote paths belong to the VM, not this repo.
            return
        assert not script.startswith("-"), f"Unhandled Python option: {script}"
        yield Path(script)


@pytest.mark.parametrize("session", SESSIONS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_session_python_references_exist(session):
    missing = []
    for job in load_jobs(session):
        for reference in local_python_references(job.cmd, job.cwd):
            if not (ROOT / reference).is_file():
                missing.append(f"{job.name}: {reference}")
    assert not missing, "Missing repository Python entry points:\n" + "\n".join(missing)


@pytest.mark.parametrize("command, expected", [
    ("python3 -m examples.vllm_llm.nvref_report x", ["examples/vllm_llm/nvref_report.py"]),
    ("PYTHONPATH=. .venv/bin/python -u -m etx.tools.timeline x", ["etx/tools/timeline.py"]),
    ("python3 examples/llm/vllm_bench.py", ["examples/llm/vllm_bench.py"]),
    ("~/llmenv/bin/python examples/vllm_llm/vendor_gemv.py", ["examples/vllm_llm/vendor_gemv.py"]),
    ("true && { .venv/bin/python -m missing.first; python3 examples/missing.py; }",
     ["missing/first.py", "examples/missing.py"]),
    ("echo 'python3 -m missing'; sudo docker run image python3 examples/missing.py", []),
    ("ssh host 'python3 -m missing'; python3 ~/remote/missing.py", []),
    ("python3 -m venv .venv; python3 -c 'print(1)'; python3 - <<'PY'\nprint(1)\nPY", []),
])
def test_local_python_reference_forms(command, expected):
    assert list(local_python_references(command)) == [Path(p) for p in expected]


def test_return_from_external_checkout():
    command = "python3 src/host/reference_run.py; cd ~/etx && python3 -m examples.dsv2lite.isolate_report"
    assert list(local_python_references(command, "~/fleet-mi300x")) == [
        Path("examples/dsv2lite/isolate_report.py")
    ]
