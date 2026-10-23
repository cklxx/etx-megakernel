"""Keep the optional vLLM dependency behind the baseline boundary."""
import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = Path("etx/bench/vllm_baseline.py")


def test_legacy_benchmark_import_shares_main_globals(monkeypatch):
    from examples.llm import vllm_bench
    from etx.bench import vllm_baseline

    assert vllm_bench is vllm_baseline
    sentinel = object()
    monkeypatch.setattr(vllm_bench, "read_golden", sentinel)
    assert vllm_bench.main.__globals__["read_golden"] is sentinel


def vllm_imports(source):
    """Find direct vLLM imports, including aliases and nested imports."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        if any(name == "vllm" or name.startswith("vllm.") for name in names):
            yield node.lineno


def test_no_vllm_imports():
    violations = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in sorted((ROOT / "etx").rglob("*.py"))
        if path.relative_to(ROOT) != ALLOWED
        for line in vllm_imports(path.read_text(encoding="utf-8"))
    ]
    assert not violations, (
        "vLLM imports belong only in etx/bench/vllm_baseline.py:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize("source", [
    "import vllm",
    "import os, vllm as backend",
    "import vllm.inputs as inputs",
    "from vllm import LLM",
    "from vllm.inputs import TokensPrompt",
    "def run():\n    import vllm",
    "if False:\n    from vllm import LLM",
])
def test_seeded_import_is_detected(source):
    assert list(vllm_imports(source))


@pytest.mark.parametrize("source", [
    '# import vllm\ntext = "from vllm import LLM"',
    "import vllm_helpers",
    "from etx.importer import vllm_kernels",
    "from .vllm import local_helper",
])
def test_unrelated_names_are_allowed(source):
    assert not list(vllm_imports(source))


def test_seeded_violation_fails_boundary(tmp_path, monkeypatch):
    """Exercise the actual repository gate against an isolated seeded tree."""
    baseline = tmp_path / ALLOWED
    baseline.parent.mkdir(parents=True)
    baseline.write_text("import vllm\n")
    monkeypatch.setitem(test_no_vllm_imports.__globals__, "ROOT", tmp_path)
    test_no_vllm_imports()  # The sole allow-listed module is accepted.
    forbidden = tmp_path / "etx" / "seed.py"
    forbidden.write_text("def run():\n    from vllm.inputs import TokensPrompt\n")
    with pytest.raises(AssertionError, match=r"etx/seed\.py:2"):
        test_no_vllm_imports()
