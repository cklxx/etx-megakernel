"""The planner and code generator take configuration from PassOptions only, never from the process environment."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('etx/ir', 'etx/passes', 'etx/codegen')


def test_planner_and_codegen_never_read_the_environment():
    reads = []
    for package in PACKAGES:
        for path in sorted((ROOT / package).rglob('*.py')):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Attribute) and node.attr in ('environ', 'getenv', 'environb'):
                    reads.append(f'{path.relative_to(ROOT)}:{node.lineno}')
    assert reads == []
