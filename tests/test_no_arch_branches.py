"""Hard decision 1: hardware differences exist only as data.

No pass, IR module, simulator or emitter may test an architecture name.  The
only place an arch name may appear in code is the machine loader (file names).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "etx"
PATTERN = re.compile(r"\b(gfx9\d\d|gfx1\d\d\d|sm_\d+|CDNA\d|Hopper|Blackwell|MI300|MI250|MI350|H100|B200)\b")
CODE_DIRS = ["ir", "passes", "sim", "codegen", "frontends"]


def test_no_architecture_names_in_code():
    hits = []
    for d in CODE_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                if '"""' in line or line.strip().startswith(("#", "*", "//")):
                    continue
                if PATTERN.search(code):
                    hits.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not hits, "architecture names in code paths (must live in machine/arch/*.yaml):\n" + "\n".join(hits)
