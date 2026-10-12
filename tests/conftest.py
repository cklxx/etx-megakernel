import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(scope="session")
def pinned_src():
    """fleet's src at 3fb67a6 (git archive, hash-checked; never its working tree)."""
    from examples.dsv2lite.pinned import fleet_export
    try:
        return fleet_export() / "src"
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
