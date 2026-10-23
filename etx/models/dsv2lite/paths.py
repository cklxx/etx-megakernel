"""The repository this harness runs in, and its declaration directory; set explicitly by the public API."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "examples/dsv2lite"


def configure(root):
    global ROOT, HERE
    ROOT = Path(root)
    HERE = ROOT / "examples/dsv2lite"
