"""The repository this harness runs in, and its declaration directory; set explicitly by the public API."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "examples/vllm_llm/expert_merge"


def configure(root):
    global ROOT, HERE
    ROOT = Path(root)
    HERE = ROOT / "examples/vllm_llm/expert_merge"
