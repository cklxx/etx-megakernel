"""python -m examples.vllm_llm.folds: translator for etx/models/vllm_llm/folds.py."""
from pathlib import Path
import sys

from etx.external import HarnessCommandRequest, run_harness_command


if __name__ == "__main__":
    raise SystemExit(run_harness_command(HarnessCommandRequest(
        "qwen-vllm", "folds", tuple(sys.argv[1:]), str(Path(__file__).resolve().parents[2]))))
