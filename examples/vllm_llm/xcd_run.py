"""python -m examples.vllm_llm.xcd_run: translator for etx/models/vllm_llm/xcd_run.py."""
from pathlib import Path
import sys

from etx.external import HarnessCommandRequest, run_harness_command


if __name__ == "__main__":
    raise SystemExit(run_harness_command(HarnessCommandRequest(
        "qwen-vllm", "xcd_run", tuple(sys.argv[1:]), str(Path(__file__).resolve().parents[2]))))
