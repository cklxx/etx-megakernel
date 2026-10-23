"""python -m examples.vllm_llm.bf16_300.launch: translator for etx/models/vllm_llm/bf16_300/launch.py."""
from pathlib import Path
import sys

from etx.external import HarnessCommandRequest, run_harness_command
# Compatibility: historical importers (examples/experiment_program providers) use these names from here.
from etx.models.vllm_llm.bf16_300.launch import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(run_harness_command(HarnessCommandRequest(
        "qwen-bf16-probe", "launch", tuple(sys.argv[1:]), str(Path(__file__).resolve().parents[3]))))
