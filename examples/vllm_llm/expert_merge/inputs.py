"""python -m examples.vllm_llm.expert_merge.inputs: translator for etx/models/vllm_llm/expert_merge/inputs.py."""
from pathlib import Path
import sys

from etx.external import HarnessCommandRequest, run_harness_command


if __name__ == "__main__":
    raise SystemExit(run_harness_command(HarnessCommandRequest(
        "qwen-expert-merge", "inputs", tuple(sys.argv[1:]), str(Path(__file__).resolve().parents[3]))))
