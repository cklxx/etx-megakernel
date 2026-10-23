"""python -m examples.vllm_llm.attention_early_merge.simulate: translator for etx/models/vllm_llm/attention_early_merge/simulate.py."""
from pathlib import Path
import sys

from etx.external import HarnessCommandRequest, run_harness_command


if __name__ == "__main__":
    raise SystemExit(run_harness_command(HarnessCommandRequest(
        "qwen-attention-early-merge", "simulate", tuple(sys.argv[1:]), str(Path(__file__).resolve().parents[3]))))
