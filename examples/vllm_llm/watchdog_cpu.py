"""python -m examples.vllm_llm.watchdog_cpu: translator for etx/models/vllm_llm/watchdog_cpu.py."""
from pathlib import Path
import sys

from etx.external import HarnessCommandRequest, run_harness_command


if __name__ == "__main__":
    raise SystemExit(run_harness_command(HarnessCommandRequest(
        "qwen-vllm", "watchdog_cpu", tuple(sys.argv[1:]), str(Path(__file__).resolve().parents[2]))))
