"""python -m examples.dsv2lite.pregpu_report: translator for etx/models/dsv2lite/pregpu_report.py."""
from pathlib import Path
import sys

from etx.external import HarnessCommandRequest, run_harness_command


if __name__ == "__main__":
    raise SystemExit(run_harness_command(HarnessCommandRequest(
        "deepseek-tools", "pregpu_report", tuple(sys.argv[1:]), str(Path(__file__).resolve().parents[2]))))
