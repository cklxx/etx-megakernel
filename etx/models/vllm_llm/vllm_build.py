"""Run the vLLM-kernel model build (etx/models/vllm_llm/build.sh) for one packed model directory.

  build.sh <packed-dir>     (VLLM, ETX_OUT, ETX_VL_*, ETX_HIPCC_FLAGS, ... exactly as before)
The build script inherits the caller's environment and terminal; only the repository root is explicit.
"""
import os
from pathlib import Path
import subprocess
import sys

from . import paths

SCRIPT = Path(__file__).resolve().with_name('build.sh')


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    completed = subprocess.run(['bash', str(SCRIPT), *args], env=dict(os.environ, ETX_VL_BUILD_ROOT=str(paths.ROOT)))
    return completed.returncode if completed.returncode >= 0 else 128 - completed.returncode
