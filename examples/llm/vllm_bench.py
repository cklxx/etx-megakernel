#!/usr/bin/env python3
"""Compatibility CLI for :mod:`etx.bench.vllm_baseline`."""
from pathlib import Path
import sys

# Direct script execution puts examples/llm, rather than the repository, on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from etx.bench.vllm_baseline import *  # noqa: F401,F403 -- legacy helper imports


if __name__ == "__main__":
    main()
