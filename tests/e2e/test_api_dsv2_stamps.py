"""Command-level cases moved from tests/test_dsv2_stamps.py, through external APIs only."""
import json
import os
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
_LAST = []


def last_output():
    return _LAST[-1] if _LAST else ""


def test_stamps2_rejects_isolate_before_build(tmp_path):
    env = {**os.environ, "ETX_DSV2_ISOLATE": "0:qkv_5", "ETX_OUT": str(tmp_path)}
    result = run_command(["bash", "examples/dsv2lite/build.sh", "/nonexistent-pinned-export", "-DETX_DSV2_STAMPS2"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "share prologue slots" in result.stderr
