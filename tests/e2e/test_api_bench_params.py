"""Command-level cases moved from tests/test_bench_params.py, through external APIs only."""
import json
import os
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_LAST = []


def last_output():
    return _LAST[-1] if _LAST else ""


def c_formats(path):
    """the RESULT printf format strings of a .hip file, adjacent literals joined"""
    src = (ROOT / path).read_text()
    fmts = []
    for m in re.finditer(r'printf\(((?:\s*"(?:[^"\\]|\\.)*"\s*)+)', src):
        s = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)))
        if s.startswith("RESULT") and "HUNG" not in s:
            fmts.append(re.sub(r"%(-?\d*)lld", r"%\1d", s.replace("\\n", "")))
    return fmts


def fmt(fmts, prefix, *args):
    f = next(x for x in fmts if x.startswith(prefix))
    return f % args


def sample():
    hw, imp = c_formats("bench/hw/hw_bench.hip"), c_formats("bench/hw/imp_bench.hip")
    L = [
        fmt(hw, "RESULT lat", "vector", "MALL", "-      ", "idle  ", 812.3, 1600, 800.1),
        fmt(hw, "RESULT lat", "scalar", "MALL", "-      ", "idle  ", 700.0, 1400, 690.0),
        fmt(hw, "RESULT lat", "vector", "L2", "inv sc1", "idle  ", 900.0, 1800, 880.0),
        fmt(hw, "RESULT lat", "scalar", "L1/K$", "inv sc1", "loaded", 150.0, 300, 140.0),
        fmt(hw, "RESULT lat", "scalar", "L1/K$", "inv sc1", "idle  ", 120.0, 240, 110.0),
        fmt(hw, "RESULT lat", "scalar", "L1/K$", "-      ", "idle  ", 100.0, 200, 100.0),
        fmt(hw, "RESULT lat", "vector", "L2", "inv sc1", "loaded", 1100.0, 2000, 1000.0),
        fmt(hw, "RESULT lat", "inv-only", "L1/K$", "inv sc1", "loaded", 600.0, 1200, 590.0),
        fmt(hw, "RESULT lf", 262144, 192, 16, 3.0, 60.0, 62.0, 0.9, 0.84, 0.81),
        fmt(hw, "RESULT lf", 262144, 296, 32, 9.25, 20.0, 22.0, 0.6, 3.88, 3.5),
        fmt(hw, "RESULT launch", 304, 1024, -1, 1.62, 1.55),
        fmt(hw, "RESULT evx", "", "xcd", 37, 37, 0, "      ", 0.3, 0.42, 0.6, 1.2),
        fmt(hw, "RESULT evx", "", "device", 296, 296, 0, "      ", 1.1, 1.4, 1.9, 2.0),
        fmt(hw, "RESULT evx", "", "xcd", 1, 1, 0, " late ", 0.0, 0.0, 0.0, 0.25),
        fmt(imp, "RESULT impbar %s __sync", "rounds", 77.0),
    ]
    for proto, per in (("sw    ", 260.0), ("rounds", 95.0)):
        L.append(fmt(imp, "RESULT impbar %s %-36s R %2d: block %7.3f us (the", proto, "fused_moe (1 x 256, 12 idle waves)", 0, 0.5))
        L.append(fmt(imp, "RESULT impbar %s %-36s R %2d: block %7.3f us, per", proto, "fused_moe (1 x 256, 12 idle waves)", 32, 0.5 + per * 32e-3, per))
    return "\n".join(L)


def test_main_runs(tmp_path):
    (tmp_path / "hw_all.log").write_text(sample())
    r = run_command([sys.executable, "-m", "examples.vllm_llm.bench_params", str(tmp_path), "--json", str(tmp_path / "o.json")],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "microbenchmark-only" in r.stdout
