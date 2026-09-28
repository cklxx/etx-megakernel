"""The microbenchmark parser against lines printed with the benchmarks' own printf formats (bench/hw/hw_bench.hip,
bench/hw/imp_bench.hip), so a format change that the parser misses fails here, not after a GPU session."""
import re
import subprocess
import sys
from pathlib import Path

from examples.vllm_llm.bench_params import derive, parse

ROOT = Path(__file__).resolve().parents[1]


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
        fmt(hw, "RESULT lf", 262144, 192, 16, 3.0, 60.0, 62.0, 0.9, 0.84, 0.81),
        fmt(hw, "RESULT lf", 262144, 296, 32, 9.25, 20.0, 22.0, 0.6, 3.88, 3.5),
        fmt(hw, "RESULT launch", 304, 1024, -1, 1.62, 1.55),
        fmt(hw, "RESULT evx", "xcd", 37, 37, 0, "      ", 0.3, 0.42, 0.6, 1.2),
        fmt(hw, "RESULT evx", "device", 296, 296, 0, "      ", 1.1, 1.4, 1.9, 2.0),
        fmt(hw, "RESULT evx", "xcd", 1, 1, 0, " late ", 0.0, 0.0, 0.0, 0.25),
        fmt(imp, "RESULT impbar %s __sync", "rounds", 77.0),
    ]
    for proto, per in (("sw    ", 260.0), ("rounds", 95.0)):
        L.append(fmt(imp, "RESULT impbar %s %-36s R %2d: block %7.3f us (the", proto, "fused_moe (1 x 256, 12 idle waves)", 0, 0.5))
        L.append(fmt(imp, "RESULT impbar %s %-36s R %2d: block %7.3f us, per", proto, "fused_moe (1 x 256, 12 idle waves)", 32, 0.5 + per * 32e-3, per))
    return "\n".join(L)


def test_parse_every_format():
    b = parse(sample())
    assert abs(b["lat"][("vector", "MALL", "-", "idle")] - 0.8123) < 1e-12
    assert b["lat"][("scalar", "L1/K$", "inv sc1", "loaded")] == 0.15
    assert b["lf"][(262144, 296, 32)]["tbs"] == 3.88
    assert b["launch"][(304, 1024, -1)] == 1.62
    assert b["evx"][("xcd", 37, 37, 0, False)]["median"] == 0.42
    assert b["evx"][("xcd", 1, 1, 0, True)]["ready"] == 0.25
    assert b["sync"]["rounds"] == 0.077
    assert abs(b["impbar"][("sw", "fused_moe", 32)]["per"] - 0.26) < 1e-9
    assert b["impbar"][("rounds", "fused_moe", 0)]["per"] is None


def test_derive():
    d = derive(parse(sample()))
    assert abs(d["t_v"][0] - 0.8123) < 1e-9 and abs(d["t_s"][0] - 0.7) < 1e-9
    assert d["F"][0] == 1.62 and d["bw"][0] == 3.88e6
    assert abs(d["t_swbar"][0] - 0.26) < 1e-9 and abs(d["t_round"][0] - 0.095) < 1e-9
    assert abs(d["A_task_vector"][0] - 1.8) < 1e-9 and abs(d["A_task_scalar"][0] - 0.24) < 1e-9
    assert d["handoff_domain"][0] == 0.42 and d["handoff_device"][0] == 1.4


def test_main_runs(tmp_path):
    (tmp_path / "hw_all.log").write_text(sample())
    r = subprocess.run([sys.executable, "-m", "examples.vllm_llm.bench_params", str(tmp_path), "--json", str(tmp_path / "o.json")],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "microbenchmark-only" in r.stdout
