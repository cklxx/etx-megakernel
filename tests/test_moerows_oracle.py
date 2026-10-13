"""hw_bench moerows (Q2 decision microbenchmark): the CPU-checkable half -- each expert half covered exactly once per
pass in both ownership modes, every measured read cold (beyond MALL + all L2s), addresses inside the buffer."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moerows_oracle", ROOT / "bench/hw/moerows_oracle.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_the_benchmark_s_ownership_and_coldness_hold():
    lines = oracle.check()
    assert [l.split(":")[0] for l in lines] == ["OK coverage", "OK cold", "OK bounds"]


def test_the_oracle_refuses_an_ownership_that_misses_rows(monkeypatch):
    monkeypatch.setattr(oracle, "task_rows", lambda j, il: [j * oracle.ROWS + r for r in range(oracle.ROWS - 1)] * 1)
    with pytest.raises(AssertionError):
        oracle.check()


def test_the_oracle_refuses_a_schedule_that_re_reads_warm(monkeypatch):
    monkeypatch.setattr(oracle, "expert", lambda s, it: s)              # the same 8 experts every pass: MALL-resident
    with pytest.raises(AssertionError, match="re-read after only"):
        oracle.check()
