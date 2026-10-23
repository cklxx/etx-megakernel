"""Command-level cases moved from tests/test_pairs.py, through external APIs only."""
import json
import os
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
_LAST = []


def pr_main(argv=None):
    """`pr.main(argv)` as the command `python -m examples.vllm_llm.pairs`."""
    result = run_command([sys.executable, "-m", "examples.vllm_llm.pairs", *map(str, argv or [])], cwd=ROOT)
    _LAST.append(result.stdout)
    return result.returncode


def last_output():
    return _LAST[-1] if _LAST else ""


FUSED = "mode: ETX megakernel (vLLM kernels imported)"


GRAPH = "mode: unfused (same imported kernels, one launch per grid, HIP graph)"


ENV = "ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH="


ROUND48 = Path(os.environ.get("ETX_DATA_ROOT", ROOT)) / "results_2026-10-01/round48/interleave.log"


def run(mode, wall, dev=None, gpu="0", job="j", env=ENV, matched=32, gen=32):
    dev = wall - 0.1 if dev is None else dev
    head = [f"# job {job} gpu={gpu}"] if job else []
    return head + ([env] if env else []) + ([mode] if mode else []) + [
        f"RESULT model=/root/llm/phi-3.5-moe requests=1 matched={matched} gen={gen} median_ms={dev:.3f} "
        f"wall_ms={wall:.3f} tokens_per_s=158 matched_each={matched}"]


def session(walls, graph_first=False):
    """walls: (fused, graph) per pair, interleaved in that order (or graph first)."""
    lines = []
    for i, (f, g) in enumerate(walls):
        a = run(FUSED, f, job=f"p{i}_fused")
        b = run(GRAPH, g, job=f"p{i}_graph")
        lines += (b + a) if graph_first else (a + b)
    return lines


def write(tmp_path, lines, name="s.log"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


GOOD = [(6.00, 6.40), (6.01, 6.41), (6.02, 6.40), (5.99, 6.39), (6.00, 6.42)]


def refused(tmp_path, capsys, lines, *args):
    assert pr_main([write(tmp_path, lines), *args]) == 2
    out = last_output()
    assert out.startswith("REFUSED:") and "No verdict" in out and "VERDICT" not in out
    return out


def test_consistent_gain_is_established(tmp_path, capsys):
    assert pr_main([write(tmp_path, session(GOOD))]) == 0
    out = last_output()
    assert "VERDICT wall ESTABLISHED" in out
    assert "VERDICT device ESTABLISHED" in out and "[secondary basis]" in out


def test_graph_first_pairs_and_split_files(tmp_path, capsys):
    lines = session(GOOD, graph_first=True)
    half = len(lines) // 2
    files = [write(tmp_path, lines[:half], "a.log"), write(tmp_path, lines[half:], "b.log")]
    assert pr_main(files) == 0
    assert "VERDICT wall ESTABLISHED" in last_output()


@pytest.mark.parametrize("walls", [
    [(6.00, 6.10)] * 5,                                                  # median 1.6% <= 2%
    [(6.00, 6.40), (6.00, 7.00), (6.00, 6.40), (6.00, 6.80), (6.00, 6.40)],   # spread too wide for the median
    [(6.00, 6.40), (6.00, 6.40), (6.50, 6.40), (6.00, 6.40), (6.00, 6.40)],   # one pair negative
], ids=["small", "spread", "negative"])
def test_not_established(tmp_path, capsys, walls):
    assert pr_main([write(tmp_path, session(walls))]) == 1
    out = last_output()
    assert "VERDICT wall NOT ESTABLISHED" in out
    assert ("every pair positive: no" in out) == any(f >= g for f, g in walls)


def test_basis_selects_the_exit_code(tmp_path, capsys):
    lines = []
    for i in range(5):
        lines += run(FUSED, 6.00, dev=5.0, job=f"f{i}") + run(GRAPH, 6.40, dev=5.0, job=f"g{i}")
    p = write(tmp_path, lines)
    assert pr_main([p]) == 0
    assert pr_main([p, "--basis", "device"]) == 1


def test_vllm_controls_are_summarised_not_paired(tmp_path, capsys):
    lines = session(GOOD) + ["VLLM label=phi aiter=0 batch=1 context=1024 median_ms=11.035",
                             "VLLM label=phi aiter=1 batch=1 context=1024 median_ms=11.398"]
    assert pr_main([write(tmp_path, lines)]) == 0
    out = last_output()
    assert "control vLLM aiter=0 batch=1 context=1024: 1 run(s), median 11.035" in out
    assert "aiter=1" in out


def test_refuses_odd_run_count(tmp_path, capsys):
    assert "odd count" in refused(tmp_path, capsys, session(GOOD) + run(FUSED, 6.0))


def test_refuses_non_interleaved_pair(tmp_path, capsys):
    lines = session(GOOD[:3]) + run(FUSED, 6.0) + run(FUSED, 6.0) + session(GOOD[:1])
    assert "not interleaved" in refused(tmp_path, capsys, lines)


def test_refuses_wrong_pair_count(tmp_path, capsys):
    assert "incomplete: 4 of 5 pairs" in refused(tmp_path, capsys, session(GOOD[:4]))
    assert "incomplete: 5 of 6 pairs" in refused(tmp_path, capsys, session(GOOD), "--pairs", "6")


def test_refuses_result_without_mode(tmp_path, capsys):
    assert "without a preceding mode line" in refused(tmp_path, capsys, run(None, 6.0))


def test_refuses_unknown_and_stream_modes(tmp_path, capsys):
    assert "unknown mode" in refused(tmp_path, capsys, run("mode: something else", 6.0))
    assert "stream launches" in refused(tmp_path, capsys, run("mode: unfused (stream launches)", 6.0))


def test_refuses_missing_or_debug_env(tmp_path, capsys):
    assert "ETX_ENV line" in refused(tmp_path, capsys, run(FUSED, 6.0, env=None))
    out = refused(tmp_path, capsys, run(GRAPH, 6.0, env="ETX_ENV ETX_UNFUSED_SYNC=1 ETX_UNFUSED_NOGRAPH="))
    assert "ETX_UNFUSED_SYNC set" in out
    out = refused(tmp_path, capsys, run(GRAPH, 6.0, env="ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH=1"))
    assert "ETX_UNFUSED_NOGRAPH set" in out


def test_env_and_mode_do_not_carry_to_the_next_run(tmp_path, capsys):
    lines = run(FUSED, 6.0) + ["RESULT requests=1 matched=32 gen=32 median_ms=6.0 wall_ms=6.1"]
    assert "without a preceding mode line" in refused(tmp_path, capsys, lines)
    lines = run(FUSED, 6.0) + [GRAPH, "RESULT requests=1 matched=32 gen=32 median_ms=6.0 wall_ms=6.1"]
    assert "ETX_ENV line" in refused(tmp_path, capsys, lines)


def test_refuses_failed_correctness(tmp_path, capsys):
    assert "matched 31 of 32" in refused(tmp_path, capsys, run(FUSED, 6.0, matched=31))
    assert "matched 0 of 0" in refused(tmp_path, capsys, run(FUSED, 6.0, matched=0, gen=0))


def test_refuses_missing_field_and_zero_time(tmp_path, capsys):
    lines = [ENV, FUSED, "RESULT model=x matched=32 gen=32 median_ms=5.793 active_GB=3.087"]
    assert "RESULT field wall_ms" in refused(tmp_path, capsys, lines)
    assert "zero time" in refused(tmp_path, capsys, run(FUSED, 0.0, dev=0.0))


def test_refuses_runs_on_different_gpus(tmp_path, capsys):
    lines = session(GOOD[:4]) + run(FUSED, 6.0, gpu="0") + run(GRAPH, 6.4, gpu="1")
    assert "different GPUs" in refused(tmp_path, capsys, lines)


def test_refuses_missing_log_and_empty_log(tmp_path, capsys):
    assert pr_main([str(tmp_path / "absent.log")]) == 2
    assert "missing input: log" in last_output()
    assert "no runs found" in refused(tmp_path, capsys, ["nothing here"])


@pytest.mark.skipif(not ROUND48.is_file(), reason=f"round48 log not present at {ROUND48} (set ETX_DATA_ROOT)")
def test_round48_interleave_log_is_refused(capsys):
    """round48 printed RESULT lines without mode or ETX_ENV lines and ran no HIP-graph arm: no verdict from it."""
    assert pr_main([str(ROUND48)]) == 2
    assert "without a preceding mode line" in last_output()
