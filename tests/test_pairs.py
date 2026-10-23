"""Paired fused / HIP-graph results: the candidate 1 verdict and every refusal."""
import os
from pathlib import Path

import pytest

from examples.vllm_llm import pairs as pr

FUSED = "mode: ETX megakernel (vLLM kernels imported)"
GRAPH = "mode: unfused (same imported kernels, one launch per grid, HIP graph)"
ENV = "ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH="
ROOT = Path(__file__).resolve().parents[1]
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








def test_a_non_positive_pair_always_fails_the_spread_rule():
    """With one gain <= 0 the spread is at least the largest gain, so the median never exceeds twice the spread:
    the every-pair-positive rule cannot fail alone. The verdict still reports it."""
    v = pr.verdict([0.30, 0.30, 0.30, 0.30, -0.001])
    assert not v["positive"] and not v["established"] and v["median"] < 2 * v["spread"]






def refused(tmp_path, capsys, lines, *args):
    assert pr.main([write(tmp_path, lines), *args]) == 2
    out = capsys.readouterr().out
    assert out.startswith("REFUSED:") and "No verdict" in out and "VERDICT" not in out
    return out
























