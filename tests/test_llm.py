"""examples/llm: the generic decoder graph compiles for dense, GQA+bias, q/k-norm and MoE configs."""
import math
import os

import pytest

from etx.passes import compile_graph

CONFIGS = ["qwen2.5-1.5b", "qwen3-8b", "qwen3-30b-a3b"]


@pytest.mark.parametrize("name", CONFIGS)
def test_llm_graph_compiles(name, monkeypatch):
    here = os.path.join(os.path.dirname(__file__), "..", "examples", "llm", "configs", f"{name}.json")
    monkeypatch.setenv("ETX_LLM_CONFIG", here)
    monkeypatch.setenv("ETX_LLM_LAYERS", "2")
    from examples.llm import model as M
    d = M.load_dims()
    p = M.partition(d)
    for rows, key in ((d.NQKV, "qkv"), (d.H, "o"), (d.V, "lm")):
        assert math.ceil(rows / p[key]) <= M.workers(d), key        # one wave of tasks per GEMV phase
    plan = compile_graph(M.build(), "gfx942", {})
    assert not plan.relay                                              # relay off by default (hierarchical acquire unsafe)
    assert all(m == "static" for m in plan.modes.values())
    assert plan.workers_per_domain * plan.n_domains == M.workers(d)    # the partition assumed the plan's worker count


@pytest.mark.parametrize("name", CONFIGS)
def test_llm_sliced_graph(name, monkeypatch):
    """One slice per XCD: two DEVICE-scope events per layer, everything else XCD-local; no worker runs two tasks of one grid."""
    import collections
    here = os.path.join(os.path.dirname(__file__), "..", "examples", "llm", "configs", f"{name}.json")
    monkeypatch.setenv("ETX_LLM_CONFIG", here)
    monkeypatch.setenv("ETX_LLM_LAYERS", "2")
    from examples.llm import model as M, model_sliced as MS
    d = M.load_dims()
    S = MS.slices(d)
    assert sum(c for _, c in S["q"]) == d.NH and sum(c for _, c in S["h"]) == d.H
    plan = compile_graph(MS.build(), "gfx942", {})
    from etx.ir.types import Scope
    dev = [e.name for e in plan.events.values() if e.scope == Scope.DEVICE]
    assert dev == ["E_embed", "E_o_0", "E_down_0", "E_o_1", "E_down_1", "E_lmf", "E_lm"]
    assert not any(g.name.startswith(("post_", "merge_")) for g in plan.graph.grids)   # folded into qkv / attention
    for g in plan.graph.grids:
        if len(g.grid) != 2 or g.grid[1] != M.workers(d) // MS.X:
            continue                                                  # (X, W) GEMV grids: one task per worker
        per_w = collections.Counter(t.worker for t in plan.tasks if t.grid == g.name)
        assert max(per_w.values()) == 1, g.name
