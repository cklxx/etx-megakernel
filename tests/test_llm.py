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
    assert plan.relay                                                  # hundreds of pollers per DEVICE word
    assert all(m == "static" for m in plan.modes.values())
    assert plan.workers_per_domain * plan.n_domains == M.workers(d)    # the partition assumed the plan's worker count
