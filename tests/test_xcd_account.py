"""etx.models.vllm_llm.xcd_account on a simulated DeepSeek-V2-Lite schedule: every worker's classes must cover the
span exactly once, and the critical-path account must telescope to the span."""
import json
from array import array

import pytest

from etx.passes import compile_graph
from etx.sim import simulate
from etx.tools import timeline as tl
from examples.dsv2lite import bytes as dsb
from examples.dsv2lite import graph as dsv
from etx.models.vllm_llm import xcd_account as xa


@pytest.fixture(scope="module")
def sim_trace():
    plan = compile_graph(dsv.build(layers=3), "gfx942", {}, {})
    res = simulate(plan, seed=0)
    assert not res.deadlock
    return tl.plan_json_from_obj(plan), tl.trace_from_sim(plan, res)


def test_classes_partition_every_worker(sim_trace):
    pj, rows = sim_trace
    r = xa.account(pj, rows, dsb.spec(), layer=1)
    tr, wiv = r["_tr"], r["_wiv"]
    for ivs in wiv.values():
        assert all(c in xa.CLASSES and b > a for c, a, b, _ in ivs)
        assert all(x[2] <= y[1] + 1e-9 for x, y in zip(ivs, ivs[1:]))            # disjoint and in order
        assert sum(xa.class_totals(ivs).values()) == pytest.approx(tr.span, rel=1e-9)
    assert sum(r["taxonomy_share"].values()) == pytest.approx(1.0, abs=1e-9)
    for x in r["per_xcd"]:
        assert sum(x["share"].values()) == pytest.approx(1.0, abs=1e-9)
    body = sum(t.done - t.ready for t in tr.by_id.values())
    assert r["machine_worker_us"]["bytes"] + r["machine_worker_us"]["machinery"] == pytest.approx(body, rel=1e-9)


def test_critical_account_telescopes(sim_trace):
    pj, rows = sim_trace
    r = xa.account(pj, rows, dsb.spec(), layer=1)
    c = r["critical"]
    assert c["sum_us"] == pytest.approx(c["span_us"], rel=1e-3)
    assert sum(v["tasks"] for v in c["by_kind"].values()) == c["steps"]
    edges = r["critical_edges"]
    assert sum(e["edges"] for e in edges) == c["steps"] - 1
    first_gap = tl.critical_path(r["_tr"])[0]["gap"]
    assert sum(e["sum_us"] for e in edges) + first_gap == pytest.approx(c["total"]["dep_gap"] + c["total"]["queue_gap"])


