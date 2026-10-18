"""CPU proofs for D2b (lever l1_loads + ETX_DSV2_COH=l1): D2's payload sites at agent scope, alone and on top of the five
levers (attn_prologue_live's own reads included), the plan drops every acquire but dense_down's WITHOUT adding a
write-back, and the build refuses a plan/tile mismatch. No device required."""
from __future__ import annotations

import re
import shutil

import pytest

from examples.dsv2lite import coh_check
from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.arms import LEVERS_ALL
from examples.dsv2lite.levers import coherent_loads as cl
from examples.dsv2lite.levers import l1_loads as l1


def _copy(tmp_path, pinned_src, *levers):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    for lever in levers:
        pf.apply(copy, lever)
    return copy


def _fn(src, rel, signature):
    return pf._function((src / rel).read_text(), signature)


def test_l1_is_d2s_sites_at_agent_scope(tmp_path, pinned_src):
    src = _copy(tmp_path, pinned_src, "l1_loads")
    gemv = (src / "kernels/gemv.h").read_text()
    assert "#define FLEET_L1_LOADS 1" in gemv and "FLEET_COH_LOADS" not in gemv
    helpers = l1._l1(cl.HELPERS)
    assert helpers.count("__HIP_MEMORY_SCOPE_AGENT") == 2 and "__HIP_MEMORY_SCOPE_SYSTEM" not in helpers and helpers in gemv
    assert "ld_sys" not in gemv and "ld_sys" not in (src / "kernels/attention.h").read_text()
    # every D2 site, same count, renamed
    assert len(l1.EDITS) == len(cl.EDITS)
    for (r1, o1, n1), (r2, o2, n2) in zip(l1.EDITS, cl.EDITS):
        assert (r1, o1) == (r2, o2) and n1.count("ld_l1") == n2.count("ld_sys")
    for rel in ("kernels/gemv.h", "kernels/fleet_kernel.hip", "kernels/attention.h"):
        text = (src / rel).read_text()
        for r, old, new in reversed(l1.EDITS):
            if r == rel:
                assert text.count(new) == 1
                text = text.replace(new, old)
        assert text == (pinned_src / rel).read_text(), rel


def test_reachable_payload_reads_are_agent_scope_on_the_best_arm(tmp_path, pinned_src):
    """all five levers, then l1_loads (the alld4r0l1 order): no plain payload read left, live prologue included."""
    src = _copy(tmp_path, pinned_src, *LEVERS_ALL, "l1_loads")
    att = "kernels/attention.h"
    live = _fn(src, att, "__device__ inline void kv_post_live(")
    assert "kv_a_raw[" not in live and live.count("ld_l1(kv_a_raw") == 3
    gather = _fn(src, att, "__device__ inline void q_absorb_gather_live(")
    assert "partials[" not in gather and "q_head[" not in gather and gather.count("ld_l1(") == 2
    for sig, n in (("__device__ inline void kv_post(", 3), ("__device__ inline void merge_and_uv(", 3)):
        body = _fn(src, att, sig)
        assert body.count("ld_l1(") == n, sig
    k = (src / "kernels/fleet_kernel.hip").read_text()
    assert "rt.coherent_acts != 0" not in k and k.count("const bool coh = true;") == 2
    assert "? ld_l1(logits + lane) :" in k


def test_plan_drops_acquires_and_adds_no_write_back(monkeypatch):
    from etx.codegen import emit_kernel
    from etx.passes import compile_graph
    from examples.dsv2lite import graph
    monkeypatch.setenv("ETX_DSV2_COH", "l1")
    g = graph.build()
    assert {gr.name for gr in g.grids if gr.body.coherence != "coherent_l1"} == {"dense_dn_0"}
    src = emit_kernel(compile_graph(g, "gfx942", {}, {}))
    assert len(re.findall(r"ETX_ACQUIRE_\w+\(\)", src)) == 1
    assert src.count("coherent-L1 tile: no acquire invalidate") == 217
    assert "write it back" not in src and "_nf(" not in src


def test_d4_with_l1_only_without_slices(monkeypatch):
    from examples.dsv2lite import graph
    monkeypatch.setenv("ETX_DSV2_COH", "l1")
    monkeypatch.setenv("ETX_DSV2_D4", "1")
    monkeypatch.setenv("ETX_DSV2_D4_ROWS", "0")
    g = graph.build()
    assert all(gr.body.coherence == "coherent_l1" for gr in g.grids if gr.body.symbol.startswith("fleet_d4_"))
    monkeypatch.setenv("ETX_DSV2_D4_ROWS", "32")
    with pytest.raises(ValueError, match="only ETX_DSV2_COH=l1 with ETX_DSV2_D4_ROWS=0"):
        graph.build()


def test_build_refuses_an_l1_plan_without_l1_tiles(tmp_path, pinned_src, monkeypatch):
    from etx.codegen import emit_kernel
    from etx.passes import compile_graph
    from examples.dsv2lite import graph
    monkeypatch.setenv("ETX_DSV2_COH", "l1")
    mk = tmp_path / "mk.hip"
    mk.write_text(emit_kernel(compile_graph(graph.build(), "gfx942", {}, {})))
    assert coh_check.main([str(mk), str(pinned_src)]) == 2
    l1src = _copy(tmp_path, pinned_src, "l1_loads")
    assert coh_check.main([str(mk), str(l1src)]) == 0
    d2src = tmp_path / "d2"
    shutil.copytree(pinned_src, d2src)
    pf.apply(d2src, "coherent_loads")
    assert coh_check.main([str(mk), str(d2src)]) == 2          # D2 tiles under a D2b plan: wrong arm, refused


def test_compacted_large_plan_is_checked_from_plan_json(tmp_path, pinned_src, monkeypatch):
    """x4 / x8 (>= 512 task types) compact their dispatch switches and lose the per-case coherence comments
    (_compact_dispatch, 85900b2): the check reads plan.json's per-type coherence instead and still refuses mismatches."""
    from etx.codegen import emit_kernel
    from etx.codegen.kernel import emit_plan_json
    from etx.passes import PassOptions, compile_graph
    from examples.dsv2lite import graph
    for k, v in dict(ETX_DSV2_COH="l1", ETX_DSV2_REQUESTS="4", ETX_DSV2_COUPLE="none", ETX_DSV2_KVA="shared").items():
        monkeypatch.setenv(k, v)
    plan = compile_graph(graph.build(), "gfx942", {}, {}, PassOptions(relay="on"))
    assert len(plan.graph.grids) >= 512
    mk = tmp_path / "megakernel_d0.hip"
    mk.write_text(emit_kernel(plan))
    assert "coherent-L1 tile" not in mk.read_text()               # the old evidence is gone at this size
    (tmp_path / "plan.json").write_text(emit_plan_json(plan))
    l1src = _copy(tmp_path, pinned_src, "l1_loads")
    assert coh_check.main([str(mk), str(l1src)]) == 0
    assert coh_check.main([str(mk), str(pinned_src)]) == 2        # l1 plan, unpatched tiles: still refused
