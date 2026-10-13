"""CPU proofs for D2 (lever coherent_loads + ETX_DSV2_COH): the patch is exactly its edits, the reachable payload reads
are all coherent, the plan drops every acquire but dense_down's and keeps every write-back, and a build whose plan and
tiles disagree is refused. No device required."""
from __future__ import annotations

import re
import shutil

import pytest

from examples.dsv2lite import coh_check
from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.levers import coherent_loads as cl


@pytest.fixture(scope="module")
def plans(monkeypatch_module):
    from etx.codegen import emit_kernel
    from etx.passes import compile_graph
    from examples.dsv2lite import graph
    out = {}
    for coh in ("0", "1"):
        monkeypatch_module.setenv("ETX_DSV2_COH", coh)
        g = graph.build()
        out[coh] = (g, emit_kernel(compile_graph(g, "gfx942", {}, {})))
    return out


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def patched(tmp_path, pinned_src):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    pf.apply(copy, "coherent_loads")
    return copy


def test_the_patch_is_exactly_its_edits(patched, pinned_src):
    changed = sorted(str(p.relative_to(patched)) for p in patched.rglob("*")
                     if p.is_file() and p.read_bytes() != (pinned_src / p.relative_to(patched)).read_bytes())
    assert changed == ["kernels/attention.h", "kernels/fleet_kernel.hip", "kernels/gemv.h"]
    for rel in changed:
        text = (patched / rel).read_text()
        for r, old, new in reversed(cl.EDITS):
            if r == rel:
                assert text.count(new) == 1
                text = text.replace(new, old)
        assert text == (pinned_src / rel).read_text(), rel


def _fn(src, rel, signature):
    return pf._function((src / rel).read_text(), signature)


def test_every_reachable_payload_read_is_coherent(patched):
    """The consumer side of each covered payload, function by function: no plain read of another task's output left."""
    att = "kernels/attention.h"
    kv_post = _fn(patched, att, "__device__ inline void kv_post(")
    assert "kv_a_raw[" not in kv_post and kv_post.count("ld_sys(kv_a_raw") == 3
    rows = _fn(patched, att, "__device__ inline void q_absorb_rows(")
    assert "q_head[" not in rows and "ld_sys(q_head + row0 + lane)" in rows
    gather = _fn(patched, att, "__device__ inline void q_absorb_gather(")
    assert "partials[" not in gather and "q_head[" not in gather and gather.count("ld_sys(") == 2
    merge = _fn(patched, att, "__device__ inline void merge_and_uv(")
    assert "partials[" not in merge and merge.count("ld_sys(partials") == 3
    gemv = "kernels/gemv.h"
    stage = _fn(patched, gemv, "__device__ __forceinline__ void stage_vector(")
    assert "= s4[i]" not in stage and "= src[i]" not in stage and "ld_sys_f4(s4 + i)" in stage
    k = "kernels/fleet_kernel.hip"
    f32 = _fn(patched, k, "__device__ inline void stage_residual_f32(")
    assert "if (false)" in f32 and "p4[" not in f32 and "= s4[" not in f32 and f32.count("ld_sys_f4(") == 3
    bf = _fn(patched, k, "__device__ inline void stage_residual(")
    assert "if (false)" in bf and "= p4[" not in bf and "= s4[" not in bf and bf.count("ld_sys_") == 3
    assert "? ld_sys(logits + lane) :" in _fn(patched, k, "__device__ inline float load_logit(")
    text = (patched / k).read_text()
    assert "rt.coherent_acts != 0" not in text and text.count("const bool coh = true;") == 2
    # the helpers are system-scope atomics through address_space(1) (global_load sc0 sc1, not flat_)
    helpers = (patched / gemv).read_text()
    assert helpers.count("__HIP_MEMORY_SCOPE_SYSTEM") == 2 and "address_space(1)" in helpers


def test_q_absorb_without_pub_wait_is_unreachable():
    """q_absorb (not patched) runs only without FLAG_PUB_WAIT; EVERY attention wrapper of the shim sets it (today's
    fleet_attention, and K1's fleet_attention_kvs behind -DETX_DSV2_KVA_SHARED), and no other code names the task."""
    shim = (pf.Path(cl.__file__).parents[1] / "fleet_shim.hip").read_text()
    wrappers = ('extern "C" __device__ void fleet_attention(', 'extern "C" __device__ void fleet_attention_kvs(')
    bodies = [pf._function(shim, w) for w in wrappers]
    assert "t.flags = fleet::FLAG_PUB_WAIT;" in bodies[0]
    assert "t.flags = fleet::FLAG_PUB_WAIT | fleet::FLAG_KVA_SHARED;" in bodies[1]
    for b in bodies:                                          # each: its descriptor and its stamp slot
        assert b.count("fleet::TASK_ATTENTION") == 2
    assert shim.count("fleet::TASK_ATTENTION") == 2 * len(wrappers)   # nowhere else


def test_shuffle_broadcasts_read_the_same_values():
    """merge_and_uv: lane c holds chunk c's (m, l) or (-inf, 0) -- what the predicated loads gave every thread.
    q_absorb_rows: every (i0 + k) the loop visits maps to a lane < rows that loaded exactly q[i0 + k]."""
    for n_chunks in range(1, 17):
        lane_m = [("m", c) if c < n_chunks else "-inf" for c in range(64)]
        assert [lane_m[c] for c in range(16)] == [("m", c) if c < n_chunks else "-inf" for c in range(16)]
    waves, batch = 4, 8
    for row0, rows in ((0, 64), (64, 64), (0, 32)):
        lane_q = {lane: row0 + lane for lane in range(64) if lane < rows}
        visited = []
        for wave in range(waves):
            i0 = row0 + wave * batch
            while i0 < row0 + rows:
                for k in range(batch):
                    assert lane_q[i0 + k - row0] == i0 + k
                    visited.append(i0 + k)
                i0 += waves * batch
        assert sorted(visited) == list(range(row0, row0 + rows))


def test_plan_drops_every_acquire_but_dense_down_and_keeps_every_write_back(plans):
    g0, src0 = plans["0"]
    g1, src1 = plans["1"]
    assert {gr.body.coherence for gr in g0.grids} == {"imported"}
    assert {gr.name for gr in g1.grids if gr.body.coherence != "coherent_loads"} == {"dense_dn_0"}
    assert "coherent-load tile" not in src0 and "write it back" not in src0
    acquires = re.findall(r"ETX_ACQUIRE_\w+\(\)", src1)
    assert len(acquires) == 1                                  # dense_dn_0's, its in-event is DEVICE scope
    assert "ETX_ACQUIRE_DEVICE()" in re.search(r"case \d+: \{ // dense_dn_0 .*?break; \}", src1, re.S).group(0)
    waiting = [gr for gr in g1.grids if gr.in_edges]           # embed waits on nothing
    assert src1.count("coherent-load tile: no acquire invalidate") == len(waiting) - 1 == 217
    # producers keep the write-back: no no-flush arrive, and every DOMAIN release now writes the L2 back
    assert "_nf(" not in src1
    phase3 = src1[src1.index("switch (t.type) {   // phase 3"):]
    plain = re.findall(r"case \d+: \{ // (\w+?)_?(\d*)\n        ETX_RELEASE_DOMAIN\(\);", phase3)
    # only events without an ETX consumer keep a plain drain: E_argmax (read after the kernel) and E_qabs, whose
    # consumer waits inside attention's body -- the shim writes that hand-off back itself (next test)
    assert sorted({n for n, _ in plain}) == ["argmax", "qabs"] and len(plain) == 1 + 27
    assert phase3.count("ETX_RELEASE_DEVICE();   // a coherent consumer bypasses this XCD's L2: write it back") == 134


def test_the_in_body_q_absorb_hand_off_writes_back():
    shim = (pf.Path(cl.__file__).parents[1] / "fleet_shim.hip").read_text()
    body = pf._function(shim, 'extern "C" __device__ void fleet_q_absorb(')
    assert re.search(r"fleet::drain_stores\(\);\n#ifdef FLEET_COH_LOADS\n(\s*//.*\n)+\s*fleet::fence_release\(\);\n#endif\n"
                     r"\s*__hip_atomic_fetch_add\(", body)
    assert cl.HELPERS.count("#define FLEET_COH_LOADS 1") == 1


def test_build_refuses_a_plan_without_the_patched_tiles(tmp_path, plans, patched, pinned_src):
    """D2's fail-closed check: a coherent plan with plain tiles (stale reads) and patched tiles with a plain plan are both
    refused (exit 2); the matching pairs pass."""
    mk0, mk1 = tmp_path / "mk0.hip", tmp_path / "mk1.hip"
    mk0.write_text(plans["0"][1])
    mk1.write_text(plans["1"][1])
    assert coh_check.main([str(mk1), str(pinned_src)]) == 2
    assert coh_check.main([str(mk0), str(patched)]) == 2
    assert coh_check.main([str(mk0), str(pinned_src)]) == 0
    assert coh_check.main([str(mk1), str(patched)]) == 0
    with pytest.raises(SystemExit, match="would read stale cached payloads"):
        coh_check.check(mk1, pinned_src)


def test_d2_verdict_applies_both_thresholds():
    from decimal import Decimal
    from etx.session.experiment import decide
    from examples.dsv2lite.arms import ARMS

    def rounds(default, relaxed, coh):
        return [{"default": {"device_ms": d}, "relaxed": {"device_ms": r}, "coh": {"device_ms": c}}
                for d, r, c in zip(default, relaxed, coh)]
    def d2_verdict(rs):
        return decide(ARMS["coh"].rule, rs, "coh")
    d = ["3.710", "3.712", "3.709", "3.711", "3.713"]
    r = list(map(Decimal, ["3.475", "3.478", "3.474", "3.476", "3.480"]))                      # relaxed-vs-default spread ~6 us -> tolerance 20 us
    ok = d2_verdict(rounds(d, r, [x + Decimal("0.010") for x in r]))       # coh 10 us behind relaxed, 225 us ahead of sc1
    assert ok["a_ok"] and ok["b_ok"] and ok["passed"] and ok["b_tol_us"] == 20.0
    slow = d2_verdict(rounds(d, r, [x + Decimal("0.050") for x in r]))     # 50 us behind relaxed: outside the tolerance
    assert slow["a_ok"] and not slow["b_ok"] and not slow["passed"]
    mixed = d2_verdict(rounds(d, r, ["3.600", "3.600", "3.600", "3.600", "3.720"]))   # one pair slower than sc1
    assert not mixed["a_ok"] and not mixed["passed"]
    assert ARMS["coh"].levers == ("coherent_loads",) and ARMS["coh"].env == {"ETX_DSV2_COH": "1"}
