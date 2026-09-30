"""ETX_VL_COH's contract in the emitted kernel: a coherent tile loses the acquire invalidate; a plain producer
whose event feeds a coherent consumer writes its L2 back even at DOMAIN scope; a DEVICE event fed only by coherent
producers arrives without the last-arriver flush. The policy itself is model-checked in formal/DataVis.tla."""
import re

from etx.codegen import emit_kernel
from etx.frontends import hip_link
from etx.ir import Graph, Resource
from etx.ops import attnprep, coh, gemv, topk
from etx.passes import compile_graph

RES = Resource(threads=256, lds_bytes=0, vgpr=64)


def _build(coherence: str, cross: bool) -> Graph:
    """imported a -> b(coherence) -> c(imported); c on another domain when cross (its in-event turns DEVICE)."""
    g = Graph("coh")
    g.tensor("t", (64,), role="activation", bytes_per_elem=2)
    g.etensor("E1", (1,), wait_count=4)
    g.etensor("E2", (1,), wait_count=4)
    g.call_device("a", (4,), hip_link("a.hip", "etx_tile_a"), resource=RES, args=["t"],
                  out_edges={"E1": "i->(0)"}, domain_map="i->(0)", duration_us=1.0)
    g.call_device("b", (4,), hip_link("b.hip", "etx_tile_b", coherence=coherence), resource=RES, args=["t"],
                  in_edges={"E1": "i->(0)"}, out_edges={"E2": "i->(0)"}, domain_map="i->(0)", duration_us=1.0)
    g.call_device("c", (1,), hip_link("c.hip", "etx_tile_c"), resource=RES, args=["t"],
                  in_edges={"E2": "i->(0)"}, domain_map="i->(1)" if cross else "i->(0)", duration_us=1.0)
    return g


def _case(src: str, name: str) -> str:
    m = re.search(rf"case \d+: \{{ // {name} \[.*?(?=case \d+: \{{ // |default: break;)", src, re.S)
    assert m, name
    return m.group(0)


def _phase3(src: str, name: str) -> str:
    m = re.search(rf"case \d+: \{{ // {name} arrives.*?break; \}}", src, re.S) or \
        re.search(rf"case \d+:.*// {name}\n.*?break;", src, re.S)
    assert m, f"phase 3 of {name}"
    return m.group(0)


def test_coherent_tile_skips_acquire_and_plain_producer_writes_back():
    plan = compile_graph(_build("coherent", cross=False), "gfx942", {}, {})
    ep1, ep2 = plan.events["E1"], plan.events["E2"]
    assert ep1.coherent_consumer and not ep1.producers_coherent          # a (plain) -> b (coherent)
    assert not ep2.coherent_consumer and ep2.producers_coherent          # b (coherent) -> c (plain)
    src = emit_kernel(plan)
    b_case = _case(src, "b")
    assert "coherent tile: no acquire invalidate" in b_case and "ETX_ACQUIRE" not in b_case
    assert "ETX_ACQUIRE_DOMAIN();" in _case(src, "c")                    # the imported consumer keeps its acquire
    # a's DOMAIN release is upgraded to the write-back because its consumer bypasses the L2
    assert "ETX_RELEASE_DEVICE();   // a coherent consumer bypasses this XCD's L2: write it back" in src


def test_imported_build_is_unchanged():
    plan = compile_graph(_build("imported", cross=False), "gfx942", {}, {})
    src = emit_kernel(plan)
    assert "coherent tile" not in src and "write it back" not in src and "_nf(" not in src
    assert "ETX_ACQUIRE_DOMAIN();" in _case(src, "b")


def test_all_coherent_device_event_arrives_without_flush():
    plan = compile_graph(_build("coherent", cross=True), "gfx942", {}, {})
    from etx.ir.types import Scope
    assert plan.events["E2"].scope == Scope.DEVICE and plan.events["E2"].producers_coherent
    src = emit_kernel(plan)
    assert re.search(r"etx_arrive_flush_DEVICE(_s)?_nf\(", src)
    plain = emit_kernel(compile_graph(_build("imported", cross=True), "gfx942", {}, {}))
    assert not re.search(r"_nf\(", plain)


def test_dsl_tiles_use_the_coherent_accessors():
    for src in (gemv.emit_tiles(), topk.emit_tiles(), attnprep.emit_tiles()):
        assert "ETX_DSL_COH_HELPERS" in src                              # guarded: safe under concatenation
        assert "dsl_a_ldbf(" in src or "dsl_a_ld64(" in src
    g = gemv.emit_tiles()
    for fn in ("dsl_a_stbf(rout + i", "dsl_a_ldbf(rows +", "dsl_a_ldi(d.ids + s)"):
        assert fn in g, fn
    # positions / slot_mapping are wave-uniform scalars: one coherent load per wave, broadcast (dsl_a_ldll_u)
    assert "dsl_a_ldll_u(positions)" in attnprep.emit_tiles()
    assert "dsl_a_ldll_u(slot_mapping)" in attnprep.emit_tiles()
    # weights stay plain: the stream loop's loads are untouched
    assert "base[(size_t)(s0 + u) * ETX_WAVE_SIZE]" in g
    # without the build flag the helpers are the plain accesses (the acquire-based builds are unchanged)
    assert "#else" in coh.HELPERS and "return *p;" in coh.HELPERS
