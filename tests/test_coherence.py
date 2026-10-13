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


def test_coherent_loads_tile_skips_acquire_but_writes_back_as_a_producer():
    """D2 (fleet's tiles, ETX_DSV2_COH): loads coherent, stores plain -> no acquire, and its own release writes back."""
    plan = compile_graph(_build("coherent_loads", cross=False), "gfx942", {}, {})
    ep1, ep2 = plan.events["E1"], plan.events["E2"]
    assert ep1.coherent_consumer and not ep1.producers_coherent          # a (plain) -> b (coherent loads)
    assert not ep2.coherent_consumer and not ep2.producers_coherent      # b is a PLAIN producer
    src = emit_kernel(plan)
    b_case = _case(src, "b")
    assert "coherent-load tile: no acquire invalidate" in b_case and "ETX_ACQUIRE" not in b_case
    assert "ETX_ACQUIRE_DOMAIN();" in _case(src, "c")
    assert "ETX_RELEASE_DEVICE();   // a coherent consumer bypasses this XCD's L2: write it back" in src
    # a chain of coherent-load tiles: the producer still writes back for its coherent-load consumer
    g = _build("coherent_loads", cross=False)
    g.grid("c").body = hip_link("c.hip", "etx_tile_c", coherence="coherent_loads")
    src2 = emit_kernel(compile_graph(g, "gfx942", {}, {}))
    assert src2.count("ETX_RELEASE_DEVICE();   // a coherent consumer bypasses this XCD's L2: write it back") == 2
    assert "ETX_ACQUIRE" not in _case(src2, "c")


def test_coherent_loads_device_event_keeps_the_flush():
    plan = compile_graph(_build("coherent_loads", cross=True), "gfx942", {}, {})
    from etx.ir.types import Scope
    assert plan.events["E2"].scope == Scope.DEVICE and not plan.events["E2"].producers_coherent
    src = emit_kernel(plan)
    assert re.search(r"etx_arrive_flush_DEVICE(_s)?\(", src) and not re.search(r"_nf\(", src)


def test_coherent_l1_tile_skips_acquire_and_adds_no_write_back():
    """D2b: agent-scope payload loads read the XCD's L2, so neither the tile nor its producers change their releases."""
    plan = compile_graph(_build("coherent_l1", cross=False), "gfx942", {}, {})
    ep1, ep2 = plan.events["E1"], plan.events["E2"]
    assert not ep1.coherent_consumer and not ep1.producers_coherent and not ep2.producers_coherent
    src = emit_kernel(plan)
    b_case = _case(src, "b")
    assert "coherent-L1 tile: no acquire invalidate" in b_case and "ETX_ACQUIRE" not in b_case
    assert "write it back" not in src and "_nf(" not in src
    assert "ETX_ACQUIRE_DOMAIN();" in _case(src, "c")
    cross = emit_kernel(compile_graph(_build("coherent_l1", cross=True), "gfx942", {}, {}))
    assert re.search(r"etx_arrive_flush_DEVICE(_s)?\(", cross) and "_nf(" not in cross     # DEVICE events keep the flush


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
    for fn in ("dsl_a_stbf(rout + i", "dsl_a_ldbf(rows +", "dsl_a_ldi(d.ids + (size_t)rq * d.srows + s)"):
        assert fn in g, fn
    # positions / slot_mapping are wave-uniform scalars: one coherent load per wave, broadcast (dsl_a_ldll_u)
    assert "dsl_a_ldll_u(positions)" in attnprep.emit_tiles()
    assert "dsl_a_ldll_u(slot_mapping)" in attnprep.emit_tiles()
    # weights never go through the coherent accessors: the stream loop loads them with dsl_wload, which is a plain
    # load (or a non-temporal one under ETX_DSL_NT -- weights are read once per token and have no reuse to keep)
    assert "dsl_wload(DSL_WP(s0 + u))" in g
    assert "#define DSL_WP(sg) (base + (size_t)(sg) * ETX_WAVE_SIZE)" in g             # the default address
    assert "#define DSL_WP(sg) (ubase + (size_t)(sg) * ETX_WAVE_SIZE + lane)" in g     # ETX_DSL_SCALAR / GU1: same address
    assert "__builtin_nontemporal_load" in g and "#ifdef ETX_DSL_NT" in g
    for bad in ("dsl_a_ld", "dsl_a_st"):
        assert bad not in g.split("dsl_wload")[1].split("consume(")[0], bad
    # without the build flag the helpers are the plain accesses (the acquire-based builds are unchanged)
    assert "#else" in coh.HELPERS and "return *p;" in coh.HELPERS
