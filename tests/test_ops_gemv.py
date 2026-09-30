"""etx.ops.gemv: the import entries match the tile's argument reads, the generated source has the entry points model.py
links, and the adapter leaves DSL entries to the generator."""
import json
import re

from etx.importer.adapter import emit_adapters, emit_host_header
from etx.ops import gemv


def test_entries_and_tiles_agree():
    ent = gemv.import_entries()
    src = gemv.emit_tiles()
    for e, form in gemv.FORMS.items():
        i = ent[e]
        assert i["dsl"] == form and i["arg_bytes"] == gemv.ARG_BYTES and i["grid_offset"] == gemv.GRID_OFF
        for name in (f"etx_tile_{e}", f"etx_tile_x_{e}", f"etx_pre_{e}", f"etx_pre_x_{e}"):
            assert re.search(rf"void {name}\(const etx_ctx\* c\)", src), name
        assert f"a + {gemv.GRID_OFF + 12}" in src                       # nblk from the grid words, as the adapter's tiles
    # every argument offset the tile reads is a declared parameter at that offset
    offs = {p["offset"] for p in ent["dsl_gemv"]["params"]}
    for m in re.finditer(r"\(a \+ (\d+)\)", src.split("struct DslArgs")[1].split("}")[1]):
        assert int(m.group(1)) in offs, m.group(0)
    assert {p[2] for p in gemv.PARAMS} == offs


def test_adapter_skips_dsl_but_host_header_includes_it():
    imp = json.loads(json.dumps(gemv.import_entries()))
    tiles = emit_adapters(imp)
    assert "etx_tile_dsl_gemv" not in tiles and "etx_imp_dsl" not in tiles
    hdr = emit_host_header(imp)
    # 25 params, 160 B block, grid words at 144: the last six are the batch fields (nb + five per-column strides)
    assert '"dsl_gemv", 25, 160, 144' in hdr and '"pppppliiiippppifiipiiiiii"' in hdr


def test_touch_hook_dense_forms_only():
    """The touch hook (waves 1.. pull the first weight trip into L2 during the wait) exists for the dense GEMV forms in
    both entry forms, not for the expert forms (their rows depend on the top-k, i.e. on the event)."""
    src = gemv.emit_tiles()
    for e in ("dsl_gemv", "dsl_gemv_fold"):
        assert re.search(rf"void etx_touch_{e}\(const etx_ctx\* c\)", src) and re.search(rf"void etx_touch_x_{e}\(const etx_ctx\* c\)", src)
    for e in ("dsl_moe_up", "dsl_moe_down", "dsl_moe_up_fold", "dsl_moe_down_fold"):
        assert f"etx_touch_{e}(" not in src and f"etx_touch_x_{e}(" not in src
    touch = src[src.index("void dsl_touch_plain("):src.index("TOUCH_FORMS") if "TOUCH_FORMS" in src else None]
    assert "DSL_RCH" in touch and 'asm volatile("" :: "v"(acc))' in src[src.index("dsl_touch_rows"):]


def test_kernel_emits_touch_before_the_wait_on_the_other_waves():
    """A grid with a touch hook and an input event calls it on threadIdx >= 64 before wave 0's wait; a grid without an
    input event (nothing to overlap) does not; the plan JSON carries the hook for the simulator."""
    from etx.codegen import emit_kernel, emit_plan_json
    from etx.frontends import hip_link
    from etx.ir.types import Graph, Resource
    from etx.passes import compile_graph
    g = Graph("touch")
    g.tensor("w", (8,), role="weight", bytes_per_elem=1)
    g.etensor("E0", (1,), wait_count=1)
    g.etensor("E1", (1,), wait_count=4)
    res = Resource(threads=1024)
    g.call_device("first", (1,), hip_link("t.hip", "etx_tile_first", touch="etx_touch_first"), resource=res, args=["w"],
                  out_edges={"E0": "i->(0)"}, duration_us=1.0)
    g.call_device("second", (4,), hip_link("t.hip", "etx_tile_second", pre_wait="etx_pre_second", touch="etx_touch_second"),
                  resource=res, args=["w"], in_edges={"E0": "i->(0)"}, out_edges={"E1": "i->(0)"}, duration_us=1.0)
    plan = compile_graph(g, "gfx942", {}, {})
    src = emit_kernel(plan)
    assert "etx_touch_first(&ctx)" not in src
    case = src[src.index("// second [static]"):]
    case = case[:case.index("break; }")]
    i_touch, i_wait, i_pre = case.index("etx_touch_second(&ctx)"), case.index("if (threadIdx.x == 0) {"), case.index("etx_pre_second(&ctx)")
    assert i_touch < i_wait and "if (threadIdx.x >= 64) etx_touch_second(&ctx)" in case and i_pre < i_wait
    assert "#ifndef ETX_NO_TOUCH" in case
    unfused = src[src.index("etx_unfused_d0"):]
    assert "etx_touch_second" not in unfused
    import json
    types = json.loads(emit_plan_json(plan))["types"]
    assert types["second"]["touch"] == "etx_touch_second" and types["first"]["touch"] == "etx_touch_first"
