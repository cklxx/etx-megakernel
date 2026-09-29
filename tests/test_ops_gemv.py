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
    assert '"dsl_gemv", 10, 80, 64' in hdr and '"pppppliiii"' in hdr
