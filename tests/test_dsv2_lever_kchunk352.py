"""CPU proofs for D3 (lever kchunk352 + ETX_DSV2_KCHUNKS=4 + -DETX_DSV2_KCHUNKS=1): the patch is exactly its two edits,
the chunks cover h exactly with aligned boundaries, every wave signals every chunk once (so 148 signals per chunk is
the right wait count), the plan drops gate_up -> down's ETX edge only in the D3 build, the default plan is unchanged,
and the tiles refuse to build without the lever. No device required."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.inplace_session import ARM_ENV, ARM_FLAGS, ARMS, LEVERS_ALL
from examples.dsv2lite.levers import kchunk352 as kc

ROOT = Path(__file__).resolve().parents[1]
INTER, WORKERS, WAVES = 1408, 37, 4


def test_the_patch_is_exactly_its_two_edits(tmp_path, pinned_src):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    pf.apply(copy, "kchunk352")
    changed = sorted(str(p.relative_to(copy)) for p in copy.rglob("*")
                     if p.is_file() and p.read_bytes() != (pinned_src / p.relative_to(copy)).read_bytes())
    assert changed == ["kernels/fleet_kernel.hip", "runtime/fleet_types.h"]
    for rel, old, new in kc.EDITS:
        text = (copy / rel).read_text()
        assert text.count(new) == 1 and text.replace(new, old) == (pinned_src / rel).read_text()
    assert "constexpr int kExpertKChunk = 352;" in (copy / "runtime/fleet_types.h").read_text()


def test_the_patch_composes_with_the_five_levers(tmp_path, pinned_src):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    for lever in (*LEVERS_ALL, "kchunk352"):
        pf.apply(copy, lever)
    gemv = (copy / "kernels/gemv.h").read_text()
    body = pf._function(gemv, "__device__ inline void gemv_gate_up_rows(")
    assert body.count("FLEET_CHUNKS_DONE(") >= 3          # gemv_no_terminal_all keeps the per-group chunk signals
    k = (copy / "kernels/fleet_kernel.hip").read_text()
    assert k.count('asm volatile("buffer_inv sc1" ::: "memory");   // etx lever kchunk352') == 1


def test_chunks_cover_h_and_every_wave_signals_each_chunk_once():
    """Emulates gemv_gate_up_rows' row split and FLEET_CHUNKS_DONE (gemv.h:490-563) for 37 workers x 4 waves."""
    rows, n_chunks = kc.ROWS, -(-INTER // kc.ROWS)
    assert (n_chunks, rows % 32, rows % 8) == (4, 0, 0)
    assert [min(INTER, (c + 1) * rows) - c * rows for c in range(n_chunks)] == [352] * 4
    signals = [0] * n_chunks
    for worker in range(WORKERS):
        first, cnt = worker, -(-(INTER - worker) // WORKERS)
        for wave in range(WAVES):
            nxt = 0
            def done(j):
                nonlocal nxt
                next_row = first + j * WORKERS if j < cnt else INTER
                while nxt < n_chunks and next_row >= min(INTER, (nxt + 1) * rows):
                    signals[nxt] += 1
                    nxt += 1
            for k in range(wave, cnt, WAVES):           # this wave's rows k, k + 4, ...
                done(k + WAVES)
            done(cnt)
    assert signals == [WORKERS * WAVES] * n_chunks       # 148 each = the KCH_SIGNALS the down waits for


def test_plan_has_no_gate_up_edge_only_in_the_d3_build(monkeypatch):
    from examples.dsv2lite import graph
    monkeypatch.setenv("ETX_DSV2_KCHUNKS", "4")
    g = graph.build()
    gu, dn = g.grid("gate_up_5"), g.grid("down_5")
    assert gu.body.symbol == "fleet_expert_gate_up_k" and gu.out_edges == {}
    assert dn.body.symbol == "fleet_expert_down_k" and set(dn.in_edges) == {"E_norm_5"}
    assert "E_gu_5" not in g.events
    monkeypatch.setenv("ETX_DSV2_KCHUNKS", "1")
    g = graph.build()
    assert set(g.grid("down_5").in_edges) == {"E_gu_5"} and g.grid("gate_up_5").body.symbol == "fleet_expert_gate_up"


def test_d3_refuses_d4_and_bad_values(monkeypatch):
    from examples.dsv2lite import graph
    monkeypatch.setenv("ETX_DSV2_KCHUNKS", "2")
    with pytest.raises(ValueError, match="must be 1 or 4"):
        graph.build()
    monkeypatch.setenv("ETX_DSV2_KCHUNKS", "4")
    monkeypatch.setenv("ETX_DSV2_D4", "1")
    with pytest.raises(ValueError, match="ETX_DSV2_D4 with ETX_DSV2_KCHUNKS"):
        graph.build()


def test_tiles_need_the_flag_and_the_lever():
    shim = (ROOT / "examples/dsv2lite/fleet_shim.hip").read_text()
    block = shim[shim.index("#if defined(ETX_DSV2_KCHUNKS) && ETX_DSV2_KCHUNKS"):shim.index("#if defined(ETX_DSV2_D4)")]
    assert re.search(r'static_assert\(fleet::kExpertKChunk == 352', block)
    assert "fleet_expert_gate_up_k" in block and "fleet_expert_down_k" in block
    assert 'asm volatile("buffer_inv sc1"' in block           # chunk 0's acquire includes the L1
    assert ARMS["alld3"] == (*LEVERS_ALL, "kchunk352")
    assert ARM_FLAGS["alld3"] == ["-DETX_DSV2_KCHUNKS=1"] and ARM_ENV["alld3"] == {"ETX_DSV2_KCHUNKS": "4"}
