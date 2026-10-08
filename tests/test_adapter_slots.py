"""The imported tiles' slot index (etx/importer/adapter.py): wave-uniform through readfirstlane only with several slots,
the one-slot form unchanged, and a slot stride that is not whole waves of the machine model refused at generation."""
import pytest

from etx.importer.adapter import emit_adapters


def _imp(st, k, block):
    return {"kern": {"kernel": "k", "block": [block, 1, 1], "slot_threads": st, "slots": k, "grid_offset": 64,
                     "lds_bytes": 0, "slot_barrier": k > 1, "barriers": 1}}


def test_several_slots_take_the_slot_through_readfirstlane():
    src = emit_adapters(_imp(256, 4, 256))
    # etx_tile_, etx_all_blocks_ and etx_tile_x_ forms
    assert src.count("const int slot = __builtin_amdgcn_readfirstlane(tx / 256), tid = tx - slot * 256;") == 3


def test_one_slot_keeps_the_old_form():
    """One slot: the guard slot < 1 already makes slot 0 known inside the call. Not a literal 0 either: waves past a
    one-slot block (block < workgroup) need slot >= 1 to skip it."""
    src = emit_adapters(_imp(256, 1, 256))
    assert "readfirstlane" not in src
    assert src.count("const int slot = tx / 256, tid = tx - slot * 256;") == 3
    assert "if (slot >= 1) return;" in src or "slot < 1 && blk < nblk" in src


def test_slot_stride_must_be_whole_waves():
    with pytest.raises(ValueError, match=r"slot stride 96 threads is not a whole number of gfx942 waves \(64 lanes\)"):
        emit_adapters(_imp(96, 2, 96))
