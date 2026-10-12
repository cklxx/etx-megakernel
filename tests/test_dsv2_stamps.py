"""Measurement format, source-copy boundaries and collision refusal (CPU only)."""
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from examples.dsv2lite import ledger, patch_fleet, patch_stamps


def test_legacy_prologue_remains_readable(tmp_path):
    path = tmp_path / "legacy.bin"
    path.write_bytes(struct.pack("<6Q", 100, 140, 190, 0, 0, 0))
    words, stride = ledger.read_prologue(path)
    assert stride == 3 and words == (100, 140, 190, 0, 0, 0)
    assert ledger.subphase_intervals(words[:3], "qkv") == {}


def test_stamps2_read_and_intervals(tmp_path):
    path = tmp_path / "stamps2.bin"
    row = (100, 110, 200, 110, 120, 150, 160, 190, 200, 200, 400, 700, 0, 0, 0, 0)
    path.write_bytes(struct.pack("<20Q", ledger.STAMPS2_MAGIC, 2, 16, 1, *row))
    words, stride = ledger.read_prologue(path)
    assert stride == 16 and words == row
    assert ledger.subphase_intervals(words, "qkv") == {
        "norm_tree": .3, "gamma_apply": .3, "norm": .9, "q_gemv": 2., "kv_a_gemv": 3.}
    with pytest.raises(ValueError, match="payload length"):
        path.write_bytes(path.read_bytes()[:-8])
        ledger.read_prologue(path)


def test_stamps2_merges_and_monotonicity():
    row = [0] * 16
    row[0] = 1
    row[12:] = [100, 300, 600, 1000]
    assert ledger.subphase_intervals(row, "merge") == {"M1_coefficients": 2., "M2_partial_fold": 3., "M3_w_uv_gemv": 4.}
    row[14] = 200
    with pytest.raises(ValueError, match="nonmonotonic"):
        ledger.subphase_intervals(row, "merge")


@pytest.mark.parametrize("patches", [[], ["gemv_no_terminal"], ["norm_gamma_early"], ["norm_tree"], ["gemv_no_terminal", "norm_tree"]])
def test_stamps_patch_composes_and_preserves_lds_copy(tmp_path, patches):
    pinned = Path("/tmp/etx-dsv2/fleet-3fb67a6/src")
    if not pinned.exists():
        pytest.skip("requires exported fleet 3fb67a6; never reads working tree")
    kernel = pinned / "kernels/fleet_kernel.hip"
    original = kernel.read_bytes()
    copy = tmp_path / "src"
    shutil.copytree(pinned, copy)
    for patch in patches:
        patch_fleet.apply(copy, patch)
    patch_stamps.apply(copy)
    result = (copy / "kernels/fleet_kernel.hip").read_text()
    attn = (copy / "kernels/attention.h").read_text()
    assert "union { int ids_blk[kBlock]; uint64_t stamps2[13]; };" in result
    assert "s.stamps2 : nullptr, 6)" in result and "s.stamps2 : nullptr, 8)" in result
    assert "etx_ds_stamp(etx_stamps, 1); // tree begin" in result
    for index in (9, 10, 11, 12):
        assert f"etx_ds_stamp(etx_stamps, {index});" in attn
    assert kernel.read_bytes() == original
    with pytest.raises(ValueError):
        patch_stamps.apply(copy)


def test_stamps2_rejects_isolate_before_build(tmp_path):
    env = {**os.environ, "ETX_DSV2_ISOLATE": "0:qkv_5", "ETX_OUT": str(tmp_path)}
    result = subprocess.run(["bash", "examples/dsv2lite/build.sh", "/nonexistent-pinned-export", "-DETX_DSV2_STAMPS2"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "share prologue slots" in result.stderr
