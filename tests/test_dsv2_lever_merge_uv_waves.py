"""Lever merge_uv_waves (M1, docs/deepseek/DSV2-D1-D5.md): merge's W_UV rows over all four waves, bit-identical by construction."""
import shutil

import pytest

from examples.dsv2lite import arms
from examples.dsv2lite.levers import merge_uv_waves as mw
from examples.dsv2lite.patch_fleet import apply, levers

WAVES, LANES = 4, 64


def owned(rpi, cnt, wave):
    """gemv_rows_pipelined's row walk: first group k = rpi * wave, then every rpi * WAVES."""
    k, rows = rpi * wave, []
    while k < cnt:
        rows += [k + r for r in range(rpi) if k + r < cnt]
        k += rpi * WAVES
    return rows


def test_every_row_once_and_every_wave_busy_with_2_rows_per_wave():
    for rpi, busy in ((8, 1), (2, 4)):
        per_wave = [owned(rpi, 8, w) for w in range(WAVES)]
        assert sorted(r for rows in per_wave for r in rows) == list(range(8))
        assert sum(bool(rows) for rows in per_wave) == busy          # today: wave 0 alone; lever: all four
    # the lane -> chunk map is K's alone (idx[d] = lane + d * 64): the same chunk feeds each row's fma8 either way
    k512 = 512 // 8
    assert all(lane < k512 for lane in range(LANES))


def test_the_lever_applies_once_on_top_of_the_kept_recipe(tmp_path, pinned_src):
    src = tmp_path / "src"
    shutil.copytree(pinned_src, src)
    for name in (*arms.LEVERS_ALL, "l1_loads", mw.NAME):
        apply(src, name)
    text = (src / "kernels/attention.h").read_text()
    assert text.count("etx lever merge_uv_waves") == 1 and text.count("gemv_rows_pipelined<2, 1>(w_uv") == 1
    assert "if (kv_lora / 8 <= kWaveLanes)" in text                   # wider K keeps the original call
    with pytest.raises(BaseException):
        apply(src, mw.NAME)                                             # a second application finds no original text


def test_the_m1_arm_is_the_kept_arm_plus_the_lever():
    kept, m1 = arms.ARMS["alld4r0l1"], arms.ARMS["alld4r0l1mw"]
    assert m1.levers == (*kept.levers, "merge_uv_waves") and m1.flags == kept.flags and m1.env == kept.env
    assert m1.order == ("alld4r0l1", "alld4r0l1mw") and m1.rule.reference == "alld4r0l1"
    assert (m1.rule.min_median_us, m1.rule.spread_factor, m1.rule.labels) == (30, 2, ("KEEP", "DROP"))
    assert "merge_uv_waves" in levers()
