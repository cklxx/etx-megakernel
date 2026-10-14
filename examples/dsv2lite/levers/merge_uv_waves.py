"""Lever merge_uv_waves (M1, docs/DSV2-D1-D5.md): merge's W_UV GEMV spreads its 8 rows over all four waves.

fleet's merge_and_uv (kernels/attention.h @3fb67a6) applies the task's slice of W_UV with
`gemv_rows(..., n_rows, kv_lora, ...)`. K = kv_lora = 512 selects `gemv_rows_pipelined<8, 1>`: 8 rows per wave per
group, groups interleaved over waves (k = 8 * wave), so with the task's 8 rows wave 0 loads, reduces and stores all of
them while waves 1-3 are idle (stamped M3: 5.24 us per task). This lever calls `gemv_rows_pipelined<2, 1>` instead:
2 rows per wave, one pass. Each row is still `fma8` of the lane's one 16-byte chunk (the lane -> chunk map depends on
K only), then `wave_sum`, then lane 0's bf16 epilogue, so the outputs are bit-identical. The direct call is valid only
while one chunk per lane covers K (kv_lora <= 512); anything wider keeps the original call (a uniform branch).
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.patch_fleet import _once

NAME = "merge_uv_waves"
OLD = """    gemv_rows(w_uv, kv_lora, o_c_lds, o + head * v_head + row0, nullptr, EPI_BF16, 1.f,
              n_rows, kv_lora, /*xcd=*/0, /*n_xcds=*/1, /*worker=*/0, /*n_workers=*/1);
"""
NEW = """    // etx lever merge_uv_waves: 2 rows per wave, so all four waves share the task's rows (gemv_rows picks <8, 1> here:
    // wave 0 alone). Same per-row fma8 + wave_sum + bf16 epilogue: bit-identical outputs. One chunk per lane covers
    // K only up to 8 * kWaveLanes; wider K keeps the original call.
    if (kv_lora / 8 <= kWaveLanes) {
        gemv_rows_pipelined<2, 1>(w_uv, kv_lora, o_c_lds, o + head * v_head + row0, nullptr, EPI_BF16, 1.f,
                                  kv_lora, /*first=*/0, /*cnt=*/n_rows, /*n_workers=*/1, /*coh=*/false, nullptr);
    } else {
        gemv_rows(w_uv, kv_lora, o_c_lds, o + head * v_head + row0, nullptr, EPI_BF16, 1.f,
                  n_rows, kv_lora, /*xcd=*/0, /*n_xcds=*/1, /*worker=*/0, /*n_workers=*/1);
    }
"""


def apply(src: Path) -> None:
    f = src / "kernels/attention.h"
    f.write_text(_once(f.read_text(), OLD, NEW, NAME))
