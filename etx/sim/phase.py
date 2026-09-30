"""The measured phase cost of MI300X, from hw_bench stream (results_2026-09-30/hw_stream.log).

This replaces the per-task tax the old model fitted (docs/MODEL-vllm-kernels.md section 19). Two facts decide a
phase, both measured with no events and no ETX, just N workgroups each reading B bytes:

  1. a floor. 296 workgroups reading 896 B each still take 3.2 us, so any full-device phase costs about 3 us
     whatever it moves. Below that the bytes are free; above it they are not.
  2. a per-workgroup rate that depends on how many bytes that workgroup reads and on which rows it reads.
     `contig` (one contiguous block per workgroup, what dsl_gemv_rows does) saturates near 7.1 KB/us per
     workgroup until the block is large; `rows` (row r to worker r % W, interleaved) is faster.

The measured table is at 296 workgroups. Aggregate rates above HBM's achievable peak mean the phase fitted in the
256 MB Infinity Cache, which a weight stream (6.1 GB per token) never does, so `hbm_rate` clamps the curve to
BW_HBM / 296 and the small `rows` points -- which imply 4.6-5.8 TB/s -- do not enter the model. Their raw values
are kept in SPAN_US so the clamp stays visible.
"""
from __future__ import annotations

FLOOR_US = 3.0          # a full-device phase costs this before any bytes (hw_bench stream, 896 B per workgroup)
BW_HBM = 3.55e6         # bytes/us an all-device stream sustains from HBM (the 1 MB per workgroup point)
N_REF = 296             # workgroups the table was measured with

# (bytes per workgroup, phase span us, workgroups) averaged over the two runs of 2026-09-30. The merge row ran with
# 256 workgroups, the rest with 296, so the width is carried rather than assumed.
SPAN_US = {
    "contig": [(896, 3.30, 296), (8192, 4.04, 256), (28672, 7.05, 296), (51200, 10.09, 296), (159744, 17.84, 296),
               (319488, 34.43, 296), (1048576, 91.32, 296)],
    "rows":   [(896, 3.16, 296), (8192, 3.37, 256), (28672, 4.86, 296), (51200, 6.16, 296), (159744, 16.64, 296),
               (319488, 32.16, 296), (1048576, 87.93, 296)],
}


def _interp(pts, b: float) -> float:
    if b <= pts[0][0]:
        return pts[0][1]
    for (b0, s0), (b1, s1) in zip(pts, pts[1:]):
        if b <= b1:
            return s0 + (s1 - s0) * (b - b0) / (b1 - b0)
    return pts[-1][1]                                     # flat beyond the largest measured transfer


# bytes/us one workgroup pulls when nothing competes with it: hw_bench stream "gate_up x8", 8 workgroups x 312 KB
# in 12.10 us (contig) and 11.74 (rows), the floor removed. Only measured at that size; treated as flat.
CU_RATE = {"contig": 319488 / (12.10 - FLOOR_US), "rows": 319488 / (11.74 - FLOOR_US)}


def _agg_points(mapping: str):
    """(bytes per workgroup, aggregate bytes/us) from each measured row, at the width that row was run with."""
    return [(b, n * b / max(span - FLOOR_US, 1e-3)) for b, span, n in SPAN_US[mapping]]


def agg_bw_raw(bytes_per_wg: float, mapping: str = "contig") -> float:
    """The aggregate rate exactly as measured, Infinity Cache included."""
    if bytes_per_wg <= 0:
        return BW_HBM
    return _interp(_agg_points(mapping), bytes_per_wg)


def raw_rate(bytes_per_wg: float, mapping: str = "contig") -> float:
    """Bytes/us one of N_REF competing workgroups pulls, as measured."""
    return agg_bw_raw(bytes_per_wg, mapping) / N_REF


def agg_bw(bytes_per_wg: float, mapping: str = "contig") -> float:
    """Bytes/us the whole GPU sustains when every workgroup reads this many bytes. It is not a constant: a
    workgroup that reads 28 KB keeps a shallower queue than one that reads 1 MB, and the device only reaches
    3.5 TB/s at the deep end (2.1 TB/s at 28 KB). Clamped to BW_HBM, which is what removes the Infinity Cache
    from the small `rows` points -- they measure 4.6-5.8 TB/s, above what HBM can deliver."""
    return min(agg_bw_raw(bytes_per_wg, mapping), BW_HBM)


def cu_rate(mapping: str = "contig") -> float:
    return CU_RATE[mapping]


def phase_us(bytes_per_wg: float, n_wg: int, mapping: str = "contig") -> float:
    """Hardware time of one phase: the floor, then the bytes at the lower of what one workgroup can pull alone
    and its share of the bandwidth the device sustains for transfers of that size."""
    if n_wg <= 0:
        return 0.0
    return FLOOR_US + bytes_per_wg / min(CU_RATE[mapping], agg_bw(bytes_per_wg, mapping) / n_wg)
