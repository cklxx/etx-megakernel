"""CPU emulation of the softmax top-k of one wave: the old selection and the ETX_TOPK_KEY32 selection.

`ref_topk` follows dsl_topk_lds (etx/ops/gemv.py) and the softmax path of etx/ops/topk.py: float32, 64 lanes, 2
logits per lane, m = fmaxf, p = expf(l - m), s = etx_wave_sum (xor butterfly), p *= 1 / s, K rounds of a 6-step
xor-butterfly argmax over p (the lower index wins ties), renorm: weight / (weights summed in round order).

`key_topk` follows etx_topk_key32 (etx/runtime/include/etx/tile_util.h) line by line: one 32-bit key per expert,
a DPP wave max per round (row_shr 1, 2, 4, 8, row_bcast:15 with row_mask 0xa, row_bcast:31 with row_mask 0xc, then
lane 63), m from the round-0 key, the weights from the winner lane's p.

numpy's float32 exp is not the device expf. Both functions use the same exp, so a comparison of the two checks the
selection and the order of the arithmetic, not the last bit of expf. See docs/core/TOPK-KEY32-2026-10-08.md.
"""
from __future__ import annotations

import numpy as np

W = 64                 # legacy default; emulators also accept wave_size=32
EMAX = 128             # fixed expert capacity; the low key field has 7 bits
f32 = np.float32


def bf16_bits(x) -> np.ndarray:
    """float32 -> bf16 bits (uint16), round to nearest even; NaN stays NaN."""
    a = np.asarray(x, np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(a), ((u >> 16) | 0x40).astype(np.uint16), r)


def bf16_to_f32(b) -> np.ndarray:
    return (np.asarray(b, np.uint32) << 16).astype(np.uint32).view(np.float32)


def order16(x) -> np.ndarray:
    """etx_key32_order: offset binary 0x8000 +- magnitude; -0 -> 0x8000 (as +0); NaN -> 0 (below -inf = 0x80)."""
    x = np.asarray(x, np.float32)
    mag = (x.view(np.uint32) >> 16) & 0x7FFF
    with np.errstate(invalid="ignore"):
        u = np.where(x < 0, 0x8000 - mag, 0x8000 + mag)
    return np.where(np.isnan(x), 0, u).astype(np.uint32)


def key32(x, e) -> np.ndarray:
    """etx_key32: bit 23 valid | order16 << 7 | (127 - e)."""
    e = np.asarray(e)
    if np.any(e >= EMAX) or np.any(e < 0):
        raise ValueError("ETX_TOPK_KEY32 needs 0 <= e < 128")
    return (np.uint32(0x800000) | (order16(x) << 7) | (127 - e).astype(np.uint32)).astype(np.uint32)


def key32_logit(key: int) -> np.float32:
    """etx_key32_logit: the inverse of order16; NaN for order 0."""
    u = (int(key) >> 7) & 0xFFFF
    if u == 0:
        return np.uint32(0x7FC00000).view(np.float32)
    b = u - 0x8000 if u >= 0x8000 else 0x8000 | (0x8000 - u)
    return np.uint32(b << 16).view(np.float32)


def key32_expert(key: int) -> int:
    return 127 - (int(key) & 127) if key else 0


def dpp_wave_max(x) -> int:
    """etx_wave_max_u32: row_shr 1/2/4/8 with bound_ctrl (invalid source -> 0), row_bcast:15 (row_mask 0xa),
    row_bcast:31 (row_mask 0xc), v_readlane_b32 lane 63."""
    x = np.asarray(x, np.int64).copy()
    if len(x) == 32:
        return int(x.max())  # warp32 unsigned max primitive
    lane = np.arange(W)
    pos, row = lane % 16, lane // 16
    for n in (1, 2, 4, 8):
        x = np.maximum(x, np.where(pos >= n, x[np.clip(lane - n, 0, W - 1)], 0))
    x = np.where(np.isin(row, [1, 3]), np.maximum(x, x[np.clip(row * 16 - 1, 0, W - 1)]), x)
    x = np.where(np.isin(row, [2, 3]), np.maximum(x, x[31]), x)
    return int(x[63])


def wave_sum(v) -> np.float32:
    """etx_wave_sum: v += shfl_xor(v, off) for off = 32 .. 1; every lane ends with the same value."""
    v = np.asarray(v, np.float32).copy()
    W = len(v)
    lane = np.arange(W)
    off = W // 2
    while off:
        v = (v + v[lane ^ off]).astype(np.float32)
        off >>= 1
    return f32(v[0])


def _lanes(bits, wave_size=64):
    if wave_size not in (32, 64):
        raise ValueError("wave_size must be 32 or 64")
    W = wave_size
    E = len(bits)
    if not 1 <= E <= EMAX:
        raise ValueError("E must be in 1..128")
    lane = np.arange(W)
    v = bf16_to_f32(bits)
    l = np.full((EMAX // W, W), -np.inf, f32)
    ok = np.zeros((EMAX // W, W), bool)
    for k in range(len(l)):
        e = lane + k * W
        ok[k] = e < E
        l[k, ok[k]] = v[e[ok[k]]]
    return lane, l, ok


def _softmax(l, ok, m):
    with np.errstate(all="ignore"):
        p = np.where(ok, np.exp((l - m).astype(f32)).astype(f32), f32(0))
        # Accumulate each lane's two/four logits in source order, then butterfly.
        s = wave_sum(np.add.accumulate(p, axis=0, dtype=f32)[-1])
        inv = f32(f32(1) / s)
        return np.where(ok, (p * inv).astype(f32), f32(0))


def _renorm(wts, renorm):
    if not renorm:
        return wts
    sel = f32(0)
    for w in wts:
        sel = f32(sel + w)
    with np.errstate(all="ignore"):
        return [f32(w / sel) for w in wts]


def ref_topk(bits, K: int, renorm: bool, wave_size=64):
    """The old path. ids 0x7fffffff where no lane passes `p > bv` (NaN rows: the existing bug)."""
    W = wave_size
    lane, l, ok = _lanes(bits, wave_size)
    m = f32(np.fmax.reduce(l.reshape(-1)))        # fmaxf ignores NaN
    p = _softmax(l, ok, m)
    taken = ~ok
    ids, wts = [], []
    for _ in range(K):
        bv = np.full(W, f32(-1)); bi = np.full(W, 0x7FFFFFFF, np.int64)
        for k in range(len(l)):
            take = (~taken[k]) & (p[k] > bv)
            bv = np.where(take, p[k], bv); bi = np.where(take, lane + k * W, bi)
        off = W // 2
        while off:
            ov, oi = bv[lane ^ off], bi[lane ^ off]
            take = (ov > bv) | ((ov == bv) & (oi < bi))
            bv = np.where(take, ov, bv); bi = np.where(take, oi, bi)
            off >>= 1
        b = int(bi[0])
        for k in range(len(l)):
            taken[k] |= (lane + k * W) == b
        ids.append(b); wts.append(f32(bv[0]))
    return ids, _renorm(wts, renorm)


def key_topk(bits, K: int, renorm: bool, wave_size=64):
    """etx_topk_key32."""
    W = wave_size
    if not 1 <= K <= W:
        raise ValueError(f"K must be in 1..{W}")
    lane, l, ok = _lanes(bits, wave_size)
    key = np.zeros((EMAX // W, W), np.int64)
    for k in range(len(l)):
        e = lane + k * W
        key[k, ok[k]] = key32(l[k, ok[k]], e[ok[k]])
    km = dpp_wave_max(np.maximum.reduce(key, axis=0))
    key[key == km] = 0
    m = key32_logit(km)
    p = _softmax(l, ok, m)
    ids, wts = [], []
    for t in range(K):
        if t:
            km = dpp_wave_max(np.maximum.reduce(key, axis=0))
            key[key == km] = 0
        bi = key32_expert(km)
        ids.append(bi); wts.append(f32(p[bi // W, bi % W]))    # v_readlane of the winner lane's p
    return ids, _renorm(wts, renorm)


def tie_explained(bits, K, renorm, ref, key) -> bool:
    """True when ids differ only where the old path cannot tell experts apart: equal float p (an exp underflow or a
    rounding tie), or a NaN p (the old path then writes 0x7fffffff). The weight lists must then still agree."""
    lane, l, ok = _lanes(bits)
    m = f32(np.fmax.reduce(l.reshape(-1)))
    p = _softmax(l, ok, m).reshape(-1)[:len(bits)]
    if np.isnan(p).any():
        return True
    for a, b in zip(ref[0], key[0]):
        if a != b and p[a] != p[b]:
            return False
    return all(x == y for x, y in zip(ref[1], key[1]))
