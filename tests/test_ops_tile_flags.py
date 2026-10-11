"""The flag-gated DSL tile changes of docs/TILES-2026-10-07.md (ETX_DSL_RBATCH, ETX_DSL_SCALAR, ETX_DSL_WSUM,
ETX_DSL_GU1). Each is meant to leave every output bit unchanged; these tests pin the arguments for that claim that can
be checked off the GPU: the lane-0 reduction tree, the gate/up single-pass condition, and that a build without the
flags compiles the original text."""
import re

import numpy as np

from etx.ops import gemv

FLAGS = ("ETX_DSL_RBATCH", "ETX_DSL_SCALAR", "ETX_DSL_WSUM", "ETX_DSL_GU1")


def _butterfly(v):
    """etx_wave_sum: v += shfl_xor(v, off) for off = 32 .. 1, every lane."""
    v = v.copy()
    lanes = np.arange(64)
    for off in (32, 16, 8, 4, 2, 1):
        v = (v + v[lanes ^ off]).astype(np.float32)
    return v


def _wave_sum0(v):
    """dsl_wave_sum0 with each cross-lane step modelled by its ISA semantics (NaN where a lane reads nothing valid)."""
    v = v.copy()
    lanes = np.arange(64)
    nan = np.float32(np.nan)

    def step(src_lane):
        nonlocal v
        other = np.array([v[s] if s is not None else nan for s in src_lane], dtype=np.float32)
        v = (v + other).astype(np.float32)

    step([l ^ 32 for l in lanes])                                          # ds_bpermute, address (lane ^ 32) * 4
    step([(l & ~31) | ((l & 31) ^ 16) for l in lanes])                     # ds_swizzle 0x401f: and 0x1f, xor 0x10
    step([(l & ~15) | ((l + 8) & 15) for l in lanes])                      # DPP row_ror:8
    step([l + 4 if (l & 15) < 12 else None for l in lanes])                # DPP row_shl:4 (lane i reads i + 4)
    step([(l & ~3) | [2, 3, 0, 1][l & 3] for l in lanes])                  # quad_perm [2,3,0,1]
    step([(l & ~3) | [1, 0, 3, 2][l & 3] for l in lanes])                  # quad_perm [1,0,3,2]
    return v


def test_wave_sum0_lane0_is_bit_identical_to_the_butterfly():
    rng = np.random.default_rng(7)
    for t in range(2000):
        scale = np.float32(10.0) ** rng.integers(-6, 6, 64)                 # wide dynamic range: rounding matters
        v = (rng.standard_normal(64) * scale).astype(np.float32)
        a, b = _butterfly(v)[0], _wave_sum0(v)[0]
        assert a.view(np.uint32) == b.view(np.uint32), (t, a, b)


def test_a_different_tree_would_be_caught():
    """The bit-identity test has power: rocprim's order (1, 2, 4, ... first) differs on lane 0 for some inputs."""
    rng = np.random.default_rng(7)
    lanes = np.arange(64)
    differ = 0
    for _ in range(500):
        v = (rng.standard_normal(64) * np.float32(10.0) ** rng.integers(-6, 6, 64)).astype(np.float32)
        w = v.copy()
        for off in (1, 2, 4, 8, 16, 32):
            w = (w + w[lanes ^ off]).astype(np.float32)
        differ += w[0].view(np.uint32) != _butterfly(v)[0].view(np.uint32)
    assert differ > 50


def _owners(nrt, spr, nw):
    """dsl_gemv_rows_nb's split: row -> list of (wave, first segment, last segment) that touch it."""
    S = nrt * spr
    per = (S + nw - 1) // nw
    rows = {}
    for wv in range(nw):
        sb, se = wv * per, min(S, wv * per + per)
        for sg in range(sb, se):
            rows.setdefault(sg // spr, []).append(wv)
    return rows, per


def _gu1_ok(n, K, nw=16):
    """The ETX_DSL_GU1 condition exactly as the silu_mul body writes it."""
    spr = K // 512
    p1 = (n * spr + nw - 1) // nw
    p2 = (2 * n * spr + nw - 1) // nw
    return K % 512 == 0 and 2 * n <= 64 and p1 % spr == 0 and p2 % spr == 0 and (n * spr) % p2 == 0


def test_gu1_condition_implies_whole_rows_and_no_straddle():
    taken = 0
    for K in range(512, 4097, 512):
        for n in range(1, 33):
            if not _gu1_ok(n, K):
                continue
            taken += 1
            spr = K // 512
            two, _ = _owners(n, spr, 16)
            one, per2 = _owners(2 * n, spr, 16)
            for r, ws in list(two.items()) + list(one.items()):
                assert len(set(ws)) == 1, (n, K, r, ws)                     # each row is one wave's in-order sum
            gate_waves = {w for r, ws in one.items() if r < n for w in ws}
            up_waves = {w for r, ws in one.items() if r >= n for w in ws}
            assert not gate_waves & up_waves                                # one base pointer per wave
            for w in gate_waves | up_waves:                                 # and the device's base test agrees
                assert (w * per2 >= n * spr) == (w in up_waves)
    assert taken >= 8


def test_gu1_applies_to_qwen3_30b_a3b_moe_up():
    # moe_up: MI = 768 rows per expert, 24 tasks per expert (model.py: TOPK * MI // 32 tasks) -> 32 rows, K = H = 2048
    assert _gu1_ok(32, 2048)


def test_flags_are_preprocessor_gated():
    src = gemv.emit_tiles()
    for f in FLAGS:
        assert f"#ifdef {f}" in src, f
    # without the flags the original text is what compiles
    assert "const int row = sg / spr;" in src
    assert "for (int s = 0; s < R; ++s) {" in src
    assert "const int R = FixedR >= 0 ? FixedR : d.R;" in src
    assert "const float v = etx_wave_sum(acc[b]);" in src
    assert re.search(r"#else\n  const int nblk = \*\(const int\*\)\(a \+ 156\);\n#endif", src)
    # the scalar argument reads use the constant address space, as etx_imp_argblock's callers do
    assert "#define DSL_ARG_AS __attribute__((address_space(4)))" in src
