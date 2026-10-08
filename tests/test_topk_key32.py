"""ETX_TOPK_KEY32: the key mapping, the DPP wave max model, and the selection against the old path (CPU only).
See docs/TOPK-KEY32-2026-10-08.md and etx/ops/topk_key32.py."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from etx.ops import topk_key32 as tk

ROOT = Path(__file__).resolve().parents[1]
ALL_BF16 = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)


def _same(a, b):
    return a[0] == b[0] and all((x == y) or (np.isnan(x) and np.isnan(y)) for x, y in zip(a[1], b[1]))


def test_order16_is_the_float_order_for_every_bf16():
    x = tk.bf16_to_f32(ALL_BF16)
    fin = ~np.isnan(x)
    u = tk.order16(x)
    xs, us = x[fin], u[fin].astype(np.int64)
    o = np.argsort(xs, kind="stable")
    assert np.all(np.diff(us[o]) >= 0)                      # monotone
    # equal keys exactly for equal floats: only -0 and +0 share a value
    assert len(np.unique(us)) == len(np.unique(xs))
    assert tk.order16(np.float32(-0.0)) == tk.order16(np.float32(0.0)) == 0x8000
    assert np.all(u[~fin] == 0)                             # NaN: below everything
    assert tk.order16(np.float32(-np.inf)) == 0x80 and tk.order16(np.float32(np.inf)) == 0xFF80
    assert u.max() <= 0xFFFF


def test_key_logit_roundtrip_and_index_field():
    x = tk.bf16_to_f32(ALL_BF16)
    for e in (0, 1, 63, 64, 127):
        keys = tk.key32(x, np.full(x.shape, e))
        assert np.all(keys & 0x800000) and np.all(keys < (1 << 24))
        assert all(tk.key32_expert(int(k)) == e for k in keys[::997])
        for xv, k in zip(x[::131], keys[::131]):
            back = tk.key32_logit(int(k))
            if np.isnan(xv):
                assert np.isnan(back)
            else:
                assert back == xv                           # -0 comes back as +0 (== holds)
    # equal logits: the smaller index has the larger key (vLLM: the lower index wins ties)
    assert tk.key32(np.float32(1.5), 3) > tk.key32(np.float32(1.5), 4)
    assert tk.key32(np.float32(-0.0), 9) > tk.key32(np.float32(0.0), 10)
    with pytest.raises(ValueError):
        tk.key32(np.float32(1.0), 128)


def test_dpp_wave_max_model():
    rng = np.random.default_rng(3)
    for _ in range(200):
        x = rng.integers(0, 1 << 24, tk.W)
        assert tk.dpp_wave_max(x) == x.max()
    for lane in range(tk.W):                                 # a single nonzero lane reaches lane 63 from anywhere
        x = np.zeros(tk.W, np.int64); x[lane] = 5
        assert tk.dpp_wave_max(x) == 5


def _rows(rng, E, n):
    for _ in range(n):
        yield tk.bf16_bits(rng.normal(0, 3, E).astype(np.float32))
    for _ in range(n // 2):                                  # heavy ties
        yield tk.bf16_bits(rng.integers(-2, 3, E).astype(np.float32))
    for _ in range(n // 4):                                  # wide range, negatives, -inf, large negative
        v = rng.normal(0, 30, E).astype(np.float32)
        v[rng.random(E) < 0.1] = -np.inf
        v[rng.random(E) < 0.1] = -1e30
        yield tk.bf16_bits(v)


@pytest.mark.parametrize("E,K", [(128, 8), (64, 6), (8, 2), (128, 1), (128, 16)])
def test_selection_matches_the_old_path(E, K):
    rng = np.random.default_rng(E * 100 + K)
    exact = 0
    for bits in _rows(rng, E, 120):
        for renorm in (False, True):
            r, k = tk.ref_topk(bits, K, renorm), tk.key_topk(bits, K, renorm)
            if _same(r, k):
                exact += 1
            else:                                            # only where the old path sees equal p (underflow)
                assert tk.tie_explained(bits, K, renorm, r, k), (bits.tolist(), r, k)
    assert exact > 0.9 * 2 * (120 + 60 + 30)


def test_plus_minus_zero_ties_pick_the_lowest_indices():
    z = np.zeros(128, np.float32); z[::2] = -0.0
    bits = tk.bf16_bits(z)
    assert tk.key_topk(bits, 8, False)[0] == list(range(8)) == tk.ref_topk(bits, 8, False)[0]


def test_one_nan_logit():
    rng = np.random.default_rng(5)
    v = rng.normal(0, 3, 128).astype(np.float32); v[5] = np.nan
    bits = tk.bf16_bits(v)
    ref, key = tk.ref_topk(bits, 8, True), tk.key_topk(bits, 8, True)
    assert ref[0] == [0x7FFFFFFF] * 8                        # the existing bug of the switch-off path (documented)
    fin = np.where(np.isnan(v), -np.inf, v)
    assert key[0] == list(np.argsort(-fin, kind="stable")[:8])  # the key path: the 8 largest finite logits
    assert all(np.isnan(w) for w in key[1])                  # s is NaN in both paths, so the weights are NaN


def test_exp_underflow_tie_difference():
    """exp(l - m) == 0 for l - m < about -104: the old path ties all such experts at p = 0 and takes the smallest
    indices; the key path takes the largest logits. The weights are 0 in both paths."""
    u = np.full(128, -200.0, np.float32); u[:3] = 0.0; u[50] = -150.0
    bits = tk.bf16_bits(u)
    ref, key = tk.ref_topk(bits, 8, True), tk.key_topk(bits, 8, True)
    assert ref[0][:3] == key[0][:3] == [0, 1, 2]
    assert ref[0][3:] == [3, 4, 5, 6, 7] and key[0][3] == 50
    assert ref[1] == key[1] and all(w == 0 for w in key[1][3:])


def _key32_block() -> str:
    src = (ROOT / "etx/runtime/include/etx/tile_util.h").read_text()
    m = re.search(r"^#ifdef ETX_TOPK_KEY32\n(.*?)^#endif\n", src, re.S | re.M)
    assert m, "ETX_TOPK_KEY32 block missing"
    return m.group(1)


def _guarded_by(src: str, i: int, macro: str) -> bool:
    """True when offset i of src is inside an `#ifdef macro` branch (nesting followed; #else flips it)."""
    stack = []
    for m in re.finditer(r"^#\s*(ifdef|ifndef|if|else|elif|endif)\b[ \t]*(\w*)", src[:i], re.M):
        d, name = m.group(1), m.group(2)
        if d in ("ifdef", "ifndef", "if"):
            stack.append((d, name))
        elif d == "endif":
            stack.pop()
        else:
            stack[-1] = ("else", stack[-1][1])
    return ("ifdef", macro) in stack


def test_switch_guards_every_key32_line():
    """With the switch off, the preprocessor removes every new line: the off build is the old source."""
    for path, needle in (("etx/ops/topk.py", "etx_topk_key32("), ("etx/ops/gemv.py", "etx_topk_key32(p, E, K, renorm != 0")):
        src = (ROOT / path).read_text()
        assert _guarded_by(src, src.index(needle), "ETX_TOPK_KEY32")
    blk = _key32_block()
    assert "__builtin_amdgcn_update_dpp" in blk and "asm" not in re.sub(r"//.*", "", blk)
    assert "ds_bpermute" not in re.sub(r"//.*", "", blk)


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++ (C++20) for the host SIMT harness")
def test_cpp_block_on_a_host_simt_wave(tmp_path):
    """Runs the C++ of the ETX_TOPK_KEY32 block itself (64 host threads = 64 lanes, modelled DPP and readlane), both
    output forms, and compares it with the Python emulation. Host exp is not numpy exp: ids must match, weights to 1 ulp-ish."""
    (tmp_path / "key32_block.inc").write_text(_key32_block())
    exe = tmp_path / "simt"
    subprocess.run(["g++", "-std=c++20", "-O1", "-pthread", "-I", str(tmp_path), str(ROOT / "tests/data/topk_key32_simt.cpp"),
                    "-o", str(exe)], check=True, capture_output=True)
    rng = np.random.default_rng(11)
    cases = []
    for E, K in ((128, 8), (64, 6), (8, 2)):
        for bits in list(_rows(rng, E, 8)):
            cases.append((E, K, 1, bits))
    z = np.zeros(128, np.float32); z[1::2] = -0.0
    cases.append((128, 8, 0, tk.bf16_bits(z)))
    v = rng.normal(0, 3, 128).astype(np.float32); v[7] = np.nan
    cases.append((128, 8, 1, tk.bf16_bits(v)))
    inp = "".join(f"{E} {K} {rn}\n" + " ".join(f"{int(b):x}" for b in bits) + "\n" for E, K, rn, bits in cases)
    outs = {}
    for variant in ("put", "lds"):                       # etx_topk_key32 (dsl_run_dsl_topk), etx_topk_key32_lds (fold)
        outs[variant] = subprocess.run([str(exe), variant], input=inp, capture_output=True, text=True, check=True,
                                       timeout=120).stdout
        out = outs[variant].split("\n")
        i = 0
        for E, K, rn, bits in cases:
            ids, ws = [], []
            for _ in range(K):
                a, b = out[i].split(); i += 1
                ids.append(int(a)); ws.append(np.uint32(int(b, 16)).view(np.float32))
            exp_ids, exp_w = tk.key_topk(bits, K, bool(rn))
            assert ids == exp_ids, (variant, E, K, bits.tolist())
            for x, y in zip(ws, exp_w):
                assert (np.isnan(x) and np.isnan(y)) or abs(float(x) - float(y)) <= 4e-7 * max(abs(float(y)), 1e-30)
    assert outs["put"] == outs["lds"]                    # the two output forms give the same bits


def test_rounding_tie_difference():
    """Two logits with |l| < about 2^-17 |m| can give the same float l - m, so the old path sees equal p and takes
    the lower index; the key path takes the larger logit. The two weights are equal."""
    v = np.array([20, -30, -30, -30, -30, -30, -30, -30], np.float32)
    bits = tk.bf16_bits(v)
    bits[1] = tk.bf16_bits(np.float32(1e-6)); bits[2] = bits[1] + 1      # adjacent bf16 values, l2 > l1
    ref, key = tk.ref_topk(bits, 2, True), tk.key_topk(bits, 2, True)
    assert ref[0] == [0, 1] and key[0] == [0, 2] and ref[1] == key[1]
    assert tk.tie_explained(bits, 2, True, ref, key)


def test_sparsemixer_mode_is_unchanged_by_the_switch():
    """Only modes 0 and 1 take the key path. The key call comes after the mode-2 branch returns, and the switch
    removes only `m = etx_wave_max(m)` before that branch; the mode-2 code never reads m."""
    src = (ROOT / "etx/ops/topk.py").read_text()
    blk = src[src.index("if (mode == 2) {"):]
    blk = blk[:blk.index("    return;\n  }\n") + len("    return;\n  }\n")]
    assert not re.search(r"\bm\b", re.sub(r"//.*", "", blk))
    assert src.index("etx_topk_key32(") > src.index("if (mode == 2) {") + len(blk)
    guard = src[src.rfind("#ifndef ETX_TOPK_KEY32", 0, src.index("if (mode == 2) {")):src.index("if (mode == 2) {")]
    assert re.sub(r"//.*", "", guard).split() == ["#ifndef", "ETX_TOPK_KEY32", "m", "=", "etx_wave_max(m);", "#endif"]


def test_key32_build_guards():
    """K <= 64: lane t keeps round t's result, so host.hip refuses TOPK > 64 when the switch is on (E <= 128 is
    already checked). ETX_ABL_TOPK_NOSTORE does not act on the key path, so that pair of switches does not compile."""
    host = (ROOT / "examples/vllm_llm/host.hip").read_text()
    assert _guarded_by(host, host.index('if (TOPK > 64) die("ETX_TOPK_KEY32'), "ETX_TOPK_KEY32")
    src = (ROOT / "etx/ops/topk.py").read_text()
    blk = src[src.index("#ifdef ETX_TOPK_KEY32        //"):src.index("etx_topk_key32(")]
    assert re.search(r"#ifdef ETX_ABL_TOPK_NOSTORE\n#error [^\n]*\n#endif", blk)
