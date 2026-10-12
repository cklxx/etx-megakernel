"""CPU proofs for lever topk_rank on the pinned fleet source; no device required."""
from __future__ import annotations

import shutil

import numpy as np

from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.levers import topk_rank as tk

LANES, TOP_K = 64, 6


def fleet_select(p):
    """select_experts_from @3fb67a6, transcribed: six xor-butterfly argmax passes, taken lane set to -inf."""
    p = [np.float32(x) for x in p]
    ids, ws = [], []
    for _ in range(TOP_K):
        bv, bi = list(p), list(range(LANES))
        off = LANES // 2
        while off:
            nv, ni = bv[:], bi[:]
            for lane in range(LANES):                       # every lane reads its xor partner's pre-step value
                ov, oi = bv[lane ^ off], bi[lane ^ off]
                if ov > bv[lane] or (ov == bv[lane] and oi < bi[lane]):
                    nv[lane], ni[lane] = ov, oi
            bv, bi = nv, ni
            off >>= 1
        ids.append(bi[0]); ws.append(bv[0])
        p[bi[0]] = np.float32(-np.inf)
    return ids, ws


def rank_select(p):
    """The lever's rank selection, transcribed from RANK_BODY."""
    p = [np.float32(x) for x in p]
    ids, ws = [None] * TOP_K, [None] * TOP_K
    for lane in range(LANES):
        rank = sum(1 for j in range(LANES) if p[j] > p[lane] or (p[j] == p[lane] and j < lane))
        if rank < TOP_K:
            ids[rank], ws[rank] = lane, p[lane]
    return ids, ws


def softmax32(v):
    v = np.asarray(v, np.float32)
    e = np.exp(v - v.max(), dtype=np.float32)
    return e * (np.float32(1) / e.sum(dtype=np.float32))


def test_rank_selection_equals_fleets_six_passes_on_every_nan_free_input():
    rng = np.random.default_rng(64)
    cases = 0
    pools = [np.float32([0, -0.0, 1e-45, 1e-38, .25, .5, 1.0]), np.float32([0.0, 0.0, 0.0, .5]),
             np.float32([1.0, 1.0, 1.0, 1.0])]
    for trial in range(6000):
        kind = trial % 5
        if kind == 0:                                  # router-like: softmax of random fp32 logits
            p = softmax32(rng.normal(0, 2, LANES))
        elif kind == 1:                                # softmax with -inf logits: many exact zeros
            v = rng.normal(0, 3, LANES).astype(np.float32)
            v[rng.random(LANES) < .5] = -np.inf
            p = softmax32(v)
        else:                                          # adversarial ties, +-0, denormals
            p = rng.choice(pools[kind - 2], LANES)
        a, b = fleet_select(p), rank_select(p)
        assert a[0] == b[0], (trial, a, b)
        assert all(x == y and np.signbit(x) == np.signbit(y) for x, y in zip(a[1], b[1])), (trial, a, b)
        cases += 1
    assert cases == 6000


def test_rank_body_matches_its_transcription_and_keeps_the_softmax_lines():
    lines = [l.strip() for l in tk.RANK_BODY.splitlines()]
    for want in ("float p = __expf(v - mx);", "const float inv = 1.f / wave_sum(p);", "p *= inv;",
                 "for (int j = 0; j < kWaveLanes; ++j) {",
                 "rank += (pj > p || (pj == p && j < lane)) ? 1 : 0;",
                 "if (rank < d.top_k) { ids[rank] = lane; ws[rank] = p; }   // unnormalised, as fleet writes it"):
        assert want in lines, want


def test_topk_rank_keeps_the_original_as_the_fallback(tmp_path, pinned_src):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    old = (copy / "kernels/fleet_kernel.hip").read_text()
    pf.apply(copy, "topk_rank")
    new = (copy / "kernels/fleet_kernel.hip").read_text()
    orig = pf._function(old, tk.ORIGINAL_SIGNATURE)
    ref = pf._function(new, "__device__ inline void select_experts_from_ref(const ModelDims& d, float v,")
    assert ref == orig.replace("void select_experts_from(", "void select_experts_from_ref(", 1)
    # The softmax lines of the original are the rank body's, byte for byte.
    for line in ("mx = fmaxf(mx, __shfl_xor(mx, off, kWaveLanes));", "const float inv = 1.f / wave_sum(p);", "p *= inv;"):
        assert line in orig
    assert sorted(p.name for p in (copy / "kernels").iterdir()
                  if p.read_bytes() != (pinned_src / "kernels" / p.name).read_bytes()) == ["fleet_kernel.hip"]
