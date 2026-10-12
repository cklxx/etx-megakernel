"""CPU proofs for lever argmax_tree on the pinned fleet source; no device required."""
from __future__ import annotations

import math
import random
import shutil

from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.levers import argmax_tree as am


def pick(a, b):
    """fleet's comparison: other wins if larger, or equal with a lower index (C float semantics: NaN never wins)."""
    (av, ai), (bv, bi) = a, b
    return b if (bv > av or (bv == av and bi < ai)) else a


def original_tree(x):
    s = list(x)
    st = 128
    while st:
        for t in range(st):
            s[t] = pick(s[t], s[t + st])
        st >>= 1
    return s[0]


def lever_tree(x):
    lanes = [pick(pick(x[t], x[t + 128]), pick(x[t + 64], x[t + 192])) for t in range(64)]
    for s in (32, 16, 8, 4, 2, 1):
        snap = lanes[:]                                   # a shuffle reads every lane's value before any update
        for t in range(64):
            other = snap[t + s] if t + s < 64 else snap[t]   # __shfl_down past the wave returns the lane's own value
            if t < s:
                lanes[t] = pick(lanes[t], other)
    return lanes[0]


def same(a, b):
    return a[1] == b[1] and (a[0] == b[0] or (math.isnan(a[0]) and math.isnan(b[0])))


def test_same_winner_for_every_input_including_ties_nan_and_infinities():
    rng = random.Random(942)
    pool = [0.0, -0.0, 1.0, -1.0, 2.5, float("inf"), float("-inf"), float("nan")]
    cases = 0
    for trial in range(20000):
        mode = trial % 4
        if mode == 0:     # dense ties
            vals = [rng.choice(pool[:5]) for _ in range(256)]
        elif mode == 1:   # with NaN / infinities
            vals = [rng.choice(pool) for _ in range(256)]
        elif mode == 2:   # random floats
            vals = [rng.uniform(-8, 8) for _ in range(256)]
        else:             # fleet's padding: -inf candidates with the int max index
            vals = [rng.choice([float("-inf"), rng.uniform(-1, 1)]) for _ in range(256)]
        idxs = [rng.randrange(0, 102400) if rng.random() < .9 else 0x7FFFFFFF for _ in range(256)]
        x = list(zip(vals, idxs))
        assert same(original_tree(x), lever_tree(x)), (trial, original_tree(x), lever_tree(x))
        cases += 1
    assert cases == 20000


def test_argmax_tree_only_rewrites_the_two_reductions(tmp_path, pinned_src):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    old = (copy / "kernels/fleet_kernel.hip").read_text()
    pf.apply(copy, "argmax_tree")
    new = (copy / "kernels/fleet_kernel.hip").read_text()
    assert new.replace(am.HELPER, "", 1).replace(am.CAND_NEW, am.CAND_OLD).replace(am.TASK_NEW, am.TASK_OLD) == old
    assert sorted(p.name for p in (copy / "kernels").iterdir() if p.read_bytes() != (pinned_src / "kernels" / p.name).read_bytes()) == ["fleet_kernel.hip"]


def test_the_emulation_matches_the_device_helper_line_for_line():
    """lever_tree above transcribes etx_argmax_tree; these are the lines it transcribes, in order."""
    lines = [l.strip() for l in am.HELPER.splitlines()]
    want = ["if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }",
            "float a = vals[t], b = vals[t + 64];",
            "int ai = idxs[t], bi = idxs[t + 64];",
            "etx_argmax_pick(a, ai, vals[t + 128], idxs[t + 128]);",
            "etx_argmax_pick(b, bi, vals[t + 192], idxs[t + 192]);",
            "etx_argmax_pick(a, ai, b, bi);",
            "for (int s = 32; s; s >>= 1) {",
            "const float ov = __shfl_down(a, s, 64);",
            "const int oi = __shfl_down(ai, s, 64);",
            "if (t < s) etx_argmax_pick(a, ai, ov, oi);"]
    positions = [lines.index(w) for w in want]
    assert positions == sorted(positions)
