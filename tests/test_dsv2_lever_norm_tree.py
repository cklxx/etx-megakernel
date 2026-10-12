"""CPU proofs for lever N (norm_tree) on the pinned fleet source; no device required."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.levers import norm_gamma_early as n


def _original_tree(values, add):
    smem = list(values)
    for stride in (128, 64, 32, 16, 8, 4, 2, 1):
        for lane in range(stride):
            smem[lane] = add(smem[lane], smem[lane + stride])
    return smem[0]


def _three_barrier_tree(values, add):
    # These are ORDERED_TREE's lo, hi, root expressions and its six shuffles.
    root = [add(add(values[t], values[t + 128]),
                add(values[t + 64], values[t + 192])) for t in range(64)]
    for stride in (32, 16, 8, 4, 2, 1):
        snapshot = root[:]
        for lane in range(stride):
            root[lane] = add(snapshot[lane], snapshot[lane + stride])
    return root[0]


def test_norm_tree_identical_ordered_expression_dag():
    leaves = list(range(256))
    add = lambda a, b: (a, b)   # order and parentheses both significant
    assert _original_tree(leaves, add) == _three_barrier_tree(leaves, add)
    assert n.ORDERED_TREE.count("__syncthreads();") == 3
    assert "const float lo = smem[lane] + smem[lane + 128];" in n.ORDERED_TREE
    assert "const float hi = smem[lane + 64] + smem[lane + 192];" in n.ORDERED_TREE
    assert "float root = lo + hi;" in n.ORDERED_TREE
    assert "if (lane < stride) root += rhs;" in n.ORDERED_TREE


def test_norm_tree_float32_bitwise_including_nonfinite():
    rng = np.random.default_rng(304)
    random_bits = rng.integers(0, 2**32, size=(128, 256), dtype=np.uint32)
    special = np.array([0, 0x80000000, 0x7f800000, 0xff800000,
                        0x7fc00001, 0xffc00002, 1, 0x80000001], dtype=np.uint32)
    cases = list(random_bits.view(np.float32))
    cases += [np.resize(special, 256).view(np.float32), np.zeros(256, np.float32),
              np.full(256, -0., np.float32)]
    with np.errstate(all="ignore"):
        for values in cases:
            old = _original_tree(values, np.add)
            new = _three_barrier_tree(values, np.add)
            assert old.view(np.uint32) == new.view(np.uint32)


def test_norm_tree_scope_restores_the_pinned_source_byte_for_byte(tmp_path, pinned_src):
    """Removing exactly N's additions (helper copy + ordered tree, two call retargets) gives back the pinned fleet_kernel.hip:
    the dense post-norm, lm_head and every other site keep the original rms_norm_lds."""
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    old_k = (copy / "kernels/fleet_kernel.hip").read_text()
    pf.apply(copy, "norm_tree")
    k = (copy / "kernels/fleet_kernel.hip").read_text()
    helper = "rms_norm_lds_tree_early"
    clone = pf._function(k, f"__device__ inline void {helper}(")
    assert pf._function(old_k, "__device__ inline void rms_norm_lds(") == pf._function(k, "__device__ inline void rms_norm_lds(")
    k = k.replace("\n\n" + n.ORDERED_TREE + clone, "")
    k = k.replace(f"{helper}(d, w.layer_norm_in[L]", "rms_norm_lds(d, w.layer_norm_in[L]")
    router = "rms_norm_lds(d, w.layer_norm_post[layer], s.vec, s.smem);"
    k = k.replace(f"if (moe) {helper}(d, w.layer_norm_post[layer], s.vec, s.smem);\n    else {router}", router)
    assert k == old_k
    assert clone.index("gamma_early[u] =") < clone.index("float local = 0.f") < clone.index("const float ssq")
    assert "bf16_round(bf16_round(v[i] * inv) * gamma_early[" in clone
    assert ("block_reduce_ordered_etx_tree(local, smem)" in clone) == True


def test_norm_variants_reject_combining(tmp_path, pinned_src):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    pf.apply(copy, "norm_tree")
    before = (copy / "kernels/fleet_kernel.hip").read_bytes()
    with pytest.raises(SystemExit, match="alternative patches"):
        pf.apply(copy, "norm_gamma_early")
    assert (copy / "kernels/fleet_kernel.hip").read_bytes() == before
