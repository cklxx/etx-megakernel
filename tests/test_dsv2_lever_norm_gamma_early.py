"""CPU proofs for lever N (norm_gamma_early) on the pinned fleet source; no device required."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.levers import norm_gamma_early as n


def test_early_gamma_clamps_only_unused_slots():
    # Actual addressing is lane + 256*u, clamped to hidden-1. Padded loads
    # remain in-bounds; every epilogue-visible gamma is exactly the original.
    indices = np.arange(2048).reshape(8, 256)
    for hidden in range(1, 2049):
        loaded = np.minimum(indices, hidden - 1)
        consumed = indices < hidden
        assert np.array_equal(loaded[consumed], indices[consumed])
        assert np.all((loaded >= 0) & (loaded < hidden))
    assert "min((int)threadIdx.x + u * kBlockThreads, d.hidden - 1)" in n.GAMMA_EARLY
    assert "if (i < d.hidden)" not in n.GAMMA_EARLY
    assert "d.hidden != kMaxHidden" not in n.GAMMA_EARLY


def test_norm_gamma_early_scope_restores_the_pinned_source_byte_for_byte(tmp_path, pinned_src):
    """Removing exactly N's additions (helper copy, two call retargets) gives back the pinned fleet_kernel.hip:
    the dense post-norm, lm_head and every other site keep the original rms_norm_lds."""
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    old_k = (copy / "kernels/fleet_kernel.hip").read_text()
    pf.apply(copy, "norm_gamma_early")
    k = (copy / "kernels/fleet_kernel.hip").read_text()
    helper = "rms_norm_lds_gamma_early"
    clone = pf._function(k, f"__device__ inline void {helper}(")
    assert pf._function(old_k, "__device__ inline void rms_norm_lds(") == pf._function(k, "__device__ inline void rms_norm_lds(")
    k = k.replace("\n\n" + "" + clone, "")
    k = k.replace(f"{helper}(d, w.layer_norm_in[L]", "rms_norm_lds(d, w.layer_norm_in[L]")
    router = "rms_norm_lds(d, w.layer_norm_post[layer], s.vec, s.smem);"
    k = k.replace(f"if (moe) {helper}(d, w.layer_norm_post[layer], s.vec, s.smem);\n    else {router}", router)
    assert k == old_k
    assert clone.index("gamma_early[u] =") < clone.index("float local = 0.f") < clone.index("const float ssq")
    assert "bf16_round(bf16_round(v[i] * inv) * gamma_early[" in clone
    assert ("block_reduce_ordered_etx_tree(local, smem)" in clone) == False


def test_norm_variants_reject_combining(tmp_path, pinned_src):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    pf.apply(copy, "norm_gamma_early")
    before = (copy / "kernels/fleet_kernel.hip").read_bytes()
    with pytest.raises(SystemExit, match="alternative patches"):
        pf.apply(copy, "norm_tree")
    assert (copy / "kernels/fleet_kernel.hip").read_bytes() == before
