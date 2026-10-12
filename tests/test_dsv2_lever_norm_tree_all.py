"""CPU proofs for lever N, norm_tree_all: the norm_tree proofs, at every RMSNorm site; no device required."""
from __future__ import annotations

import shutil

import pytest

from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.levers import norm_gamma_early as n
from tests.test_dsv2_lever_norm_tree import (  # noqa: F401  (the same add-DAG proofs run under this lever)
    test_norm_tree_float32_bitwise_including_nonfinite, test_norm_tree_identical_ordered_expression_dag)

SITES = ("w.layer_norm_in[L]", "w.layer_norm_post[layer]", "w.final_norm")


def test_norm_tree_all_reaches_every_site_and_nothing_else(tmp_path, pinned_src):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    old_k = (copy / "kernels/fleet_kernel.hip").read_text()
    pf.apply(copy, "norm_tree_all")
    k = (copy / "kernels/fleet_kernel.hip").read_text()
    helper = "rms_norm_lds_tree_early"
    clone = pf._function(k, f"__device__ inline void {helper}(")
    assert pf._function(old_k, "__device__ inline void rms_norm_lds(") == pf._function(k, "__device__ inline void rms_norm_lds(")
    for site in SITES:
        assert k.count(f"{helper}(d, {site}, s.vec, s.smem);") == 1
        k = k.replace(f"{helper}(d, {site}, s.vec, s.smem);", f"rms_norm_lds(d, {site}, s.vec, s.smem);")
    assert k.replace("\n\n" + n.ORDERED_TREE + clone, "") == old_k
    assert "block_reduce_ordered_etx_tree(local, smem)" in clone


@pytest.mark.parametrize("other", ["norm_tree", "norm_gamma_early"])
def test_norm_tree_all_rejects_the_other_variants(tmp_path, pinned_src, other):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    pf.apply(copy, "norm_tree_all")
    before = (copy / "kernels/fleet_kernel.hip").read_bytes()
    with pytest.raises(SystemExit, match="alternative patches"):
        pf.apply(copy, other)
    assert (copy / "kernels/fleet_kernel.hip").read_bytes() == before
