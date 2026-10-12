"""Lever N, norm_tree_all: norm_tree at every RMSNorm site: qkv input norm, the post-attention norm of every layer
(MoE router and the dense layer 0) and lm_head's final norm. An alternative to norm_tree / norm_gamma_early.
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.levers.norm_gamma_early import norm_early


def apply(src: Path) -> None:
    norm_early(src, tree=True, all_sites=True)
