"""Lever N, norm_tree: norm_gamma_early plus block_reduce_ordered's exact add DAG in 3 barriers instead of 10.

An alternative to norm_gamma_early, not an addition (applying both is refused).
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.levers.norm_gamma_early import norm_early


def apply(src: Path) -> None:
    norm_early(src, tree=True)
