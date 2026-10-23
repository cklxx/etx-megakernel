"""Phase-0 CPU simulator studies (not hardware measurements), design §15 items 6/7.

Run: .venv/bin/python bench/sim_experiments.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etx.external import run_protocol_study  # noqa: E402


def study_variance(size=256) -> None:
    print(f"study 6: tile-duration variance vs schedule (split-K, n={size} -> {size * 5} tasks, gfx942 costs)")
    print(f"{'cv':>5} {'static':>9} {'dynamic':>9} {'hybrid':>9}   best")
    # One row per call retains the historical incremental output.
    for cv in (0.0, 0.1, 0.3, 0.6, 1.0):
        row = run_protocol_study('variance', values=(cv,), size=size)['rows'][0]
        res = row['makespan_us']
        print(f"{cv:5.2f} {res['static']:9.1f} {res['dynamic']:9.1f} {res['hybrid']:9.1f}   {row['best']}")


def study_routing_skew(size=32) -> None:
    print(f"\nstudy 7: MoE routing imbalance vs schedule (B={size} tokens, 8 experts, gfx942 costs)")
    print(f"{'skew':>5} {'max/mean':>9} {'static':>9} {'dynamic':>9} {'hybrid':>9}   best")
    for skew in (0.0, 0.5, 1.0, 2.0, 4.0):
        row = run_protocol_study('routing-skew', values=(skew,), size=size)['rows'][0]
        res = row['makespan_us']
        print(f"{skew:5.1f} {row['max_mean']:9.2f} {res['static']:9.1f} {res['dynamic']:9.1f} {res['hybrid']:9.1f}   {row['best']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, help='explicit reduced study size; defaults stay n=256/B=32')
    args = parser.parse_args()
    study_variance(256 if args.size is None else args.size)
    study_routing_skew(32 if args.size is None else args.size)
