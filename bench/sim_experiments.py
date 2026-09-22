"""Simulator studies for the two phase-0 items that are model experiments, not
hardware measurements (design §15, microbenchmarks 6 and 7):

  6. static queue length vs straggler time: how the static schedule's makespan
     degrades as tile-duration variance grows, against dynamic and hybrid
  7. routing imbalance vs dynamic-scheduling gain: MoE layer with increasingly
     skewed expert routing

Uses the calibrated gfx942 cost table.  Run:  .venv/bin/python bench/sim_experiments.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etx.passes import PassOptions, compile_graph          # noqa: E402
from etx.sim import simulate                                # noqa: E402
from examples import moe_layer, splitk_sum                  # noqa: E402


def study_variance() -> None:
    print("study 6: tile-duration variance vs schedule (split-K, n=256 -> 1280 tasks, gfx942 costs)")
    print(f"{'cv':>5} {'static':>9} {'dynamic':>9} {'hybrid':>9}   best")
    for cv in (0.0, 0.1, 0.3, 0.6, 1.0):
        res = {}
        for mode in ("static", "dynamic", "hybrid"):
            g = splitk_sum.build()
            for gg in g.grids:
                gg.duration_cv = cv
            b = splitk_sum.bindings(n=256)
            plan = compile_graph(g, "gfx942", b, {}, PassOptions(force_mode=mode))
            r = simulate(plan, seed=1)
            res[mode] = r.makespan_us
        best = min(res, key=res.get)
        print(f"{cv:5.2f} {res['static']:9.1f} {res['dynamic']:9.1f} {res['hybrid']:9.1f}   {best}")


def study_routing_skew() -> None:
    print("\nstudy 7: MoE routing imbalance vs schedule (B=32 tokens, 8 experts, gfx942 costs)")
    print(f"{'skew':>5} {'max/mean':>9} {'static':>9} {'dynamic':>9} {'hybrid':>9}   best")
    for skew in (0.0, 0.5, 1.0, 2.0, 4.0):
        rng = random.Random(3)
        # replace the LCG routing with a skewed distribution over experts
        weights = [pow(2.0, -skew * e) for e in range(moe_layer.NE)]
        B = 32
        topk = []
        for _ in range(B):
            picks: list[int] = []
            while len(picks) < moe_layer.K:
                e = rng.choices(range(moe_layer.NE), weights=weights)[0]
                if e not in picks:
                    picks.append(e)
            topk.append(sorted(picks))
        counts = [0] * moe_layer.NE
        for row in topk:
            for e in row:
                counts[e] += 1
        tiles = [-(-c // moe_layer.TPT) for c in counts]
        tok_indptr, indptr = [0], [0]
        for e in range(moe_layer.NE):
            tok_indptr.append(tok_indptr[-1] + counts[e]); indptr.append(indptr[-1] + tiles[e])
        fill = [0] * moe_layer.NE
        tok_slot = []
        for row in topk:
            s = []
            for e in row:
                s.append(tok_indptr[e] + fill[e]); fill[e] += 1
            tok_slot.append(s)
        runtime = {"topk": topk, "expert_counts": counts, "expert_tiles": tiles, "tok_indptr": tok_indptr,
                   "indptr": indptr, "tok_slot": tok_slot, "tile_expert": [[e] for e in range(moe_layer.NE) for _ in range(tiles[e])]}
        b = {"B": B, "seed": 0, "n_gg": indptr[-1]}
        res = {}
        for mode in ("static", "dynamic", "hybrid"):
            plan = compile_graph(moe_layer.build(), "gfx942", b, runtime, PassOptions(force_mode=mode))
            res[mode] = simulate(plan, seed=1).makespan_us
        best = min(res, key=res.get)
        print(f"{skew:5.1f} {max(counts) / (sum(counts) / len(counts)):9.2f} {res['static']:9.1f} {res['dynamic']:9.1f} {res['hybrid']:9.1f}   {best}")


if __name__ == "__main__":
    study_variance()
    study_routing_skew()
