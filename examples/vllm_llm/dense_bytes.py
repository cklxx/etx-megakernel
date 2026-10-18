"""Account the dense GEMV weight bytes of two vl_plan.txt files, replicated vs column-batched.

    python -m examples.vllm_llm.dense_bytes --replicated vl_plan_rep.txt --batched vl_plan_bat.txt

The dense roles qkv, o_proj, router and lm_head are the ones ETX_VL_BATCH=1 emits once for all requests; every other
role (attention, MoE, norms) stays replicated. A GEMV instance reads M x K bf16 weights once however many activation
columns (`nb`) it serves, so the weight bytes per instance are M * K * 2 and the batched plan reads them once instead
of once per request. The delta is the dense weight re-read the batched recipe removes; divide by the stream rate for
the bandwidth floor (the phase model in etx/sim uses a fixed 2.0 us per DSL GEMV launch, so the simulator does not see
this delta).

This is the E1 accounting check: it must show the dense bytes fall by the request count, not stay flat.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DENSE_ROLES = {"qkv", "o_proj", "router", "lm_head"}
BF16 = 2


def dense_bytes(path: Path) -> tuple[float, int, set[int]]:
    total = 0.0
    instances = 0
    nb_seen: set[int] = set()
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "total")):
            continue
        f = line.split()
        if len(f) < 17:
            raise ValueError(f"{path}: malformed instance row: {line!r}")
        role = f[3]
        nb = int(f[16]) if len(f) > 16 else 1
        nb_seen.add(nb)
        if role in DENSE_ROLES:
            # weights are read once per instance regardless of the column count nb
            total += int(f[9]) * int(f[10]) * BF16
            instances += 1
    return total, instances, nb_seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m examples.vllm_llm.dense_bytes", description=__doc__.split("\n\n")[0])
    ap.add_argument("--replicated", required=True, type=Path, help="vl_plan.txt of the replicated recipe")
    ap.add_argument("--batched", required=True, type=Path, help="vl_plan.txt of the batched recipe")
    ap.add_argument("--rate", type=float, default=3.56, help="stream rate in TB/s for the floor (hw_bench lf)")
    a = ap.parse_args(argv)
    rep, rep_n, rep_nb = dense_bytes(a.replicated)
    bat, bat_n, bat_nb = dense_bytes(a.batched)
    delta = rep - bat
    print(f"replicated {a.replicated}: dense weights {rep / 1e9:.4f} GB across {rep_n} instances")
    print(f"batched    {a.batched}: dense weights {bat / 1e9:.4f} GB across {bat_n} instances (nb {sorted(bat_nb)})")
    print(f"delta {delta / 1e9:.4f} GB = {delta / (a.rate * 1e12) * 1e3:.3f} ms at {a.rate} TB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
