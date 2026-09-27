#!/usr/bin/env python3
"""One decoder layer of vLLM's decode as a kernel sequence, averaged over the layers of the traced decode steps.

  python3 examples/vllm_llm/layer_trace.py <rocprofv3 dir> --first <kernel name substring that opens a layer>

The kernel trace of a HIP graph has no gaps (each kernel's start is the previous one's end), so a kernel's duration is
everything the launch costs: dispatch, ramp, body, tail and the end-of-kernel release. Each layer is the run of kernels
from one occurrence of --first to the next; runs whose kernel-name sequence differs from the most common one (the first
and last layers, lm_head, sampling) are dropped. Prints, per position: mean / min duration, workgroups x threads, VGPR /
AGPR, LDS, name; then the layer period and the count of launches under 8 us.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from statistics import mean


def load(d: str):
    rows = []
    for f in glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True):
        rows += list(csv.DictReader(open(f)))
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--first", default="fused_add_rms_norm_moe_forward")
    ap.add_argument("--last-steps", type=int, default=20000, help="only the last N kernels (the long generate call)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = load(a.dir)[-a.last_steps:]
    starts = [i for i, r in enumerate(rows) if a.first in r["Kernel_Name"]]
    layers = [rows[s:e] for s, e in zip(starts, starts[1:])]
    sig = Counter(tuple(r["Kernel_Name"] for r in L) for L in layers).most_common(1)[0][0]
    keep = [L for L in layers if tuple(r["Kernel_Name"] for r in L) == sig]
    out = []
    for k, name in enumerate(sig):
        ds = [(int(L[k]["End_Timestamp"]) - int(L[k]["Start_Timestamp"])) / 1e3 for L in keep]
        r = keep[0][k]
        wg = int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"])
        grid = int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"])
        out.append({"pos": k, "mean_us": mean(ds), "min_us": min(ds), "wgs": grid // wg, "threads": wg,
                    "vgpr": int(r["VGPR_Count"]), "agpr": int(r["Accum_VGPR_Count"]), "lds": int(r["LDS_Block_Size"]), "name": name})
    # a layer's kernels run back to back, so the period is the sum of its durations (the span of a run can include the
    # host's idle time between two generate calls)
    period = [sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in L) / 1e3 for L in keep]
    print(f"{len(keep)} of {len(layers)} layers share the sequence of {len(sig)} kernels; period mean {mean(period):.1f} us")
    for o in out:
        print(f"{o['pos']:3d} {o['mean_us']:6.2f} {o['min_us']:6.2f} us  {o['wgs']:4d}x{o['threads']:<4d} v{o['vgpr']:3d}/a{o['agpr']:<3d} "
              f"lds {o['lds']:6d}  {o['name'][:80]}")
    small = [o for o in out if o["mean_us"] < 8]
    print(f"small (< 8 us): {len(small)} launches, {sum(o['mean_us'] for o in small):.1f} us per layer")
    if a.json:
        json.dump({"period_us": mean(period), "layers": len(keep), "kernels": out}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
