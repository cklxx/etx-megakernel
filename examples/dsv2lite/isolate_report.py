"""Report of an ETX_DSV2_ISOLATE run (docs/MODEL-dsv2.md section 4): per isolated phase, over its repetitions (median),
the span (first task ready -> last done), mean task body, ramp (median ready - first ready), tail, and the achieved
bandwidth of the phase's weight bytes over the span.

  python -m examples.dsv2lite.isolate_report PLAN TRACE
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict

from etx.tools import timeline as tl
from examples.dsv2lite.bytes import per_layer


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    tr = tl.build_trace(tl.load_plan(a[0]), tl.read_raw(a[1]))
    inst = defaultdict(list)
    for t in tr.tasks:
        m = re.match(r"iso(\d+)x(\d+)_r(\d)_([a-z_]+?)_(-?\d+)$", t.name)
        if m:
            inst[(int(m.group(1)), int(m.group(3)), m.group(4), int(m.group(5)), int(m.group(2)))].append(t)
    if not inst:
        raise SystemExit("no iso* tasks in the trace (build with ETX_DSV2_ISOLATE)")
    rows = defaultdict(lambda: defaultdict(list))
    b = per_layer(1024)
    for (blk, r, ph, L, k), ts in inst.items():
        ready = sorted(t.ready for t in ts)
        first, med, last = ready[0], ready[len(ready) // 2], max(t.done for t in ts)
        body = sum(t.done - t.ready for t in ts) / len(ts)
        span = last - first
        key = (blk, r, ph, L)
        rows[key]["span"].append(span); rows[key]["body"].append(body); rows[key]["ramp"].append(med - first)
        rows[key]["tail"].append(last - (med + body)); rows[key]["tasks"].append(len(ts))
        rows[key]["tbs"].append(b.get(ph, 0.0) / span / 1e6 if span > 0 else 0.0)
    print(f"{'block':>5} {'request':>7} {'phase':10} {'reps':>4} {'tasks':>5} {'span':>7} {'body':>7} {'ramp':>6} {'tail':>6} {'TB/s':>6}   (us, medians over repetitions)")
    for key in sorted(rows):
        v = rows[key]
        med = {k: statistics.median(x) for k, x in v.items()}
        print(f"{key[0]:5d} {key[1]:7d} {key[2] + '_' + str(key[3]):10} {len(v['span']):4d} {int(med['tasks']):5d} {med['span']:7.2f} {med['body']:7.2f} "
              f"{med['ramp']:6.2f} {med['tail']:6.2f} {med['tbs']:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
