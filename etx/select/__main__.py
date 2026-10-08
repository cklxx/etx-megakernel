"""python -m etx.select [--exports | --manifest] <config.json> [--machine gfx942]

--exports   the imported exports of the plan, comma-separated (the import list build.sh derives today)
--manifest  Plan.manifest() as JSON
default     one line per phase: key, export, kind, reason
"""
import argparse
import json

from . import plan

ap = argparse.ArgumentParser(prog="python -m etx.select")
ap.add_argument("config")
ap.add_argument("--machine", default="gfx942")
g = ap.add_mutually_exclusive_group()
g.add_argument("--exports", action="store_true")
g.add_argument("--manifest", action="store_true")
a = ap.parse_args()
p = plan(a.config, a.machine)
if a.exports:
    print(",".join(p.exports("imported")))
elif a.manifest:
    print(json.dumps(p.manifest(), indent=1))
else:
    print(f"budget {p.budget}  cost {p.cost:.2f} us  resources {p.resources()}")
    for k, v in p.phases.items():
        print(f"  {k:<10} {v.export:<20} {v.kind:<8} {p.reasons[k]}")
