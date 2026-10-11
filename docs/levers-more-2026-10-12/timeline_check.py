"""Inflate the conditional simulated schedule and run the timeline CLI on it."""
import gzip
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
with tempfile.TemporaryDirectory(prefix="levers-more-") as tmp:
    tmp = Path(tmp)
    for source, dest in (("combined-plan.json.gz", "plan.json"), ("combined.bin.gz", "trace.bin")):
        (tmp / dest).write_bytes(gzip.decompress((OUT / source).read_bytes()))
    for label, extra in (("timeline-full", ["--json", str(OUT / "timeline-full.json")]),
                         ("timeline-layer24", ["--grids", "_24$", "--html", str(OUT / "timeline-layer24.html")])):
        command = [sys.executable, "-m", "etx.tools.timeline", str(tmp / "plan.json"), str(tmp / "trace.bin"),
                   "--bytes", str(ROOT / "docs/xcd-timeline-2026-10-11/bytes.json"), *extra]
        print("COMMAND", " ".join(command), flush=True)
        with (OUT / f"{label}.txt").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("SIMULATED timeline CLI: full-step JSON and layer-24 HTML completed", flush=True)
