"""Rebuild the batch-8 arm and re-run its one gate, after the host stopped refusing an identical greedy pair.

The 2026-10-14 combined run passed the B = 2 and B = 4 gates and failed B = 8 with
`error: requests 0 and 4 have the same golden tokens: one prompt twice` - the host's own duplicate check, which the
docs/BATCH-2026-10-14.md 4a amendment had not reached yet. The host now prints `BATCHCOLLISION requests 0 and 4` and
continues; the x8 arm has to be rebuilt because that string lives in the host binary, and its gate re-run.

Only three jobs do work here: `build_x8` (the arm binary), `prep_goldens` (the reference set already exists in the data
directory; the job re-checks it and writes goldens.json) and `bgate_x8`. Their dependencies are pulled in so the runner
accepts every read.

Run: examples/dsv2lite/reproduce.sh stage <ip> && ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=2700 \
     tools/session.sh start <ip> examples/sessions/x8_gate.py
"""
import importlib.util
from pathlib import Path

MAX_SESSION_SECONDS = 2700


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_batch = _load("ds_batch")
_by_name = {j["name"]: dict(j) for j in _batch.JOBS}

# resolve the closure: a read is satisfied by a job of that name, or by whichever job writes that path
_writer = {w.lstrip("?"): n for n, j in _by_name.items() for w in j.get("writes", [])}
_want = ["build_x8", "prep_goldens", "bgate_x8"]
_order = list(_want)
_selected = []
for n in _order:
    if n in _selected:
        continue
    _selected.append(n)
    for r in _by_name[n].get("reads", []):
        r = r.lstrip("?")
        dep = r if r in _by_name else _writer.get(r)
        if dep and dep not in _order:
            _order.append(dep)

JOBS = [_by_name[n] for n in _selected]
STUDIES = {"DeepSeek batch": ["bgate_x8"]}
