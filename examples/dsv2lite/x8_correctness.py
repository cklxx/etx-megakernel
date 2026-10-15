"""Six predeclared correctness diagnostics; timings have no performance verdict."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
from etx.session.batch import etx_result
from examples.dsv2lite.arms import ARMS, build_env
from examples.dsv2lite.batch_goldens import sha, binding
from examples.dsv2lite.inplace_session import Session, write_json
from examples.dsv2lite.reference_prepare import verify

KEPT = "alld4r0l1kvsmw"
CASES = (("x1km_r5", "x1km", 5), ("x1km_r7", "x1km", 7),
         ("kept_r5", KEPT, 5), ("kept_r7", KEPT, 7),
         ("x8_repeat0", "x8", None), ("x8_repeat1", "x8", None))


def diagnostic(log: str, b: int, rc: int) -> dict:
    rows = re.findall(r"^LAYERGATE r=(\d+) layer=(\d+) rel=(\S+) cosine=(\S+) pass=([01])$", log, re.M)
    keys = [(int(r), int(l)) for r, l, *_ in rows]
    expected = [(r, l) for r in range(b) for l in range(27)]
    if sorted(keys) != expected or any(not math.isfinite(float(v)) for row in rows for v in row[2:4]):
        return dict(status="INCOMPLETE", reason="missing, duplicate or nonfinite layer metrics")
    failed = [(int(r), int(l)) for r, l, rel, cos, flag in rows
              if flag != "1" or float(rel) > .02 or float(cos) < .999]
    # Parse the unchanged strict hidden AND token gate. A failed gate never produces a timing result here.
    try:
        etx_result(log, b)
    except (ValueError, KeyError) as exc:
        heads = re.findall(r"^BATCH requests=(\d+) tokens=32 step_median_ms=\S+$", log, re.M)
        reqs = re.findall(r"^BATCHREQ r=(\d+) tokens=(\d+)/32 layers=(\d+) warmup=(\d+)/32 ", log, re.M)
        complete = heads == [str(b)] and sorted(int(q[0]) for q in reqs) == list(range(b))
        status = "FAIL" if complete and rc in (0, 3) and (failed or any(q[1] != "32" or q[3] != "32" for q in reqs)) else "INCOMPLETE"
        return dict(status=status, reason=str(exc), failed_boundaries=failed)
    return dict(status="PASS" if rc == 0 and not failed else "INCOMPLETE", failed_boundaries=failed)


def classify(results: dict) -> str:
    if any(v["status"] == "INCOMPLETE" for v in results.values()) or set(results) != {c[0] for c in CASES}:
        return "UNRESOLVED"
    if any(results[n]["status"] == "FAIL" for n in ("x1km_r5", "x1km_r7")):
        return "SINGLE_REQUEST_FAILURE_CONCURRENCY_NOT_NECESSARY"
    repeats = [results[f"x8_repeat{i}"]["status"] for i in range(2)]
    if repeats == ["FAIL", "FAIL"]:
        return "MULTI_REQUEST_PATH_DEPENDENT_NO_RACE_PROOF"
    if repeats == ["PASS", "PASS"]:
        return "NOT_REPRODUCED_ON_CERTIFIED_INPUTS"
    return "UNRESOLVED"


class CorrectnessSession(Session):
    def diagnostics(self):
        out = self.logs / "x8_correctness.json"
        if out.exists():
            raise ValueError("fresh result directory required")
        report = dict(status="INCOMPLETE", classification="UNRESOLVED", results={}, performance_verdict=None,
                      original_payload_recovered=False)
        try:
            ready = json.loads((self.logs / "references_ready.json").read_text())
            if ready["status"] != "READY" or len(ready["requests"]) != 8:
                raise ValueError("eight producer-bound references required")
            self.data = self.logs / "references"
            for r in range(8):
                d = self.data if r == 0 else self.data / f"r{r}"
                verify(d, ready["requests"][r]["receipt"]["recipe"])
            identities = {}
            for arm in ("x1km", "x8", KEPT):
                paths = ("run", "plan.json", "megakernel_d0.hip", "resources.json")
                identities[arm] = dict(files={f: sha(self.out(arm) / f) for f in paths}, recipe=build_env(ARMS[arm]))
            report["builds"] = identities
            report["source"] = self.source_identity()
            report["references"] = ready
            for name, arm, r in CASES:
                ref = self.data / f"r{r}" if r is not None else self.data
                args = self.args(ref)
                if r is None:
                    args += [a for k in range(1, 8) for a in ("--req-dir", str(self.data / f"r{k}"))]
                error = None
                try:
                    text = self.run(name, [self.out(arm) / "run", *args], env=dict(ETX_DS_LAYER_METRICS="1"))
                    rc = 0
                except RuntimeError as exc:
                    error = str(exc)
                    text = (self.logs / f"{name}.raw.log").read_text()
                    match = re.search(r"failed rc=(\d+)", error)
                    rc = int(match[1]) if match else -1
                value = diagnostic(text, 8 if r is None else 1, rc)
                value.update(arm=arm, reference_request=r, exit_code=rc, error=error,
                             raw_sha256=sha(self.logs / f"{name}.raw.log"))
                report["results"][name] = value
                write_json(out, report)
                print(f"DIAGNOSTIC name={name} status={value['status']} rc={rc}", flush=True)
            report["classification"] = classify(report["results"])
            report["status"] = "COMPLETE" if report["classification"] != "UNRESOLVED" else "INCOMPLETE"
        except Exception as exc:
            report["error"] = str(exc)
            raise
        finally:
            write_json(out, report)


def finalize(s: Session):
    """The final lane writes a result even when a producer or build dependency failed."""
    result = dict(status="INCOMPLETE", classification="UNRESOLVED", performance_verdict=None)
    for name in ("references_ready", "x8_correctness"):
        p = s.logs / f"{name}.json"
        result[name] = json.loads(p.read_text()) if p.exists() else dict(status="MISSING")
    if result["x8_correctness"].get("status") == "COMPLETE":
        result.update(status="COMPLETE", classification=result["x8_correctness"]["classification"])
    write_json(s.logs / "x8_summary.json", result)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("operation", nargs="?", default="diagnostics", choices=("diagnostics", "finalize"))
    a = ap.parse_args(argv)
    if a.operation == "finalize":
        finalize(Session())
    else:
        CorrectnessSession().diagnostics()


if __name__ == "__main__":
    main()
