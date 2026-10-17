"""Extract the key information of one GPU experiment into key.json and key.md.

The extractor reads committed text logs. It keeps only the lines that a pattern in
PATTERNS matches. Each kept line records its source file, its line number and the
SHA256 of the source file. key.json and key.md contain the same numbers. key.json is
for machine checks. key.md is for people.

Usage:
  python tools/extract_key.py --registry examples/gpu-live/results/registry.json --write
  python tools/extract_key.py --registry examples/gpu-live/results/registry.json --check
  python tools/extract_key.py --registry ... --only 2026-10-15/b4 --write

The output is deterministic. CI runs --check and fails when a committed key file
differs from a fresh extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 300

# Pattern registry. The first matching kind wins, in this order.
# Add a pattern here when a new runner prints a new key line.
PATTERNS = [
    # Compiler resource remarks (-Rpass-analysis=kernel-resource-usage).
    ("resource", re.compile(r"remark: +(Function Name|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|SGPRs Spill"
                            r"|VGPRs Spill|LDS Size \[bytes/block\]|Occupancy \[waves/SIMD\]): (\S+)")),
    # Build summary lines of the session runner (resources per arm).
    ("build", re.compile(r"\bBUILD arm=\S+ .*resources=")),
    # Session runner job results: "[t] ok   name rc=0 12s" and "[t] fail name rc=1 16s".
    ("exit", re.compile(r"^\[[^\]]+\]\s+(ok|fail|timeout|skip)\s+(\S+)\s+rc=(-?\d+)")),
    # Errors.
    ("error", re.compile(r"\berror:|fatal error|^Traceback|\b[A-Z]\w*Error\b|Failing command"
                         r"|Connection refused|failed to execute|Segmentation fault")),
    # Correctness gates and completion seals.
    ("gate", re.compile(r"\bBGATE\b|\bSEALED\b|\bB4-RESULT\b|\bGATE\b|\b(PASS|FAIL)\b|\b\d+ (passed|failed)\b")),
    # Timing summary lines.
    # ABREPORT and ABRUN come from tools/ab_same_box.py; ABRUN rc!=0 also counts as a failed run.
    ("timing", re.compile(r"\bBREPORT\b|\bABREPORT\b|\bABRUN\b|\bTRACE COMPLETE\b|\bmedian\b")),
    # Experiment identity.
    ("meta", re.compile(r"Marketing Name:\s+(AMD|NVIDIA)|\bgfx9\d\d\b|\bMI300X\b|\b[0-9a-f]{40}\b")),
]

BREPORT = re.compile(r"BREPORT b=(\d+) vs vLLM (\S+): throughput (\w+) median ([+-]?[\d.]+)% spread ([\d.]+)%"
                     r".*?per-request latency (\w+) median ([+-]?[\d.]+)% spread ([\d.]+)%")
ABREPORT = re.compile(r"ABREPORT label=(\S+) B vs A: gain median ([+-]?[\d.]+)% spread ([\d.]+)% "
                      r"control median ([+-]?[\d.]+)% spread ([\d.]+)% rounds=(\d+) verdict=(\S+)")
ABRUN = re.compile(r"ABRUN round=(\d+) arm=(\w) rc=(-?\d+)")
TRACE = re.compile(r"TRACE COMPLETE round=(\d+) b=(\d+) tasks=(\d+) span_us=([\d.]+)")
META = {
    "commit": re.compile(r"(?:[Cc]ommit|revision)[^0-9a-f\n]{0,20}([0-9a-f]{40})\b"),
    "machine": re.compile(r"Marketing Name:\s+((?:AMD|NVIDIA) .+?)\s*$|\b(MI300X(?: VF)?)\b"),
    "arch": re.compile(r"\b(gfx9\d\d)\b"),
}
RESOURCE_KEYS = {"VGPRs": "vgpr", "AGPRs": "agpr", "ScratchSize [bytes/lane]": "scratch_bytes_per_lane",
                 "SGPRs Spill": "sgpr_spill", "VGPRs Spill": "vgpr_spill", "LDS Size [bytes/block]": "lds_bytes",
                 "Occupancy [waves/SIMD]": "occupancy"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(text: str):
    for kind, pattern in PATTERNS:
        match = pattern.search(text)
        if match:
            return kind, match
    return None, None


def gate_verdict(text: str) -> str:
    if re.search(r"\bFAIL\b", text):
        return "FAIL"
    if re.search(r"complete=False|\bINCOMPLETE\b|NOT[- ]COMPLETE", text):
        return "INCOMPLETE"
    failed = re.search(r"\b(\d+) failed\b", text)
    if failed and int(failed.group(1)) > 0:
        return "FAIL"
    if re.search(r"\bPASS\b|complete=True|\bCOMPLETE\b|\b\d+ passed\b", text):
        return "PASS"
    return "UNKNOWN"


def spread(values):
    return dict(n=len(values), median=statistics.median(values), minimum=min(values),
                maximum=max(values), spread=max(values) - min(values))


def extract(directory: Path, entry: dict) -> dict:
    """Return the key.json value for one registry entry."""
    names = []
    for glob in entry["inputs"]:
        if any(c in glob for c in "*?["):
            found = sorted(p for p in directory.glob(glob) if p.is_file())
        else:
            found = [directory / glob] if (directory / glob).is_file() else []
        if not found:
            raise FileNotFoundError(f"{directory}: input {glob!r} matches no file")
        names += [p for p in found if p not in names]
    sources, lines = [], []
    builds, runs, gates, errors, comparisons, traces, resources = [], [], [], [], [], [], []
    meta = {}
    for path in names:
        rel = Path(os.path.relpath(path, directory)).as_posix()
        digest = sha256(path)
        sources.append(dict(file=rel, sha256=digest))
        function = None
        for number, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
            text = raw.rstrip()
            kind, match = classify(text)
            if kind is None:
                continue
            src = f"{rel}:{number}"
            lines.append(dict(file=rel, line=number, sha256=digest, kind=kind, text=text[:MAX_TEXT]))
            if kind == "resource":
                key, value = match.groups()
                if key == "Function Name":
                    function = dict(file=rel, function=value, src=src)
                    resources.append(function)
                elif function is not None:
                    function[RESOURCE_KEYS[key]] = int(value) if value.isdigit() else value
            elif kind == "build":
                builds.append(dict(text=text.strip()[:MAX_TEXT], src=src))
            elif kind == "exit":
                status, job, rc = match.groups()
                runs.append(dict(job=job, status=status, rc=int(rc), src=src))
            elif kind == "error":
                errors.append(dict(text=text.strip()[:MAX_TEXT], src=src))
            elif kind == "gate":
                gates.append(dict(text=text.strip()[:MAX_TEXT], verdict=gate_verdict(text), src=src))
            elif kind == "timing":
                b = BREPORT.search(text)
                t = TRACE.search(text)
                ab = ABREPORT.search(text)
                abr = ABRUN.search(text)
                if ab:
                    label, gmed, gspr, cmed, cspr, rounds, outcome = ab.groups()
                    comparisons.append(dict(batch=label, baseline="A (same machine, interleaved)",
                                            metric=f"paired gain, {rounds} rounds, control median {cmed}% "
                                                   f"spread {cspr}%", outcome=outcome,
                                            median_pct=float(gmed), spread_pct=float(gspr), src=src))
                elif abr:
                    rnd, arm, rc = abr.groups()
                    runs.append(dict(job=f"ab r{rnd} {arm}", status="ok" if rc == "0" else "fail", rc=int(rc),
                                     src=src))
                elif b:
                    batch, base, tout, tmed, tspr, lout, lmed, lspr = b.groups()
                    comparisons += [
                        dict(batch=int(batch), baseline=f"vLLM {base}", metric="throughput gain", outcome=tout,
                             median_pct=float(tmed), spread_pct=float(tspr), src=src),
                        dict(batch=int(batch), baseline=f"vLLM {base}", metric="per-request latency gain",
                             outcome=lout, median_pct=float(lmed), spread_pct=float(lspr), src=src)]
                elif t:
                    rnd, batch, tasks, span = t.groups()
                    traces.append(dict(round=int(rnd), batch=int(batch), tasks=int(tasks), span_us=float(span),
                                       src=src))
            if kind != "resource":
                for name, pattern in META.items():
                    found = pattern.search(text)
                    if found and name not in meta:
                        meta[name] = dict(value=next(g for g in found.groups() if g), src=src)
    exits = []
    for glob in entry.get("exits", []):
        for path in sorted(directory.glob(glob)):
            rel = path.relative_to(directory).as_posix()
            value = path.read_text().strip()
            exits.append(dict(file=rel, rc=int(value), sha256=sha256(path)))
    trace_stats = []
    for batch in sorted({t["batch"] for t in traces}):
        rows = [t for t in traces if t["batch"] == batch]
        trace_stats.append(dict(batch=batch, tasks=sorted({r["tasks"] for r in rows}),
                                span_us=spread([r["span_us"] for r in rows])))
    if any(g["verdict"] == "FAIL" for g in gates):
        verdict, reason = "FAIL", "a correctness gate line reports FAIL"
    elif any(g["verdict"] == "INCOMPLETE" for g in gates):
        verdict, reason = "INCOMPLETE", "a gate or seal line reports INCOMPLETE"
    elif any(r["rc"] != 0 for r in runs) or any(e["rc"] != 0 for e in exits):
        verdict, reason = "INCOMPLETE", "a job or exit receipt is nonzero"
    elif errors:
        verdict, reason = "INCOMPLETE", "an error line is present"
    else:
        verdict, reason = "COMPLETE", "all jobs exit 0, no failed gate, no error line"
    for key in ("commit", "machine", "arch"):
        if key in entry.get("meta", {}):
            given = entry["meta"][key]
            meta[key] = dict(value=given["value"], src=f"registry.json (from {given['source']})")
    return dict(schema=1, experiment=entry["dir"], title=entry["title"], config=entry.get("config", {}),
                commit=meta.get("commit"), machine=meta.get("machine"), arch=meta.get("arch"),
                verdict=dict(value=verdict, rule=reason), build=builds, comparisons=comparisons, traces=trace_stats,
                gates=gates, runs=runs, exits=exits, resources=resources,
                errors=dict(count=len(errors), lines=errors), sources=sources, lines=lines)


def cell(value):
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return out


def render(key: dict) -> str:
    def m(name):
        item = key.get(name)
        return (item["value"], item["src"]) if item else ("not found", "")
    md = [f"# Key results: {key['title']}", "",
          "Generated by `tools/extract_key.py`. Do not edit by hand. "
          "The numbers are the same as in `key.json`.", "", "## Conclusion", ""]
    rows = [("Verdict", key["verdict"]["value"], key["verdict"]["rule"]),
            ("Commit", *m("commit")), ("Machine", *m("machine")), ("Architecture", *m("arch"))]
    rows += [(f"Config: {k}", v, "registry.json") for k, v in sorted(key["config"].items())]
    if key["build"]:
        rows += [("Build summary lines", len(key["build"]), "see Extracted log lines")]
    rows += [("Error lines", key["errors"]["count"], "")]
    md += table(("Item", "Value", "Source"), rows)
    if key["comparisons"]:
        md += ["", "## Comparisons (median and spread of paired rounds)", ""]
        md += table(("B", "Baseline", "Metric", "Outcome", "Median %", "Spread pp", "Source"),
                    [(c["batch"], c["baseline"], c["metric"], c["outcome"], c["median_pct"], c["spread_pct"],
                      c["src"]) for c in key["comparisons"]])
    if key["traces"]:
        md += ["", "## Trace spans", ""]
        md += table(("B", "Rounds", "Tasks", "Median us", "Min us", "Max us", "Spread us"),
                    [(t["batch"], t["span_us"]["n"], ",".join(map(str, t["tasks"])), t["span_us"]["median"],
                      t["span_us"]["minimum"], t["span_us"]["maximum"], round(t["span_us"]["spread"], 6))
                     for t in key["traces"]])
    if key["gates"]:
        md += ["", "## Correctness gates and seals", ""]
        md += table(("Verdict", "Line", "Source"), [(g["verdict"], g["text"], g["src"]) for g in key["gates"]])
    if key["runs"] or key["exits"]:
        md += ["", "## Jobs and exit codes", ""]
        md += table(("Job", "Status", "rc", "Source"),
                    [(r["job"], r["status"], r["rc"], r["src"]) for r in key["runs"]] +
                    [(e["file"], "exit receipt", e["rc"], e["file"]) for e in key["exits"]])
    if key["resources"]:
        md += ["", "## Compiler resources", ""]
        cols = ("vgpr", "agpr", "scratch_bytes_per_lane", "sgpr_spill", "vgpr_spill", "lds_bytes", "occupancy")
        md += table(("Function", "VGPR", "AGPR", "Scratch B/lane", "SGPR spill", "VGPR spill", "LDS B",
                     "Waves/SIMD", "Source"),
                    [(r["function"], *(r.get(c) for c in cols), r["src"]) for r in key["resources"]])
    md += ["", "## Extracted log lines", "",
           "Each block lists the kept lines of one source file. The format is `line kind | text`.", ""]
    for source in key["sources"]:
        kept = [l for l in key["lines"] if l["file"] == source["file"]]
        md += [f"### `{source['file']}`", "", f"SHA256 `{source['sha256']}`, {len(kept)} kept lines.", ""]
        if kept:
            md += ["```text"] + [f"{l['line']:>5} {l['kind']:<8}| {l['text']}" for l in kept] + ["```", ""]
    return "\n".join(md).rstrip() + "\n"


def outputs(registry_path: Path, entry: dict):
    directory = (registry_path.parent / entry["dir"]).resolve()
    key = extract(directory, entry)
    return directory, json.dumps(key, indent=1, sort_keys=True) + "\n", render(key)


def shown(path: Path) -> str:
    return os.path.relpath(path, ROOT) if path.is_relative_to(ROOT) else str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=ROOT / "examples/gpu-live/results/registry.json")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write key.json and key.md")
    mode.add_argument("--check", action="store_true", help="fail when a committed key file differs")
    ap.add_argument("--only", action="append", help="registry dir to process (repeatable)")
    a = ap.parse_args(argv)
    entries = json.loads(a.registry.read_text())["key_extraction"]
    bad = 0
    for entry in entries:
        if a.only and entry["dir"] not in a.only:
            continue
        directory, key_json, key_md = outputs(a.registry, entry)
        for name, text in (("key.json", key_json), ("key.md", key_md)):
            path = directory / name
            if a.write:
                path.write_text(text)
                print(f"wrote {shown(path)}")
            elif not path.exists() or path.read_text() != text:
                print(f"KEY MISMATCH {shown(path)}: rerun tools/extract_key.py --write", file=sys.stderr)
                bad += 1
            else:
                print(f"KEY OK {shown(path)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
