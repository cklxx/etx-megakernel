"""The session record (docs/core/SESSION-ARCH-2026-10-13.md, component B).

Invariant: a result is a SEALED record whose every byte was written by this session's processes after they were
reaped. The runner creates the record directory itself (never an existing one), owns every file in it, and seals it
last, after every job is terminal and swept (etx/session/proc.py): MANIFEST lists the sha256 of every file, SEAL names
the session, its source, its job table and the MANIFEST hash. A copy is verified against its own SEAL; anything the
SEAL does not cover, or an unsealed record, is not a result.

  python -m etx.session.record verify <copy>      0 sealed, complete, every byte matches; 3 unsealed or a job not ok;
                                                  2 a byte, a file or the MANIFEST differs
  python -m etx.session.record local-only <dir>   the globs a fetch leaves on the box (one per line)
"""
# Driver laptops run this file with the system python3 (3.9 on macOS); keep annotations lazy.
from __future__ import annotations
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time

LOCAL_ONLY = ("b_*/run", "b_*/fleet_src/*", "*.o")     # binaries and patched source copies stay on the box
OWN = ("MANIFEST", "SEAL")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def create(root: Path) -> Path:
    """A new record directory under root, named by the session id (UTC time + nonce). Never an existing one."""
    root.mkdir(parents=True, exist_ok=True)
    while True:
        path = root / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{secrets.token_hex(3)}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def files(d: Path) -> list[str]:
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if not p.is_dir() and str(p.relative_to(d)) not in OWN)


def seal(d: Path, *, jobs: dict, plan: str, local_only=LOCAL_ONLY, failure: str | None = None) -> dict:
    """Write MANIFEST and SEAL. The caller guarantees that nothing writes into d any more."""
    manifest = "".join(f"{sha256(d / name)}  {name}\n" for name in files(d))
    (d / "MANIFEST").write_text(manifest)
    identity = Path("SOURCE_IDENTITY")
    s = dict(session=d.name, sealed=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), jobs=jobs,
             complete=failure is None and bool(jobs) and all(v == "ok" for v in jobs.values()), plan_sha256=plan,
             source=identity.read_text() if identity.is_file() else None, local_only=list(local_only),
             manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
    if failure is not None:
        s['failure'] = failure
    (d / "SEAL").write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
    return s


def verify(d: Path) -> tuple[int, list[str]]:
    try:
        s = json.loads((d / "SEAL").read_text())
        manifest = (d / "MANIFEST").read_text()
    except (OSError, ValueError):
        return 3, [f"UNSEALED {d}: no SEAL / MANIFEST -- not a result (the runner did not finish)"]
    if hashlib.sha256(manifest.encode()).hexdigest() != s["manifest_sha256"] or d.name != s["session"]:
        return 2, [f"FAILED: MANIFEST or session name of {d} is not what SEAL {s['session']} sealed"]
    listed = dict(reversed(line.split("  ", 1)) for line in manifest.splitlines())
    lines, bad, left = [], 0, 0
    for name, digest in listed.items():
        if not (d / name).exists():
            if any(fnmatch.fnmatch(name, g) for g in s["local_only"]):
                left += 1
            else:
                bad += 1
                lines.append(f"MISSING {name}")
        elif sha256(d / name) != digest:
            bad += 1
            lines.append(f"MISMATCH {name}")
    for name in sorted(set(files(d)) - set(listed)):
        bad += 1
        lines.append(f"NOT IN THE SEALED MANIFEST {name}")
    done = [k for k, v in s["jobs"].items() if v != "ok"]
    completion = (f"INCOMPLETE: runner failure: {s['failure']}" if s.get('failure') is not None else
                  f"INCOMPLETE: not ok: {done}" if done else "complete")
    lines.append(f"{'FAILED' if bad else 'VERIFIED'}: session {s['session']}, {len(listed) - left} files match, {bad} bad,"
                 f" {left} left on the box by design; " + completion)
    return (2 if bad else 3 if not s["complete"] else 0), lines


def verify_legacy(d: Path, archives=()) -> tuple[int, list[str]]:
    """A result kept before records were sealed: its SHA256SUMS (computed after the fetch) against the copy, files
    archived outside git looked up in `archives`. Never more than 3: without a SEAL it is not a sealed record."""
    sums = d / "SHA256SUMS"
    if not sums.exists():
        return 3, [f"UNSEALED {d}: no SEAL and no SHA256SUMS"]
    bad, archived = [], 0
    for row in sums.read_text().splitlines():
        digest, name = row.split(None, 1)
        name = name.lstrip("*").removeprefix("./")
        path = next((p for p in (d / name, *(a / Path(name).name for a in archives)) if p.exists()), None)
        archived += path is None
        bad += [f"MISMATCH {name}"] if path is not None and sha256(path) != digest else []
    return (2 if bad else 3), bad + [f"LEGACY (no SEAL): SHA256SUMS {'FAILED' if bad else 'matches'}, "
                                     f"{archived} archived file(s) not present"]


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 2 or a[0] not in ("verify", "local-only"):
        raise SystemExit(__doc__)
    if a[0] == "local-only":
        print("\n".join(json.loads((Path(a[1]) / "SEAL").read_text())["local_only"]))
        return 0
    rc, lines = verify(Path(a[1]))
    print("\n".join(lines), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
