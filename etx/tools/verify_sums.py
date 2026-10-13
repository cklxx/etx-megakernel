"""Verify a fetched session directory against the SHA256SUMS the box wrote (review 07, R03 / R04).

  python -m etx.tools.verify_sums <dir> [--excluded GLOB ...]

The box's SHA256SUMS is the provenance record; it is never overwritten. Every listed file must be present with the
listed hash, unless it matches one of the --excluded globs (files a fetch leaves on the box by design, e.g. binaries).
Writes <dir>/SHA256SUMS.local (hashes of everything fetched) next to it. Exit 0 = every fetched file verified,
2 = a mismatch or an unexpected absence (the copy is not what the box wrote), 3 = no box-side SHA256SUMS.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(d: Path, excluded: list[str]) -> tuple[int, list[str]]:
    sums = d / "SHA256SUMS"
    if not sums.exists():
        return 3, [f"NO BOX-SIDE SHA256SUMS in {d}: provenance missing"]
    lines, bad, skipped, ok = [], 0, 0, 0
    for row in sums.read_text().splitlines():
        if not row.strip():
            continue
        digest, name = row.split(None, 1)
        name = name.lstrip("*").removeprefix("./")
        f = d / name
        if any(fnmatch.fnmatch(name, g) for g in excluded):
            skipped += 1
            continue
        if not f.exists():
            bad += 1
            lines.append(f"MISSING {name}")
        elif sha256(f) != digest:
            bad += 1
            lines.append(f"MISMATCH {name}")
        else:
            ok += 1
    local = sorted(p for p in d.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.local"))
    (d / "SHA256SUMS.local").write_text("".join(f"{sha256(p)}  ./{p.relative_to(d)}\n" for p in local))
    lines.append(f"{'VERIFIED' if not bad else 'FAILED'}: {ok} files match the box, {bad} bad, {skipped} not fetched "
                 f"by design ({', '.join(excluded) or 'none'})")
    return (2 if bad else 0), lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--excluded", nargs="*", default=[])
    a = ap.parse_args(argv)
    rc, lines = verify(a.dir, a.excluded)
    print("\n".join(lines), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
