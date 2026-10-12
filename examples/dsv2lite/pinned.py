"""The pinned inputs of the DeepSeek port: fleet's source at 3fb67a6 and the archived 2026-09-27 data sets.

Never fleet's working tree (it carries uncommitted experiments): the export is `git archive 3fb67a6`, verified by a
content hash. Both live under ~/code/etx-cache (the repo's existing cache, tools/session.sh:24), not /tmp.

  python -m examples.dsv2lite.pinned export     # create/verify the fleet export, print its path
  python -m examples.dsv2lite.pinned data       # fetch/verify the 09-27b/d data sets, print their path
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile

FLEET_COMMIT = "3fb67a6"
FLEET_SRC_SHA256 = "28f5cbb4f39524df318883522341c48ca4162c18b71a3a6223c52eabb40ce4b3"
CACHE = Path(os.environ.get("ETX_CACHE", Path.home() / "code" / "etx-cache"))
FLEET_REPO = Path(os.environ.get("ETX_FLEET_REPO", Path.home() / "code" / "fleet-mi300x"))
DATA_RELEASE = ("cklxx/etx-data", "data-2026-09-27")
# Tarball checksums from the release's own TARBALLS.sha256 (verified 2026-10-13).
DATA_SETS = ("results_2026-09-27b", "results_2026-09-27d")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def source_sha(src: Path) -> str:
    """Order-independent content hash of a source tree: relative path + file hash, sorted."""
    h = hashlib.sha256()
    for path in sorted(src.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(src).as_posix().encode() + b"\0" + bytes.fromhex(sha256(path)))
    return h.hexdigest()


def fleet_export(dest: Path | None = None) -> Path:
    """The pinned fleet tree (with src/), created from the fleet repository on first use, always hash-checked."""
    dest = Path(os.environ.get("ETX_DSV2_FLEET_EXPORT", dest or CACHE / f"fleet-{FLEET_COMMIT}"))
    if not (dest / "src").is_dir():
        if not (FLEET_REPO / ".git").exists():
            raise FileNotFoundError(f"no pinned export at {dest} and no fleet repository at {FLEET_REPO}")
        dest.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "-C", str(FLEET_REPO), "archive", FLEET_COMMIT],
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    if (dest / ".git").exists():
        raise RuntimeError(f"{dest} is a checkout, not an export")
    got = source_sha(dest / "src")
    if got != FLEET_SRC_SHA256:
        raise RuntimeError(f"{dest}/src is not fleet {FLEET_COMMIT}: {got}")
    return dest


def data_sets(dest: Path | None = None) -> Path:
    """The archived raw data (traces, plans, logs) every account in this directory is computed from."""
    dest = Path(os.environ.get("ETX_DSV2_DATA_SETS", dest or CACHE / DATA_RELEASE[1]))
    missing = [s for s in DATA_SETS if not (dest / s / "SHA256SUMS").is_file()]
    if missing:
        dest.mkdir(parents=True, exist_ok=True)
        repo, tag = DATA_RELEASE
        patterns = [arg for s in missing for arg in ("-p", f"{s}.tgz")] + ["-p", "TARBALLS.sha256"]
        subprocess.run(["gh", "release", "download", tag, "-R", repo, "--clobber", "-D", str(dest), *patterns], check=True)
        sums = dict(reversed(line.split()) for line in (dest / "TARBALLS.sha256").read_text().splitlines() if line.strip())
        for s in missing:
            tgz = dest / f"{s}.tgz"
            if sha256(tgz) != sums[tgz.name]:
                raise RuntimeError(f"{tgz} does not match the release's TARBALLS.sha256")
            with tarfile.open(tgz) as t:
                t.extractall(dest, filter="data")
    for s in DATA_SETS:
        for line in (dest / s / "SHA256SUMS").read_text().splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*").removeprefix("./")
            if sha256(dest / s / name) != digest:
                raise RuntimeError(f"{dest / s / name} does not match its SHA256SUMS")
    return dest


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "export"
    print(fleet_export() if what == "export" else data_sets())
