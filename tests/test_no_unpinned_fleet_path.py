"""No tracked code, test or tool file names the unpinned fleet checkout; builds use pinned.fleet_export()."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = "/workspace/" + "fleet-mi300x"   # split so this file does not match itself
CODE = (".py", ".sh", ".hip", ".h", ".cpp", ".c", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".json")


def test_no_tracked_code_file_names_the_unpinned_fleet_checkout():
    out = subprocess.run(["git", "-C", str(ROOT), "grep", "-l", "-F", BANNED, "--", "."],
                         capture_output=True, text=True)
    assert out.returncode in (0, 1), out.stderr
    hits = [f for f in out.stdout.split() if f.endswith(CODE) and not f.startswith("docs/history/")]
    assert hits == [], f"{BANNED} in {hits}: use examples.dsv2lite.pinned.fleet_export()"
