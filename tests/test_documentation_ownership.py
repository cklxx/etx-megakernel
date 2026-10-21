"""Canonical documentation ownership must not rewrite sealed evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'docs/ownership-moves.json').read_text())
MOVES = {row['old']: row['new'] for row in MANIFEST['rows']}


def test_move_destinations_and_anchors_exist():
    for row in MANIFEST['rows']:
        assert (ROOT / row['new']).exists(), row['new']
        for anchor in row['anchors']:
            assert (ROOT / anchor).exists(), anchor


def test_sealed_evidence_bytes_are_unchanged():
    for old, digest in MANIFEST['immutable'].items():
        path = ROOT / MOVES.get(old, old)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, old


def test_historical_aliases_resolve_to_recorded_targets():
    for old, target in MANIFEST['compatibility'].items():
        alias = ROOT / old
        assert alias.is_symlink(), old
        assert alias.resolve() == (ROOT / target).resolve(), old
        assert alias.exists(), old
