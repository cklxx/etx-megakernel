"""Canonical documentation ownership must not rewrite sealed evidence."""
import hashlib
import json
import re
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


def test_moved_documents_and_snapshots_keep_relative_links():
    paths = {ROOT / row['new'] for row in MANIFEST['rows']}
    paths.update(path for path in (ROOT / 'docs/archive/edc2cbe').glob('*.md')
                 if not path.is_symlink())
    link = re.compile(r'\]\(([^\s)]+)\)|(?:href|src)=["\x27]([^"\x27]+)')
    for path in paths:
        if not path.is_file() or path.suffix not in ('.md', '.html'):
            continue
        for match in link.finditer(path.read_text()):
            url = next(value for value in match.groups() if value)
            target = url.split('#')[0].split('?')[0]
            if not target or ':' in target or target.startswith('/'):
                continue
            assert (path.parent / target).exists(), (path, target)


def test_archived_snapshots_match_pinned_audit_inputs():
    inputs = json.loads((ROOT / 'docs/citation-audit-2026-10-11/inputs.sha256.json').read_text())
    for path in (ROOT / 'docs/archive/edc2cbe').glob('*.md'):
        if path.is_symlink():
            continue
        digests = {digest for name, digest in inputs.items() if Path(name).name == path.name}
        assert hashlib.sha256(path.read_bytes()).hexdigest() in digests, path
