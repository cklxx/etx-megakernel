"""Truthful scoped filesystem snapshots for copied trees without Git metadata."""
import hashlib
from pathlib import Path

from etx.api.artifacts import payload_path
from etx.api.types import Configuration


_IGNORED = {'__pycache__', '.pytest_cache', '.git', '.venv'}


def source_snapshot(root, paths):
    root = Path(root).resolve()
    if isinstance(paths, str) or not paths:
        raise ValueError('source snapshot requires explicit nonempty paths')
    scope = sorted(set(payload_path(path) for path in paths))
    hashes = {}
    for name in scope:
        path = root / name
        if not path.exists():
            raise ValueError(f'missing snapshot source: {name}')
        candidates = [path, *sorted(path.rglob('*'))] if path.is_dir() else [path]
        for candidate in candidates:
            relative = candidate.relative_to(root)
            if any(part in _IGNORED for part in relative.parts) or candidate.suffix == '.pyc':
                continue
            if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents if root in parent.parents):
                raise ValueError(f'snapshot sources must not traverse symlinks: {relative}')
            if candidate.is_file():
                with candidate.open('rb') as source:
                    hashes[relative.as_posix()] = hashlib.file_digest(source, 'sha256').hexdigest()
    record = dict(schema_version=1, kind='scoped-filesystem-snapshot', scope=scope,
                  commit=None, dirty=None, source_sha256=dict(sorted(hashes.items())))
    identity = hashlib.sha256(Configuration(record)._json.encode()).hexdigest()
    return Configuration(dict(identity=identity, **record))


def verify_source_snapshot(root, record):
    record = Configuration(record)
    if record != source_snapshot(root, record['scope']):
        raise ValueError('source snapshot identity/content mismatch')
    return record
