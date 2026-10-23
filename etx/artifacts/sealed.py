"""Content provenance for legacy absolute-path sealed source inventories."""
import hashlib
import json
from pathlib import Path
import subprocess


def source_provenance(sources, revision=None, *, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    origin = 'declared' if revision else 'unknown'
    if not revision and (root / '.git').exists():
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        origin = 'git-head'
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError('source revision must be a nonempty string or null')
    content = json.dumps(sources, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return dict(schema_version=1, kind='sealed-source-inventory', commit=revision,
                revision_origin=origin, dirty=None,
                inventory_sha256=hashlib.sha256(content.encode()).hexdigest())
