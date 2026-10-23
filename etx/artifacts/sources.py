"""Relocatable, checksum-checked source payloads; never executable status."""
import hashlib
import json
from pathlib import Path

from etx.api.artifacts import SourceArtifact, payload_path
from etx.api.types import PlannedWorkload


def save_sources(artifact, directory):
    if not isinstance(artifact, SourceArtifact):
        raise TypeError('save_sources requires SourceArtifact')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, text in artifact.files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='') as output:
            output.write(text)
    # Manifest last: interrupted emission must not look like a complete artifact.
    with (directory / 'manifest.json').open('x') as output:
        json.dump(dict(identity=artifact.identity, **artifact.manifest()), output, sort_keys=True, indent=2)
        output.write('\n')
    return artifact


def load_sources(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / 'manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('artifact manifest must not be a symlink')
    manifest = json.loads(manifest_path.read_text())
    fields = {'schema_version', 'status', 'planned', 'rank_sources', 'rank_headers',
              'external_sources', 'payloads', 'identity'}
    if not isinstance(manifest, dict) or set(manifest) != fields or not isinstance(manifest['payloads'], dict):
        raise ValueError('invalid source artifact manifest')
    files = {}
    for name, expected in manifest['payloads'].items():
        payload_path(name)
        path = directory / name
        if any(part.is_symlink() for part in [path, *path.parents] if part != directory and directory in part.parents):
            raise ValueError('artifact payload must not traverse symlinks')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'artifact payload checksum mismatch: {name}')
        files[name] = data.decode('utf-8')
    artifact = SourceArtifact(PlannedWorkload.from_dict(manifest['planned']), files,
                              manifest['rank_sources'], manifest['rank_headers'],
                              manifest['external_sources'], manifest['status'], manifest['schema_version'])
    if artifact.identity != manifest['identity']:
        raise ValueError('source artifact identity mismatch')
    return artifact
