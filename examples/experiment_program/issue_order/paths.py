"""Resolve sealed compiler inputs without bypassing explicit staging prefixes."""
import hashlib
import json
from pathlib import Path
from examples.experiment_program.protocol import relocation_seal,resolve_path,require


def dependency_path(name):
    original=Path(name).expanduser().absolute()
    mapping=relocation_seal()  # shared validator rejects unknown/chained maps
    resolved=resolve_path(original)
    pinned=False
    for filename,digest in mapping.items():
        payload=Path(filename).read_bytes()
        require(hashlib.sha256(payload).hexdigest()==digest,'staging map changed during dependency resolution')
        prefixes=json.loads(payload)['paths']
        pinned=any(original.is_relative_to(Path(prefix)) for prefix in prefixes)
    # Default Mac mounting convenience only. An explicit prefix pins even an
    # identity mapping, so a missing staged file cannot pass using host bytes.
    if not pinned and not resolved.is_file() and original.is_relative_to('/code'):
        resolved=resolve_path(Path.home()/'code'/original.relative_to('/code'))
    return resolved
