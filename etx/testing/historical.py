"""Checksum-bound historical source fixtures; never fresh hardware evidence."""
import hashlib
import json
from pathlib import Path

from etx.api.types import Configuration


_SEALED_HOST_SHA = '28570234c305ab6608fe0e24126d3022beeab259cd0b232c39fe6ad1afb802d8'


def sealed_original_host():
    directory = Path(__file__).with_name('data') / 'sealed'
    source = (directory / 'original-host.hip').read_bytes()
    receipt = json.loads((directory / 'original-host.json').read_text())
    if (hashlib.sha256(source).hexdigest() != _SEALED_HOST_SHA or
            receipt['sha256'] != _SEALED_HOST_SHA or
            receipt['revision'] != 'b573664faa4e7988dff353bc03aca4ce8561a729' or
            receipt['path'] != 'examples/vllm_llm/host.hip'):
        raise ValueError('historical sealed host fixture identity mismatch')
    return Configuration(dict(source=source.decode(), provenance=receipt,
                              historical_only=True, device_qualification=False))
