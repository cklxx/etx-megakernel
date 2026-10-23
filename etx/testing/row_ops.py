"""Historical row capture conformance; no Git, build, or device execution."""
from pathlib import Path

from etx.api.types import Configuration
from .historical import sealed_original_host


def row_capture_probe(directory):
    """Transform the hash-checked original host using the real capture seam."""
    from examples.experiment_program.row_ops.capture import checking_host
    from examples.experiment_program.row_ops.producer import BASE

    retained = sealed_original_host()
    if retained['provenance']['revision'] != BASE:
        raise ValueError('row capture baseline differs from historical host fixture')
    directory = Path(directory)
    original = directory / 'original.hip'
    original.write_bytes(retained['source'].encode('utf-8'))
    output = directory / 'host.hip'
    checking_host(original, output)
    return Configuration(dict(source=output.read_text(),
                              provenance=retained['provenance'],
                              historical_only=True, device_qualification=False))
