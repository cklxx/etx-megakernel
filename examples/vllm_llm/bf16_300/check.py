"""Compare stage snapshots against independent ORIGINAL constituent launches.

The device check adapter captures byte ranges before the next in-place stage.
A reference must be produced by the original constituent launch, never by this
fused adapter. No certificate is issued here (full imports/tokens remain gates).
"""
import hashlib
from dataclasses import dataclass
from .protocol import require


@dataclass(frozen=True)
class Snapshot:
    constituent: str
    partition: int
    original_sha256: str
    payload: bytes


def compare(original, candidate):
    require(bool(original) and len(original) == len(candidate), "snapshot_coverage")
    for ref, got in zip(original, candidate):
        require((ref.constituent, ref.partition, ref.original_sha256) ==
                (got.constituent, got.partition, got.original_sha256), "snapshot_identity")
        require(ref.payload == got.payload, "snapshot_bytes")


def capture(constituent, partition, original_binary, buffers):
    # bytes copies NOW; mutable qkv/cache/output buffers may change afterwards.
    return Snapshot(constituent, partition, hashlib.sha256(original_binary).hexdigest(),
                    b"".join(bytes(b) for b in buffers))
