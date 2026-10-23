"""Static source-level routed tile reads, not compiler loads or HBM traffic.

See FC2A-ROUTED-2026-10-24.md for source anchors and derivation. Counts
include padded gate/up pairs, which execute even when their result is dropped.
"""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TileReads:
    tile: int
    rows: tuple
    payload_bytes: int
    scale_bytes: int

    @property
    def total_bytes(self):
        return self.payload_bytes + self.scale_bytes


def routed_tiles(record, *, projection, tiles, selected_slot=0, topk=2,
                 path='decode_device', block=128):
    """Symbolically enumerate row ownership for one selected expert matrix.

    decode_* follows phases.h (gate/up pairs and down rows). verify follows
    verify_moe.h: partition one expert, no padding, weights outside P loop.
    native follows native.h glm_mv, one matrix invocation. CPU and verify
    use scalar glm_w; aligned decode_device uses one scale per 16 FP8 bytes.
    Scale counts are lookup bytes, not distinct scale storage.
    """
    if projection not in ('gate', 'up', 'down'):
        raise ValueError('invalid projection')
    if path not in ('decode_cpu', 'decode_device', 'verify', 'native'):
        raise ValueError('invalid path')
    if type(tiles) is not int or tiles < 1:
        raise ValueError('tiles must be positive')
    if not 0 <= selected_slot <= topk:
        raise ValueError('invalid selected slot')
    if type(block) is not int or block < 1:
        raise ValueError('block must be positive')
    rows, cols = record['shape']
    item = {'torch.float8_e4m3fn': 1, 'torch.bfloat16': 2,
            'torch.float32': 4}[record['dtype']]
    fp8 = record['dtype'] == 'torch.float8_e4m3fn'
    vector = (path == 'decode_device' and cols*item % 16 == 0
              and record.get('offset', 0) % 16 == 0
              and (not fp8 or block % 16 == 0))
    result = []
    for tile in range(tiles):
        if path.startswith('decode') and projection in ('gate', 'up'):
            lo = (topk+1)*rows*tile//tiles
            hi = (topk+1)*rows*(tile+1)//tiles
            owned = []
            for first in range(lo, hi, 2):
                # phases.h pr[i]: the odd tail duplicates the first row.
                for pair in (first, first+1 if first+1 < hi else first):
                    if pair//rows == selected_slot:
                        owned.append(pair % rows)
        else:
            lo, hi = rows*tile//tiles, rows*(tile+1)//tiles
            owned = list(range(lo, hi))
        elements = len(owned)*cols
        scales = elements//16 if vector and fp8 else elements if fp8 else 0
        result.append(TileReads(tile, tuple(owned), elements*item, 4*scales))
    return result


def row_multiplicity(tiles):
    """Each count applies to every payload byte in that output row."""
    return Counter(row for tile in tiles for row in tile.rows)


def iape_lookups(*, pos, pool, index_topk, width, tiles=1):
    """phases.h MLA: three scalar IAPE passes per pool, on lane 0 per tile."""
    if min(pool, width, tiles) < 1 or min(pos, index_topk) < 0:
        raise ValueError('invalid pool dimensions or position')
    pools = (pos+1)//pool
    return 0 if pools <= index_topk//pool else 3*pools*pool*width*tiles
