"""Host-side argument blocks for imported kernels.

The layout is the one ir_import computed: every kernel parameter at its natural alignment in declaration
order, then u32 grid[3] and u32 nblocks at `grid_offset`. `pack(info, values, grid)` returns the bytes for
one launch; values are given in parameter order (ints, floats, or pointers as ints). The C++ side of the
example uses the same offsets from imports.json, so both agree by construction.
"""
from __future__ import annotations

import struct

_FMT = {"i8": "b", "i16": "h", "i32": "i", "i64": "q", "float": "f", "double": "d", "half": "e"}


def pack(info: dict, values: list, grid: tuple[int, int, int]) -> bytes:
    params = info["params"]
    if len(values) != len(params):
        raise ValueError(f"{info['export']}: {len(values)} values for {len(params)} parameters")
    buf = bytearray(info["arg_bytes"])
    for p, v in zip(params, values):
        fmt = "Q" if p["ty"].startswith("ptr") else _FMT[p["ty"]]
        struct.pack_into("<" + fmt, buf, p["offset"], v)
    g = info["grid_offset"]
    struct.pack_into("<IIII", buf, g, grid[0], grid[1], grid[2], grid[0] * grid[1] * grid[2])
    return bytes(buf)
