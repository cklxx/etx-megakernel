"""L1 tile contract helpers.

A frontend answers three questions per operator (coordinate space, dependency
maps, resource needs) and hands over a tile body handle.  These helpers build
the handle; the questions are answered in Graph.call_device.
"""
from __future__ import annotations

from ..ir.types import TileBody


def hip_link(source: str, symbol: str) -> TileBody:
    """Link mode: `symbol` is an extern "C" __device__ function with the etx_ctx ABI in `source`."""
    return TileBody(kind="hip_link", symbol=symbol, source=source)


def cuda_link(source: str, symbol: str) -> TileBody:
    return TileBody(kind="cuda_link", symbol=symbol, source=source)


def triton_fn(module: str, name: str) -> TileBody:
    """Host-DSL mode: a @triton.jit function; ETX emits a Triton persistent kernel that calls it."""
    return TileBody(kind="triton", symbol=name, source=module)


def builtin(name: str) -> TileBody:
    """Compiler-provided tile (reduction, copy); resolved by the backend."""
    return TileBody(kind="builtin", symbol=f"etx_builtin_{name}", source=None)
