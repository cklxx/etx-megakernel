"""Shape-specific contracts for the four checked-in bf16 decode MoE captures.

These are conservative supported shapes, not a promise that every legal Triton
shape has been captured. Both graph construction and import use this registry.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import re
from pathlib import Path

REGISTRY = Path(__file__).with_name("specs") / "moe-decode-v1.json"
CAPTURE_ROOT = Path(__file__).resolve().parents[2] / "examples/vllm_llm/triton"


@dataclass(frozen=True)
class MoeLaunch:
    export: str
    grid: tuple[int, int, int]
    block_m: int
    block_n: int
    block_k: int
    n: int
    k: int
    top_k: int
    multiply_routed_weight: bool
    reads: tuple[str, ...]
    writes: tuple[str, ...] = ("output",)
    layout: str = "row_major"
    dtype: str = "bf16"
    aliases: tuple = ()

    def validate(self, *, grid, n, k, top_k, block_m):
        expected = (self.grid, self.n, self.k, self.top_k, self.block_m)
        observed = (tuple(grid), n, k, top_k, block_m)
        if observed != expected:
            raise ValueError(f"{self.export}: launch conflicts with captured specification: "
                             f"expected grid/N/K/top_k/BM={expected}, got {observed}")


def registry():
    value = json.loads(REGISTRY.read_text())
    if value["schema_version"] != 1:
        raise ValueError("unsupported MoE specification version")
    return value["captures"]


def verify_capture(export, root=CAPTURE_ROOT):
    records = registry()
    if export not in records:
        raise ValueError(f"{export}: no captured operator specification")
    record = records[export]
    for filename, expected in record["sha256"].items():
        path = root / export / filename
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{export}: capture fingerprint mismatch for {filename}; review and recapture")
    meta = json.loads((root / export / "fused_moe_kernel.json").read_text())
    if (meta["target"]["arch"] != record["arch"] or meta["shared"] != record["shared_bytes"]
            or meta["num_warps"] != record["num_warps"]):
        raise ValueError(f"{export}: incompatible target or shared-memory metadata")
    ttir = (root / export / "fused_moe_kernel.ttir").read_text()
    dots = re.findall(r"tt\.dot[^\n]*: tensor<(\d+)x(\d+)xbf16> \* tensor<(\d+)x(\d+)xbf16> -> tensor<(\d+)x(\d+)xf32>", ttir)
    c = record["constants"]
    bm, bn, bk = (c[key] for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"))
    if not dots or any(tuple(map(int, dot)) != (bm, bk, bk, bn, bm, bn) for dot in dots):
        raise ValueError(f"{export}: block constants conflict with captured TTIR dot shapes")
    return record


def launch_for(dims, role, *, batch=1):
    if role not in ("up", "down") or batch != 1:
        raise ValueError("captured MoE supports up/down with forward batch=1 only")
    shape = {"H": dims.H, "MI": dims.MI, "E": dims.E, "TOPK": dims.TOPK}
    for export, record in registry().items():
        if record["shape"] == shape and record["role"] == role:
            verify_capture(export)
            constants = record["constants"]
            n, k = (2 * dims.MI, dims.H) if role == "up" else (dims.H, dims.MI)
            bn = constants["BLOCK_SIZE_N"]
            reads = ("input", "weights", "expert_ids", "num_tokens_post_padded")
            if role == "down":
                reads += ("routing_weights",)
            return MoeLaunch(export, (dims.TOPK * ((n + bn - 1) // bn), 1, 1),
                             constants["BLOCK_SIZE_M"], bn, constants["BLOCK_SIZE_K"],
                             n, k, dims.TOPK, role == "down", reads)
    raise ValueError(f"no captured bf16 MoE {role} specification for {shape}; recapture required")


def validate_wvsplitk(*, grid, cu_count, rows, ytile, wave_groups):
    """Reject the historical lm_head row-stride hole before launching a kernel."""
    if len(grid) != 3 or any(type(x) is not int or x <= 0 for x in grid):
        raise ValueError("wvSplitK: grid must contain three positive integers")
    if tuple(grid[1:]) != (1, 1):
        raise ValueError("wvSplitK: only a one-dimensional grid is supported (kernel uses blockIdx.x)")
    if (type(rows) is not int or rows <= 0 or type(ytile) is not int or ytile not in (1, 2)
            or type(wave_groups) is not int or not 1 <= wave_groups <= 16):
        raise ValueError("wvSplitK: unsupported rows/ytile/wave-group configuration")
    if type(cu_count) is not int or cu_count != grid[0]:
        raise ValueError("wvSplitK: CuCount must equal launched blocks, otherwise output rows may be unwritten")


def model_specs(config):
    from examples.llm.model import load_dims
    dims = load_dims(str(config))
    return {role: asdict(launch_for(dims, role)) for role in ("up", "down")}
