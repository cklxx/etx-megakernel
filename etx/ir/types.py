"""Core IR types (design §07).

Everything here is hardware-independent data.  Scope, domain_id and memory
placement on events are filled in by L4 passes, never by the frontend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .dims import Dim, free_symbols
from .edgemap import EdgeMap


class Scope(IntEnum):
    WORKGROUP = 0
    CLUSTER = 1
    DOMAIN = 2      # one XCD / GCD / die: the level ETC does not have
    DEVICE = 3
    SYSTEM = 4


class InitKind(IntEnum):
    STATIC = 0      # constant wait_count
    PER_STEP = 1    # host writes the counts each step
    RUNTIME = 2     # a task grid writes the counts (MoE routing)


@dataclass
class Tensor:
    name: str
    shape: tuple[Dim, ...]
    dtype: str = "bf16"
    role: str = "activation"        # activation | weight | runtime | scratch
    bytes_per_elem: int = 2

    def nbytes(self, bindings) -> int:
        from .dims import eval_shape
        n = 1
        for d in eval_shape(self.shape, bindings):
            n *= d
        return n * self.bytes_per_elem


@dataclass
class ETensor:
    name: str
    shape: tuple[Dim, ...]
    wait_count: Dim | str = 1       # int, symbolic expr, or "runtime"
    init_kind: InitKind = InitKind.STATIC
    runtime_init_by: str | None = None      # task grid that writes the counts
    runtime_count: str | None = None        # expression over event coords (i, j, ...) and runtime tensors,
                                            # e.g. "expert_counts[i]"; the init grid's epilogue copies it into the counters
    # filled by passes
    scope: Scope | None = None
    domain_id: int | None = None
    memory: str | None = None
    epoch: bool = False

    @property
    def is_runtime_count(self) -> bool:
        return self.wait_count == "runtime" or self.init_kind == InitKind.RUNTIME


@dataclass
class Resource:
    threads: int = 256
    lds_bytes: int = 0
    vgpr: int = 128
    agpr: int = 0
    tensor_core: bool = False
    prefetch_bytes: int = 0         # reserved for cross-barrier prefetch (Pass 7)


@dataclass
class TileBody:
    kind: str                       # hip_link | cuda_link | triton | builtin
    symbol: str                     # exported device function (link) or jit fn (triton)
    source: str | None = None       # path to source file
    prefetch: str | None = None     # optional device function that warms this tile's weights (Pass 7 hook)


@dataclass
class TaskGrid:
    name: str
    grid: tuple[Dim, ...]
    body: TileBody
    resource: Resource = field(default_factory=Resource)
    args: list[str] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    in_edges: dict[str, EdgeMap] = field(default_factory=dict)
    out_edges: dict[str, EdgeMap] = field(default_factory=dict)
    # optimisation annotations
    pure: bool = False
    out_to_in: str | None = None
    weight_args: list[str] = field(default_factory=list)
    domain_map: str | None = None   # pin tasks to exec domains, e.g. "xw->x" (fleet's Chiplet-tasks: one per XCD, 37 workers each)
    bytes_per_tile: Dim = 0
    duration_us: float = 1.0
    duration_cv: float = 0.0        # coefficient of variation of tile duration
    device: int = 0
    # filled by passes
    prologue: list[tuple[str, str]] = field(default_factory=list)   # (inlined producer grid, out_to_in map) from Pass 3

    @property
    def has_runtime_edges(self) -> bool:
        return any(m.is_runtime for m in list(self.in_edges.values()) + list(self.out_edges.values()))


@dataclass
class Graph:
    name: str
    tensors: dict[str, Tensor] = field(default_factory=dict)
    events: dict[str, ETensor] = field(default_factory=dict)
    grids: list[TaskGrid] = field(default_factory=list)     # program order == topological order
    symbols: list[str] = field(default_factory=list)

    # builder API -----------------------------------------------------------
    def tensor(self, name: str, shape: tuple[Dim, ...], **kw) -> Tensor:
        t = Tensor(name, tuple(shape), **kw)
        self.tensors[name] = t
        self._collect(shape)
        return t

    def etensor(self, name: str, shape: tuple[Dim, ...], wait_count: Dim | str = 1, **kw) -> ETensor:
        e = ETensor(name, tuple(shape), wait_count, **kw)
        if wait_count == "runtime" or e.runtime_count is not None:
            e.init_kind = InitKind.RUNTIME
            e.wait_count = "runtime"
        self.events[name] = e
        self._collect(shape)
        if wait_count != "runtime":
            self._collect((wait_count,))
        return e

    def call_device(self, name: str, grid: tuple[Dim, ...], body: TileBody, *,
                    in_edges: dict[str, str] | None = None,
                    out_edges: dict[str, str] | None = None, **kw) -> TaskGrid:
        g = TaskGrid(name=name, grid=tuple(grid), body=body,
                     in_edges={k: EdgeMap.parse(v) for k, v in (in_edges or {}).items()},
                     out_edges={k: EdgeMap.parse(v) for k, v in (out_edges or {}).items()}, **kw)
        if any(gg.name == name for gg in self.grids):
            raise ValueError(f"duplicate task grid {name!r}")
        self.grids.append(g)
        self._collect(grid)
        return g

    def grid(self, name: str) -> TaskGrid:
        for g in self.grids:
            if g.name == name:
                return g
        raise KeyError(name)

    def _collect(self, dims) -> None:
        for d in dims:
            for s in free_symbols(d):
                if s not in self.symbols:
                    self.symbols.append(s)

    def producers_of(self, event: str) -> list[TaskGrid]:
        return [g for g in self.grids if event in g.out_edges]

    def consumers_of(self, event: str) -> list[TaskGrid]:
        return [g for g in self.grids if event in g.in_edges]

    def readers_of(self, tensor: str) -> list[TaskGrid]:
        return [g for g in self.grids if tensor in g.reads or tensor in g.args]
