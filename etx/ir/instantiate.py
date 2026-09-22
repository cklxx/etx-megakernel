"""Instantiate a symbolic graph for concrete bindings.

The result is the fully enumerated task/event graph for one step: every task
coordinate, every event coordinate, and who notifies / waits on what.  It is
the common input of verification, placement, the simulator and (for dynamic
consumer lists) the plan.  Nothing here is materialised in the generated
kernel; the kernel recomputes the same relations from the edge maps.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Mapping

from .dims import eval_dim, eval_shape
from .types import Graph, TaskGrid

TaskId = tuple[str, tuple[int, ...]]
EvId = tuple[str, tuple[int, ...]]


@dataclass
class Instance:
    bindings: dict[str, int]
    grid_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    event_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    tasks: dict[str, list[tuple[int, ...]]] = field(default_factory=dict)
    # event coordinate -> producers / consumers (task ids)
    producers: dict[EvId, list[TaskId]] = field(default_factory=dict)
    consumers: dict[EvId, list[TaskId]] = field(default_factory=dict)
    # task -> in / out event coordinates
    task_in: dict[TaskId, list[EvId]] = field(default_factory=dict)
    task_out: dict[TaskId, list[EvId]] = field(default_factory=dict)
    wait_counts: dict[EvId, int] = field(default_factory=dict)

    def n_tasks(self) -> int:
        return sum(len(v) for v in self.tasks.values())

    def all_tasks(self) -> list[TaskId]:
        return [(g, c) for g, cs in self.tasks.items() for c in cs]


def _coords(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    return list(itertools.product(*[range(n) for n in shape]))


def instantiate(graph: Graph, bindings: Mapping[str, int],
                runtime: Mapping[str, Any] | None = None) -> Instance:
    inst = Instance(bindings=dict(bindings))
    runtime = runtime or {}
    for e in graph.events.values():
        inst.event_shapes[e.name] = eval_shape(e.shape, bindings)
        for c in _coords(inst.event_shapes[e.name]):
            inst.producers[(e.name, c)] = []
            inst.consumers[(e.name, c)] = []
    for g in graph.grids:
        shape = eval_shape(g.grid, bindings)
        inst.grid_shapes[g.name] = shape
        inst.tasks[g.name] = _coords(shape)
        for c in inst.tasks[g.name]:
            tid: TaskId = (g.name, c)
            inst.task_in[tid] = []
            inst.task_out[tid] = []
            for ev, m in g.out_edges.items():
                for t in m.targets(c, inst.event_shapes[ev], runtime, bindings):
                    inst.task_out[tid].append((ev, t))
                    inst.producers[(ev, t)].append(tid)
            for ev, m in g.in_edges.items():
                for t in m.targets(c, inst.event_shapes[ev], runtime, bindings):
                    inst.task_in[tid].append((ev, t))
                    inst.consumers[(ev, t)].append(tid)
    for e in graph.events.values():
        for c in _coords(inst.event_shapes[e.name]):
            if e.is_runtime_count:
                inst.wait_counts[(e.name, c)] = len(inst.producers[(e.name, c)])
            else:
                inst.wait_counts[(e.name, c)] = eval_dim(e.wait_count, bindings)
    return inst
