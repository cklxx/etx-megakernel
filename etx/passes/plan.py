"""The Plan: everything L4 decides, as data, plus the decision log.

A Plan is per (graph, machine, bindings).  The generated kernel is per
(graph, machine); only the descriptor tables and queues in the plan change
between steps, and those are what the host uploads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ir.instantiate import Instance, TaskId
from ..ir.types import Graph, Scope
from ..machine.model import MachineModel


@dataclass
class PassOptions:
    event_elimination: bool = True
    prefetch: bool = True
    split_resource_classes: bool = False
    force_mode: str | None = None          # static | dynamic | hybrid (debug / experiments)
    affinity_imbalance: float = 1.25       # max load / mean load tolerated when following producers
    sentinel_signals: bool = False         # lower waits to sentinel polling where the capability exists
    inline_tiles: bool = False             # emit #include of the tile sources into the kernel TU (single-TU build, bodies inlinable)


@dataclass
class TaskInst:
    id: int
    grid: str
    coord: tuple[int, ...]
    type_id: int
    device: int = 0
    domain: int = 0
    worker: int | None = None
    mode: str = "static"
    duration_us: float = 1.0


@dataclass
class EventPlan:
    name: str
    shape: tuple[int, ...]
    scope: Scope
    memory: str = "device"
    offset: int = 0
    numel: int = 0
    cross_domain_edges: int = 0
    cross_device_edges: int = 0
    counts: list[int] = field(default_factory=list)
    runtime_init: bool = False
    share: dict[tuple[int, int], list[int]] = field(default_factory=dict)   # (device, domain) -> producers per linear coordinate


@dataclass
class Plan:
    arch: str
    graph: Graph
    inst: Instance
    machine: MachineModel
    bindings: dict[str, int]
    options: PassOptions
    runtime: Any = None                     # the runtime sample the plan was instantiated with
    tasks: list[TaskInst] = field(default_factory=list)
    task_index: dict[TaskId, int] = field(default_factory=dict)
    type_ids: dict[str, int] = field(default_factory=dict)
    events: dict[str, EventPlan] = field(default_factory=dict)
    modes: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    n_devices: int = 1
    n_domains: int = 1
    workers_per_domain: int = 1
    wg_per_cu: int = 1
    resource_classes: dict[str, str] = field(default_factory=dict)
    kernel_instances: list[dict[str, Any]] = field(default_factory=list)
    static_queues: dict[tuple[int, int], list[int]] = field(default_factory=dict)   # (device, worker) -> task ids
    local_queue_capacity: dict[tuple[int, int], int] = field(default_factory=dict)  # (device, domain)
    global_queue_capacity: dict[int, int] = field(default_factory=dict)             # device
    ev_consumers: dict[tuple[str, int], list[int]] = field(default_factory=dict)    # (event, linear) -> dynamic task ids
    tensor_placement: dict[str, str] = field(default_factory=dict)
    prefetch: list[dict[str, Any]] = field(default_factory=list)
    eliminated_events: list[str] = field(default_factory=list)
    inlined: dict[str, Any] = field(default_factory=dict)      # producer grids removed by Pass 3 (name -> TaskGrid)
    log: list[str] = field(default_factory=list)

    # helpers -----------------------------------------------------------------
    def say(self, msg: str) -> None:
        self.log.append(msg)

    def workers_per_device(self) -> int:
        return self.n_domains * self.workers_per_domain

    def worker_domain(self, worker: int) -> int:
        return worker // self.workers_per_domain

    def task(self, tid: TaskId) -> TaskInst:
        return self.tasks[self.task_index[tid]]

    def linear(self, event: str, coord: tuple[int, ...]) -> int:
        shape = self.events[event].shape
        idx = 0
        for d, c in zip(shape, coord):
            idx = idx * d + c
        return idx

    def summary(self) -> str:
        lines = [f"plan: graph={self.graph.name} arch={self.arch} bindings={self.bindings}",
                 f"  devices={self.n_devices} domains/device={self.n_domains} workers/domain={self.workers_per_domain} (wg/CU={self.wg_per_cu})",
                 f"  tasks={len(self.tasks)} events={len(self.events)} eliminated={self.eliminated_events}"]
        for g, m in self.modes.items():
            lines.append(f"  {g:24s} {m:8s} {self.reasons.get(g, '')}")
        for e in self.events.values():
            lines.append(f"  event {e.name:16s} scope={e.scope.name:7s} mem={e.memory:12s} numel={e.numel} cross_domain={e.cross_domain_edges} cross_device={e.cross_device_edges}")
        lines.append(f"  prefetch entries={len(self.prefetch)}")
        return "\n".join(lines)


def verify_plan(plan: Plan) -> list[str]:
    """Checks 5 and 6 of design §07 6.4 plus static-queue order safety."""
    errors: list[str] = []
    m = plan.machine
    # 5. scope monotonic
    for name, ep in plan.events.items():
        need = Scope.DOMAIN
        if ep.cross_device_edges:
            need = Scope.SYSTEM
        elif ep.cross_domain_edges:
            need = Scope.DEVICE
        if m.effective_scope(ep.scope) < m.effective_scope(need):
            errors.append(f"event {name}: scope {ep.scope.name} < required {need.name}")
        if ep.scope == Scope.SYSTEM and m.vis(Scope.SYSTEM).memory == "fine_grained" and ep.memory != "fine_grained":
            errors.append(f"event {name}: system scope on {m.name} requires fine_grained memory, got {ep.memory}")
    # 5b. globally scheduled grids need at least DEVICE scope on their events
    for g in plan.graph.grids:
        if plan.modes.get(g.name) == "dynamic":
            for ev in list(g.in_edges) + list(g.out_edges):
                if m.effective_scope(plan.events[ev].scope) < m.effective_scope(Scope.DEVICE):
                    errors.append(f"event {ev}: DOMAIN scope but {g.name} is dynamically scheduled across domains")
    # 6. co-residency
    max_workers = m.total_cus() * plan.wg_per_cu
    if plan.workers_per_device() > max_workers:
        errors.append(f"{plan.workers_per_device()} workers exceed co-resident capacity {max_workers}")
    # static queue order: a task must not sit behind a producer on the same worker
    pos: dict[int, tuple[tuple[int, int], int]] = {}
    for key, q in plan.static_queues.items():
        for i, tid in enumerate(q):
            pos[tid] = (key, i)
    for key, q in plan.static_queues.items():
        for i, tid in enumerate(q):
            t = plan.tasks[tid]
            for ev in plan.inst.task_in[(t.grid, t.coord)]:
                for p in plan.inst.producers[ev]:
                    pid = plan.task_index[p]
                    if pid in pos and pos[pid][0] == key and pos[pid][1] > i:
                        errors.append(f"worker {key}: task {tid} ({t.grid}{list(t.coord)}) queued before its producer {pid} -> deadlock")
    return errors
