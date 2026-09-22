"""L3 machine model (design §08).

A machine is data: an exec-domain tree, a visibility table (what fence /
poll / memory type each event scope needs), a capability table (booleans and
enums, wrong = wrong code) and a cost table (numbers, wrong = slow).  Passes
and code generators only read this object; they never test the arch name.
A test enforces that (tests/test_no_arch_branches.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..ir.types import Scope

ARCH_DIR = Path(__file__).parent / "arch"


@dataclass
class ExecDomain:
    name: str
    count: int
    cus: int
    l2_mb: float = 0.0


@dataclass
class Visibility:
    release: list[str]          # C statements executed by the producer before arrive
    acquire: list[str]          # C statements executed by the consumer after wait
    poll: str                   # C expression template reading a counter: uses {ptr}
    arrive: str                 # C expression template atomically decrementing: uses {ptr}
    memory: str                 # any | uncached | fine_grained


@dataclass
class MachineModel:
    name: str
    family: str
    vendor: str
    wave_size: int
    devices_per_package: int
    exec_domains: list[ExecDomain]
    shared_cache: dict[str, Any]
    resources: dict[str, Any]
    bandwidth: dict[str, float]
    visibility: dict[Scope, Visibility]
    capabilities: dict[str, Any]
    costs: dict[str, Any]
    lowering: dict[str, str] = field(default_factory=dict)   # misc C snippets (backoff, domain_id, timer)
    notes: list[str] = field(default_factory=list)

    # ---- topology -------------------------------------------------------
    @property
    def leaf_domain(self) -> ExecDomain:
        return self.exec_domains[-1]

    @property
    def num_domains(self) -> int:
        n = 1
        for d in self.exec_domains:
            n *= d.count
        return n

    @property
    def has_domain_level(self) -> bool:
        return self.num_domains > 1

    def cus_per_domain(self) -> int:
        return self.leaf_domain.cus

    def total_cus(self) -> int:
        return self.num_domains * self.cus_per_domain()

    # ---- scopes ---------------------------------------------------------
    def effective_scope(self, scope: Scope) -> Scope:
        """Collapse scopes the machine does not distinguish (no domain level ->
        DOMAIN becomes DEVICE; no cluster support -> CLUSTER becomes DOMAIN/DEVICE)."""
        s = scope
        if s == Scope.CLUSTER and not self.capabilities.get("cluster_launch", False):
            s = Scope.DOMAIN
        if s == Scope.DOMAIN and not self.has_domain_level:
            s = Scope.DEVICE
        if s == Scope.WORKGROUP:
            s = Scope.DOMAIN if self.has_domain_level else Scope.DEVICE
        return s

    def vis(self, scope: Scope) -> Visibility:
        s = self.effective_scope(scope)
        if s not in self.visibility:
            raise KeyError(f"{self.name}: no visibility entry for scope {s.name}; the lowering table must cover it")
        return self.visibility[s]

    def memory_for(self, scope: Scope) -> str:
        m = self.vis(scope).memory
        return "device" if m == "any" else m

    # ---- costs (microseconds) -----------------------------------------
    def t_sync_us(self, scope: Scope) -> float:
        s = self.effective_scope(scope)
        key = {Scope.DOMAIN: "t_local_ns", Scope.DEVICE: "t_cross_ns", Scope.SYSTEM: "t_dev_ns"}[s]
        if s == Scope.DEVICE and not self.has_domain_level:
            key = "t_local_ns"
        v = self.costs.get(key)
        if v is None:
            # uncalibrated: fall back to a pessimistic guess and say so via notes
            fallback = {"t_local_ns": 200, "t_cross_ns": 800, "t_dev_ns": 10000}[key]
            self.notes.append(f"cost {key} uncalibrated on {self.name}; using {fallback} ns")
            self.costs[key] = fallback
            v = fallback
        return float(v) / 1000.0

    def t_pop_us(self) -> float:
        return float(self.costs.get("t_pop_ns", 300)) / 1000.0

    def t_push_us(self, cross_domain: bool) -> float:
        base = float(self.costs.get("t_push_ns", 300)) / 1000.0
        return base + (self.t_sync_us(Scope.DEVICE) if cross_domain else 0.0)

    def cache_bw_bytes_per_us(self) -> float:
        return float(self.bandwidth.get("cache_bw_tbs", self.bandwidth["hbm_tbs"])) * 1e6

    def hbm_bw_bytes_per_us(self) -> float:
        return float(self.bandwidth["hbm_tbs"]) * 1e6

    def supports(self, cap: str) -> Any:
        return self.capabilities.get(cap, False)


def _load_visibility(raw: dict[str, Any]) -> dict[Scope, Visibility]:
    out: dict[Scope, Visibility] = {}
    for k, v in raw.items():
        out[Scope[k.upper()]] = Visibility(
            release=list(v.get("release", [])), acquire=list(v.get("acquire", [])),
            poll=v["poll"], arrive=v["arrive"], memory=v.get("memory", "any"))
    return out


def load_machine(name_or_path: str | Path) -> MachineModel:
    p = Path(name_or_path)
    if not p.exists():
        p = ARCH_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no machine model {name_or_path!r}; known: {list_archs()}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return MachineModel(
        name=raw["name"], family=raw["family"], vendor=raw["vendor"], wave_size=int(raw["wave_size"]),
        devices_per_package=int(raw.get("devices_per_package", 1)),
        exec_domains=[ExecDomain(**d) for d in raw["exec_domains"]],
        shared_cache=raw.get("shared_cache", {}), resources=raw["resources"], bandwidth=raw["bandwidth"],
        visibility=_load_visibility(raw["visibility"]), capabilities=raw.get("capabilities", {}),
        costs=dict(raw.get("costs", {})), lowering=raw.get("lowering", {}), notes=list(raw.get("notes", [])),
    )


def list_archs() -> list[str]:
    return sorted(p.stem for p in ARCH_DIR.glob("*.yaml"))
