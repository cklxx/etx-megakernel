"""Versioned declarations; no compiler, filesystem or execution dependencies."""
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Literal


SCHEMA_VERSION = 1


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError('configuration keys must be strings')
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ValueError(f'configuration requires finite JSON values, got {type(value).__name__}')


@dataclass(frozen=True, init=False)
class Configuration(Mapping):
    """Immutable JSON value store; returned containers are detached snapshots."""
    _json: str

    def __init__(self, values: Mapping):
        if not isinstance(values, Mapping):
            raise ValueError('configuration must be a mapping')
        object.__setattr__(self, '_json', json.dumps(
            _json_value(values), sort_keys=True, separators=(',', ':'), allow_nan=False))

    def to_dict(self) -> dict:
        return json.loads(self._json)

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def _name(value: str):
    if type(value) is not str or not value or value != value.strip():
        raise ValueError('name must be a nonempty string without surrounding whitespace')


@dataclass(frozen=True)
class KernelSpec:
    name: str
    implementation: Literal['graph', 'imported', 'native']
    config: Configuration

    def __post_init__(self):
        _name(self.name)
        if self.implementation not in ('graph', 'imported', 'native'):
            raise ValueError('unknown kernel implementation')
        object.__setattr__(self, 'config', Configuration(self.config))


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    kind: Literal['model', 'graph', 'imported', 'native']
    config: Configuration
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        _name(self.name)
        if self.kind not in ('model', 'graph', 'imported', 'native'):
            raise ValueError('unknown workload kind')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported workload schema')
        object.__setattr__(self, 'config', Configuration(self.config))

    def to_dict(self) -> dict:
        return dict(schema_version=self.schema_version, name=self.name,
                    kind=self.kind, config=self.config.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping) -> 'WorkloadSpec':
        if not isinstance(value, Mapping) or set(value) != {'schema_version', 'name', 'kind', 'config'}:
            raise ValueError('workload record requires exactly schema_version/name/kind/config')
        return cls(**value)

    @property
    def identity(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class PlanRequest:
    workload: WorkloadSpec
    target: str = 'gfx942'
    bindings: Configuration = field(default_factory=lambda: Configuration({}))
    runtime: Configuration = field(default_factory=lambda: Configuration({}))
    options: Configuration = field(default_factory=lambda: Configuration({}))

    def __post_init__(self):
        if not isinstance(self.workload, WorkloadSpec):
            raise TypeError('plan request requires WorkloadSpec')
        _name(self.target)
        for name in ('bindings', 'runtime', 'options'):
            object.__setattr__(self, name, Configuration(getattr(self, name)))
        if any(type(value) is not int or value < 0 for value in self.bindings.values()):
            raise ValueError('bindings require nonnegative integers')

    def to_dict(self) -> dict:
        return dict(workload=self.workload.to_dict(), target=self.target,
                    bindings=self.bindings.to_dict(), runtime=self.runtime.to_dict(),
                    options=self.options.to_dict())


@dataclass(frozen=True)
class PlannedWorkload:
    """Detached planner output; not an executable or a correctness certificate."""
    request: PlanRequest
    machine: Configuration
    options: Configuration
    report: Configuration
    decisions: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.request, PlanRequest):
            raise TypeError('planned workload requires PlanRequest')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported plan schema')
        for name in ('machine', 'options', 'report'):
            object.__setattr__(self, name, Configuration(getattr(self, name)))
        if any(type(line) is not str for line in self.decisions):
            raise ValueError('decisions require strings')
        object.__setattr__(self, 'decisions', tuple(self.decisions))

    def to_dict(self) -> dict:
        return dict(schema_version=self.schema_version, request=self.request.to_dict(),
                    machine=self.machine.to_dict(), options=self.options.to_dict(),
                    report=self.report.to_dict(), decisions=list(self.decisions))

    @classmethod
    def from_dict(cls, value: Mapping) -> 'PlannedWorkload':
        fields = {'schema_version', 'request', 'machine', 'options', 'report', 'decisions'}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError('invalid planned workload record')
        request = dict(value['request'])
        request['workload'] = WorkloadSpec.from_dict(request['workload'])
        return cls(**{**value, 'request': PlanRequest(**request)})

    @property
    def identity(self) -> str:
        return hashlib.sha256(Configuration(self.to_dict())._json.encode()).hexdigest()


@dataclass(frozen=True)
class SimulationResult:
    plan_identity: str
    seed: int
    metrics: Configuration
    deadlock: bool
    blocked: tuple[str, ...]
    backend: Literal['simulation'] = 'simulation'

    def __post_init__(self):
        if self.backend != 'simulation':
            raise ValueError('simulation results cannot claim a device backend')
        if type(self.seed) is not int or type(self.deadlock) is not bool:
            raise ValueError('invalid simulation result')
        object.__setattr__(self, 'metrics', Configuration(self.metrics))
        object.__setattr__(self, 'blocked', tuple(self.blocked))


@dataclass(frozen=True)
class Inspection:
    schema_version: int
    identity: str
    declaration: WorkloadSpec
    status: Literal['DECLARED'] = 'DECLARED'
