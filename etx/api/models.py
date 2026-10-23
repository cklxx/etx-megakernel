"""Model build requests: plan a model workload and link its ranks with a host into one executable."""
from dataclasses import dataclass, field
from pathlib import Path

from .types import Configuration, PlanRequest, SCHEMA_VERSION


class BuildRefused(ValueError):
    """The requested build variant is rejected before anything is created (historical exit status 2)."""


@dataclass(frozen=True)
class ModelBuildRequest:
    plan: PlanRequest              # kind='model' workload, target and planner options
    root: str                      # repository the compiler runs in; relative sources resolve here
    output: str                    # directory receiving the emitted files and `run`
    host: str                      # host translation unit linked with the rank kernels
    flags: tuple[str, ...] = ()    # compiler flags, before the include directories
    includes: tuple[str, ...] = ()
    resource_usage: bool = False   # ask the compiler for per-kernel VGPR/scratch remarks
    compiler: str = 'hipcc'        # executable on PATH, or 'docker:<local image>'
    deadline_s: float = 1800.0
    extra_sources: tuple[str, ...] = ()   # translation units between the rank kernels and the host
    provider: str = 'host-link'           # model-specific preparation and gates: 'host-link' or 'dsv2lite'
    variant: Configuration = field(default_factory=lambda: Configuration({}))
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.plan, PlanRequest) or self.plan.workload.kind != 'model':
            raise ValueError('model build requires a PlanRequest for a kind="model" workload')
        for name in ('root', 'output', 'host'):
            value = getattr(self, name)
            if type(value) is not str or '\0' in value or not Path(value).is_absolute():
                raise ValueError(f'{name} must be an absolute path')
        object.__setattr__(self, 'variant', Configuration(self.variant))
        if type(self.provider) is not str or not self.provider:
            raise ValueError('provider must be a name')
        for name in ('flags', 'includes', 'extra_sources'):
            values = getattr(self, name)
            if isinstance(values, str) or any(type(v) is not str or '\0' in v for v in values):
                raise ValueError(f'{name} requires a sequence of strings')
            object.__setattr__(self, name, tuple(values))
        if any(not Path(path).is_absolute() for path in self.includes):
            raise ValueError('include directories must be absolute paths')
        if type(self.compiler) is not str or not self.compiler or self.compiler.startswith('-'):
            raise ValueError('compiler must be an executable name or docker:<image>')
        if type(self.deadline_s) not in (int, float) or not self.deadline_s > 0:
            raise ValueError('deadline must be positive seconds')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported model build schema')
