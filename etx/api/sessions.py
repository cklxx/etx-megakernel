"""Model experiment sessions: the prepared, fail-closed GPU session of one model, one operation per request."""
from dataclasses import dataclass, field
from pathlib import Path

from .types import Configuration, SCHEMA_VERSION


# Operations and the inputs each provider requires.
PROVIDERS = {'dsv2lite': dict(operations=('source', 'preflight', 'build', 'pairs', 'trace', 'layers', 'd4trace', 'report'),
                              inputs=('fleet', 'data'))}


@dataclass(frozen=True)
class ExperimentSessionRequest:
    provider: str
    operation: str
    logs: str                          # the session record directory (historically ETX_SESS_LOGDIR)
    root: str                          # the (staged) repository whose declarations the session runs
    inputs: Configuration = field(default_factory=lambda: Configuration({}))   # absolute input paths by name
    arm: str | None = None
    stamped: bool = False
    reference_dir: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        spec = PROVIDERS.get(self.provider)
        if spec is None:
            raise ValueError(f'unknown experiment session provider {self.provider!r}')
        if self.operation not in spec['operations']:
            raise ValueError(f'{self.provider} has no operation {self.operation!r}')
        object.__setattr__(self, 'inputs', Configuration(self.inputs))
        if set(self.inputs) != set(spec['inputs']):
            raise ValueError(f'{self.provider} sessions need inputs {list(spec["inputs"])}')
        for value in (self.logs, self.root, *self.inputs.values(), *([self.reference_dir] if self.reference_dir else [])):
            if type(value) is not str or not Path(value).is_absolute():
                raise ValueError('session paths must be absolute')
        if (self.operation == 'layers') != (self.reference_dir is not None):
            raise ValueError('layers requires reference_dir, and only layers takes it')
        if type(self.stamped) is not bool or (self.stamped and self.operation != 'build'):
            raise ValueError('stamped applies to build only')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported experiment session schema')
