"""Native microbenchmark probes (bench/): build, run and check them through one request."""
from dataclasses import dataclass, field
from pathlib import Path

from .types import Configuration, SCHEMA_VERSION


# Probe -> operations it supports.
PROBES = {'calibration': ('run',), 'fresh-handoff': ('plan', 'run', 'analyze')}


@dataclass(frozen=True)
class NativeProbeRequest:
    probe: str
    operation: str
    root: str                      # repository holding bench/ sources
    arguments: Configuration = field(default_factory=lambda: Configuration({}))
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.operation not in PROBES.get(self.probe, ()):
            raise ValueError(f'unknown probe operation {self.probe!r}/{self.operation!r}')
        if type(self.root) is not str or not Path(self.root).is_absolute():
            raise ValueError('root must be an absolute path')
        object.__setattr__(self, 'arguments', Configuration(self.arguments))
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported native probe schema')
