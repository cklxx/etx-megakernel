"""Immutable requests for explicitly staged, sealed experiment sessions."""
from dataclasses import dataclass

from .types import Configuration


def _path(value, name, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or not value or '\0' in value:
        raise ValueError(f'{name} requires a nonempty path string')


@dataclass(frozen=True)
class SealedPreparationRequest:
    inventory: Configuration
    destination: str
    output: str
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported sealed preparation schema')
        object.__setattr__(self, 'inventory', Configuration(self.inventory))
        _path(self.destination, 'destination')
        _path(self.output, 'output')


@dataclass(frozen=True)
class SealedSessionRequest:
    action: str
    config: str | None = None
    output: str | None = None
    records: tuple[str, ...] = ()
    resume: bool = False
    execute: bool = False
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported sealed session schema')
        if self.action not in ('list', 'preflight', 'dry-run', 'run', 'export'):
            raise ValueError('unknown sealed session action')
        _path(self.config, 'config', optional=self.action in ('list', 'export'))
        _path(self.output, 'output', optional=self.action == 'list')
        if isinstance(self.records, str):
            raise ValueError('records must be a sequence, not a string')
        records = tuple(self.records)
        if any(not isinstance(item, str) or item not in {f'G{i:02}' for i in range(1, 14)} for item in records):
            raise ValueError('unknown sealed session record')
        if len(set(records)) != len(records):
            raise ValueError('duplicate sealed session record')
        object.__setattr__(self, 'records', records)
        if type(self.resume) is not bool or type(self.execute) is not bool:
            raise ValueError('resume and execute must be booleans')
