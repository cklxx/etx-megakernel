"""Immutable source artifacts. Sources are not compiled or launchable builds."""
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath

from .types import Configuration, PlannedWorkload, SCHEMA_VERSION


def payload_path(value):
    if type(value) is not str or not value or '\\' in value or '\0' in value:
        raise ValueError('invalid artifact payload path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('artifact payload paths must be normalized and relative')
    if value == 'manifest.json':
        raise ValueError('manifest.json is reserved')
    return value


@dataclass(frozen=True)
class SourceArtifact:
    planned: PlannedWorkload
    files: Configuration
    rank_sources: tuple[str, ...]
    rank_headers: tuple[str, ...]
    external_sources: tuple[str, ...] = ()
    status: str = 'SOURCES'
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.planned, PlannedWorkload):
            raise TypeError('source artifact requires a planned workload')
        if self.status != 'SOURCES':
            raise ValueError('source artifact cannot claim compilation or execution')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported source artifact schema')
        object.__setattr__(self, 'files', Configuration(self.files))
        for name, text in self.files.items():
            payload_path(name)
            if type(text) is not str:
                raise ValueError('source payloads require text')
        for name in ('rank_sources', 'rank_headers', 'external_sources'):
            values = getattr(self, name)
            if isinstance(values, str) or any(type(value) is not str for value in values):
                raise ValueError('source inventories require sequences of strings')
            object.__setattr__(self, name, tuple(values))
        count = self.planned.report['devices']
        for names in (self.rank_sources, self.rank_headers):
            if len(names) != count or len(set(names)) != count or any(name not in self.files for name in names):
                raise ValueError('source artifact requires a unique source/header for every rank')
        if set(self.rank_sources) & set(self.rank_headers):
            raise ValueError('rank source and header payloads must be distinct')

    def manifest(self):
        return dict(schema_version=self.schema_version, status=self.status,
                    planned=self.planned.to_dict(), rank_sources=list(self.rank_sources),
                    rank_headers=list(self.rank_headers), external_sources=list(self.external_sources),
                    payloads={name: hashlib.sha256(text.encode()).hexdigest() for name, text in self.files.items()})

    @property
    def identity(self):
        return hashlib.sha256(Configuration(self.manifest())._json.encode()).hexdigest()
