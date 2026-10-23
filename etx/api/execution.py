"""Bounded process requests and evidence; exit zero is not qualification."""
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path

from .types import Configuration, SCHEMA_VERSION


@dataclass(frozen=True)
class ProcessRequest:
    argv: tuple[str, ...]
    cwd: str
    deadline_s: float
    environment: Configuration = field(default_factory=lambda: Configuration({}))
    silence_s: float = 30.0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if isinstance(self.argv, str) or not self.argv:
            raise ValueError('argv requires a nonempty sequence')
        object.__setattr__(self, 'argv', tuple(self.argv))
        if any(type(arg) is not str or '\0' in arg for arg in self.argv):
            raise ValueError('argv requires strings without NUL')
        if not Path(self.argv[0]).is_absolute():
            raise ValueError('executable must be absolute')
        if type(self.cwd) is not str or '\0' in self.cwd or not Path(self.cwd).is_absolute():
            raise ValueError('cwd must be absolute')
        for value in (self.deadline_s, self.silence_s):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('deadline and silence intervals must be positive finite seconds')
        object.__setattr__(self, 'environment', Configuration(self.environment))
        for key, value in self.environment.items():
            if not key or '=' in key or '\0' in key or type(value) is not str or '\0' in value:
                raise ValueError('environment requires valid string keys and values')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported process schema')

    def to_dict(self):
        return dict(schema_version=self.schema_version, argv=list(self.argv), cwd=self.cwd,
                    deadline_s=self.deadline_s, silence_s=self.silence_s,
                    environment=self.environment.to_dict())

    @property
    def identity(self):
        return hashlib.sha256(Configuration(self.to_dict())._json.encode()).hexdigest()


@dataclass(frozen=True)
class ProcessResult:
    request_identity: str
    backend: str
    status: str
    returncode: int | None
    output_sha256: str
    events: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.backend not in ('host-process', 'scripted', 'recorded'):
            raise ValueError('process results are not device qualification')
        if self.status not in ('complete', 'failed', 'deadline', 'interrupted', 'error'):
            raise ValueError('unknown process status')
        if self.returncode is not None and type(self.returncode) is not int:
            raise ValueError('invalid process return code')
        if (self.status == 'complete') != (self.returncode == 0):
            raise ValueError('completion requires exit zero and vice versa')
        if self.status == 'deadline' and self.returncode != 124:
            raise ValueError('deadline requires return code 124')
        for value in (self.request_identity, self.output_sha256):
            if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid evidence digest')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported process result schema')
        if isinstance(self.events, str) or any(type(event) is not str for event in self.events):
            raise ValueError('process events require strings')
        object.__setattr__(self, 'events', tuple(self.events))

    def to_dict(self):
        return dict(schema_version=self.schema_version, request_identity=self.request_identity,
                    backend=self.backend, status=self.status, returncode=self.returncode,
                    output_sha256=self.output_sha256, events=list(self.events))
