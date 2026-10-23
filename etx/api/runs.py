"""Artifact-bound process records; completion is never device qualification."""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from .execution import ProcessRequest, ProcessResult
from .types import Configuration, SCHEMA_VERSION


@dataclass(frozen=True)
class ArtifactRunRequest:
    directory: str
    build_identity: str
    cwd: str
    deadline_s: float
    arguments: tuple[str, ...] = ()
    environment: Configuration = field(default_factory=lambda: Configuration({}))
    silence_s: float = 30.0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if type(self.directory) is not str or '\0' in self.directory or not Path(self.directory).is_absolute():
            raise ValueError('artifact directory must be absolute')
        if (type(self.build_identity) is not str or len(self.build_identity) != 64 or
                any(c not in '0123456789abcdef' for c in self.build_identity)):
            raise ValueError('invalid build identity')
        if isinstance(self.arguments, str):
            raise ValueError('arguments require a sequence')
        object.__setattr__(self, 'arguments', tuple(self.arguments))
        object.__setattr__(self, 'environment', Configuration(self.environment))
        self.process_request()

    def process_request(self):
        return ProcessRequest((str(Path(self.directory) / 'run'), *self.arguments), self.cwd,
                              self.deadline_s, self.environment, self.silence_s, self.schema_version)

    def to_dict(self):
        return dict(schema_version=self.schema_version, directory=self.directory,
                    build_identity=self.build_identity, cwd=self.cwd, deadline_s=self.deadline_s,
                    arguments=list(self.arguments), environment=self.environment.to_dict(),
                    silence_s=self.silence_s)

    @property
    def identity(self):
        return hashlib.sha256(Configuration(self.to_dict())._json.encode()).hexdigest()


@dataclass(frozen=True)
class ArtifactRun:
    request: ArtifactRunRequest
    process: ProcessResult
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.request, ArtifactRunRequest) or not isinstance(self.process, ProcessResult):
            raise TypeError('artifact run requires request and process evidence')
        if self.process.request_identity != self.request.process_request().identity:
            raise ValueError('artifact run process identity mismatch')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported artifact run schema')

    def to_dict(self):
        return dict(schema_version=self.schema_version, request=self.request.to_dict(),
                    process=self.process.to_dict())

    @property
    def identity(self):
        return hashlib.sha256(Configuration(self.to_dict())._json.encode()).hexdigest()
