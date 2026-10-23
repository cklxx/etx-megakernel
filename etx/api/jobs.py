"""Immutable scheduler declarations; a successful command is not device qualification."""
from dataclasses import dataclass, field
import math
import re

from .types import Configuration


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def _strings(values, name):
    values = (values,) if isinstance(values, str) else tuple(values)
    if any(type(v) is not str or not v or '\0' in v for v in values):
        raise ValueError(f'{name} requires nonempty strings')
    return values


@dataclass(frozen=True)
class JobSpec:
    name: str
    cmd: str
    lane: str = 'gpu'
    gpu: int | None = None
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    timeout: float = 900.0
    env: Configuration = field(default_factory=lambda: Configuration({}))
    machine: str = 'A'
    summary: tuple[str, ...] = ()
    quiet_s: float = 180.0
    cwd: str = '~/etx'
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported job schema')
        if type(self.name) is not str or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', self.name):
            raise ValueError('job name must be a safe log basename')
        for name in ('cmd', 'machine', 'cwd'):
            if type(getattr(self, name)) is not str or not getattr(self, name) or '\0' in getattr(self, name):
                raise ValueError(f'invalid job {name}')
        if self.lane not in ('gpu', 'cpu', 'final'):
            raise ValueError('job lane must be gpu, cpu or final')
        if self.gpu is not None and (type(self.gpu) is not int or self.gpu < 0):
            raise ValueError('invalid GPU pin')
        _positive(self.timeout, 'timeout')
        _positive(self.quiet_s, 'quiet_s')
        for name in ('reads', 'writes', 'summary'):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if any(v == '?' for v in (*self.reads, *self.writes)):
            raise ValueError('empty optional dependency/output')
        for pattern in self.summary:
            re.compile(pattern)
        object.__setattr__(self, 'env', Configuration(self.env))
        if any(not k or '=' in k or '\0' in k or type(v) is not str or '\0' in v
               for k, v in self.env.items()):
            raise ValueError('job environment requires valid string keys and values')

    def to_dict(self):
        return {name: (value.to_dict() if isinstance(value, Configuration) else
                       list(value) if isinstance(value, tuple) else value)
                for name in self.__dataclass_fields__
                for value in (getattr(self, name),)}


@dataclass(frozen=True)
class JobSession:
    jobs: tuple[JobSpec, ...]
    max_seconds: float | None = None
    local_only: tuple[str, ...] | None = None
    lease_path: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported session schema')
        object.__setattr__(self, 'jobs', tuple(self.jobs))
        if any(not isinstance(job, JobSpec) for job in self.jobs):
            raise TypeError('session jobs require JobSpec')
        if self.max_seconds is not None:
            _positive(self.max_seconds, 'max_seconds')
        if self.local_only is not None:
            object.__setattr__(self, 'local_only', _strings(self.local_only, 'local_only'))
        if self.lease_path is not None and (type(self.lease_path) is not str or not self.lease_path):
            raise ValueError('invalid lease path')


@dataclass(frozen=True)
class JobSessionRequest:
    session: JobSession | str
    logdir: str = '~/vlnext'
    gpus: tuple[int, ...] = (0,)
    cpu_slots: int = 1
    machine: str = 'A'
    only: str | None = None
    progress_s: float = 60.0
    poll_s: float = 1.0
    dry_run: bool = False
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported scheduler request schema')
        if not isinstance(self.session, JobSession) and (type(self.session) is not str or not self.session):
            raise TypeError('session requires JobSession or an explicit Python session path')
        for name in ('logdir', 'machine'):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f'invalid {name}')
        object.__setattr__(self, 'gpus', tuple(self.gpus))
        if any(type(gpu) is not int or gpu < 0 for gpu in self.gpus) or len(set(self.gpus)) != len(self.gpus):
            raise ValueError('GPU lanes require unique nonnegative integers')
        if type(self.cpu_slots) is not int or self.cpu_slots < 1:
            raise ValueError('cpu_slots must be positive')
        _positive(self.progress_s, 'progress_s')
        _positive(self.poll_s, 'poll_s')
        if type(self.dry_run) is not bool:
            raise ValueError('dry_run must be boolean')
        if self.only is not None:
            re.compile(self.only)


@dataclass(frozen=True)
class JobOutcome:
    name: str
    status: str
    rc: int | None
    start: float
    end: float
    on_gpu: int | None
    log: str | None
    note: str
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported job outcome schema')
        if type(self.name) is not str or not self.name or type(self.note) is not str:
            raise ValueError('invalid job outcome name/note')
        if self.status not in ('pending', 'running', 'ok', 'fail', 'timeout', 'skipped'):
            raise ValueError('invalid job outcome status')
        if self.rc is not None and type(self.rc) is not int:
            raise ValueError('invalid job return code')
        if self.on_gpu is not None and (type(self.on_gpu) is not int or self.on_gpu < 0):
            raise ValueError('invalid job outcome lane')
        if self.log is not None and type(self.log) is not str:
            raise ValueError('invalid job log path')
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in (self.start, self.end)):
            raise ValueError('invalid job timestamps')


@dataclass(frozen=True)
class JobSessionResult:
    returncode: int
    record_directory: str | None
    jobs: tuple[JobOutcome, ...]
    max_seconds: float | None
    backend: str
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported scheduler result schema')
        if type(self.returncode) is not int:
            raise ValueError('invalid scheduler return code')
        if self.record_directory is not None and type(self.record_directory) is not str:
            raise ValueError('invalid record directory')
        if self.max_seconds is not None:
            _positive(self.max_seconds, 'max_seconds')
        object.__setattr__(self, 'jobs', tuple(self.jobs))
        if any(not isinstance(job, JobOutcome) for job in self.jobs):
            raise TypeError('session result requires immutable JobOutcome records')
        if self.backend not in ('host-process', 'not-executed'):
            raise ValueError('unsupported scheduler backend')
