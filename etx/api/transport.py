"""Session transport requests: stage/push/preflight/start/status/fetch/postmortem against a GPU box."""
from dataclasses import dataclass
from pathlib import Path

from .types import SCHEMA_VERSION


@dataclass(frozen=True)
class SessionTransportRequest:
    # The historical command line (action, host, action arguments); the transport parses it.
    argv: tuple[str, ...]
    # Repository that `stage` archives and whose record tools verify fetched records.
    root: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if isinstance(self.argv, str):
            raise ValueError('argv requires a sequence of strings')
        object.__setattr__(self, 'argv', tuple(self.argv))
        if any(type(arg) is not str or '\0' in arg for arg in self.argv):
            raise ValueError('argv requires strings without NUL')
        if type(self.root) is not str or '\0' in self.root or not Path(self.root).is_absolute():
            raise ValueError('root must be an absolute path')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported transport schema')

    def to_dict(self):
        return dict(schema_version=self.schema_version, argv=list(self.argv), root=self.root)
