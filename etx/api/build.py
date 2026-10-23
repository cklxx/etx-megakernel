"""Offline compilation receipts, deliberately distinct from linked executables."""
from dataclasses import dataclass
import hashlib

from .artifacts import payload_path
from .types import Configuration, SCHEMA_VERSION
from .execution import ProcessResult


@dataclass(frozen=True)
class DeviceCheck:
    source_identity: str
    toolchain: Configuration
    ranks: tuple[ProcessResult, ...]
    objects: Configuration
    status: str
    tiles: tuple[ProcessResult, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'toolchain', Configuration(self.toolchain))
        object.__setattr__(self, 'objects', Configuration(self.objects))
        object.__setattr__(self, 'ranks', tuple(self.ranks))
        object.__setattr__(self, 'tiles', tuple(self.tiles))
        invocations = self.ranks + self.tiles
        if not self.ranks or any(not isinstance(result, ProcessResult) or result.backend != 'host-process'
                                 for result in invocations):
            raise ValueError('device checks require actual host compiler invocations')
        complete = all(result.status == 'complete' for result in invocations)
        checks = [(f'megakernel_d{rank}.o', result) for rank, result in enumerate(self.ranks)]
        checks += [(f'native_tile_{index}.o', result) for index, result in enumerate(self.tiles)]
        expected = {name for name, result in checks if result.status == 'complete'}
        if complete != (self.status == 'DEVICE_CHECKED') or self.status not in ('DEVICE_CHECKED', 'FAILED'):
            raise ValueError('device-check status disagrees with compiler results')
        if set(self.objects) != expected:
            raise ValueError('device check requires objects for every successful rank and tile')
        for digest in [self.source_identity, *self.objects.values()]:
            if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid device-check digest')

    def to_dict(self):
        result = dict(source_identity=self.source_identity, toolchain=self.toolchain.to_dict(),
                      ranks=[rank.to_dict() for rank in self.ranks], objects=self.objects.to_dict(),
                      status=self.status)
        if self.tiles:
            result['tiles'] = [tile.to_dict() for tile in self.tiles]
        return result


@dataclass(frozen=True)
class BuildArtifact:
    """Linked host binary receipt; EXECUTABLE does not mean device-qualified."""
    device_check: DeviceCheck
    link: ProcessResult | None
    files: Configuration
    status: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.device_check, DeviceCheck):
            raise TypeError('build requires a device-check receipt')
        object.__setattr__(self, 'files', Configuration(self.files))
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported build schema')
        if self.link is not None and (not isinstance(self.link, ProcessResult) or
                                      self.link.backend != 'host-process'):
            raise ValueError('build requires an actual host linker invocation')
        if self.device_check.status != 'DEVICE_CHECKED' and self.link is not None:
            raise ValueError('linking cannot follow failed device checks')
        if self.device_check.status == 'DEVICE_CHECKED' and self.link is None:
            raise ValueError('completed device checks require a linker outcome')
        complete = self.link is not None and self.link.status == 'complete'
        if self.status not in ('EXECUTABLE', 'FAILED') or complete != (self.status == 'EXECUTABLE'):
            raise ValueError('build status disagrees with linker result')
        if set(self.files) != ({'run'} if complete else set()):
            raise ValueError('build requires exactly its linked executable')
        for name, digest in self.files.items():
            payload_path(name)
            if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid build digest')

    def to_dict(self):
        return dict(schema_version=self.schema_version, device_check=self.device_check.to_dict(),
                    link=self.link.to_dict() if self.link is not None else None,
                    files=self.files.to_dict(), status=self.status)

    @property
    def identity(self):
        return hashlib.sha256(Configuration(self.to_dict())._json.encode()).hexdigest()
