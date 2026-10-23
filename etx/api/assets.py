"""Model asset preparation: pack weights and write references for one model, from explicit inputs."""
from dataclasses import dataclass, field
from pathlib import Path

from .types import Configuration, SCHEMA_VERSION


PROVIDERS = {'llm': ('model', 'out', 'gen', 'prompt', 'src', 'context', 'sliced', 'rotate', 'golden_only', 'prefill_only')}


@dataclass(frozen=True)
class AssetPreparationRequest:
    provider: str
    root: str                      # repository whose model declarations define the packed layout
    arguments: Configuration = field(default_factory=lambda: Configuration({}))
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        fields = PROVIDERS.get(self.provider)
        if fields is None:
            raise ValueError(f'unknown asset provider {self.provider!r}')
        if type(self.root) is not str or not Path(self.root).is_absolute():
            raise ValueError('root must be an absolute path')
        object.__setattr__(self, 'arguments', Configuration(self.arguments))
        if set(self.arguments) != set(fields):
            raise ValueError(f'{self.provider} asset preparation takes exactly {list(fields)}')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported asset preparation schema')
