"""Explicit legacy compiler invocation; no argparse/compiler objects cross the API."""
from dataclasses import dataclass, field

from .types import Configuration


@dataclass(frozen=True)
class CompilerRequest:
    command: str
    graph: str | None = None
    target: str = 'gfx942'
    output: str = 'build/out'
    bindings: Configuration = field(default_factory=lambda: Configuration({}))
    options: Configuration = field(default_factory=lambda: Configuration({}))
    simulate: bool = False
    seed: int = 0

    def __post_init__(self):
        if self.command not in ('compile', 'explain', 'archs'):
            raise ValueError('unknown compiler command')
        if self.command != 'archs' and (type(self.graph) is not str or not self.graph):
            raise ValueError('compiler graph file is required')
        if type(self.target) is not str or not self.target or type(self.output) is not str or not self.output:
            raise ValueError('compiler target/output must be nonempty strings')
        if type(self.simulate) is not bool or type(self.seed) is not int:
            raise ValueError('invalid compiler simulation controls')
        object.__setattr__(self, 'bindings', Configuration(self.bindings))
        object.__setattr__(self, 'options', Configuration(self.options))
        if any(type(value) is not int for value in self.bindings.values()):
            raise ValueError('compiler binding overrides require integers')
