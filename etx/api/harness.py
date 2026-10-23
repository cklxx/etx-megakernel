"""Multi-command experiment harnesses: one historical command of a registered harness, from an explicit repository."""
from dataclasses import dataclass
from pathlib import Path

from .types import SCHEMA_VERSION


# Harness -> the command modules it exposes (each was `python -m examples.<model>.<harness>.<module>`).
HARNESSES = {'qwen-expert-merge': ('runner', 'prepare', 'capture', 'inputs', 'cpu', 'validate', 'protocol'),
             'qwen-attention-early-merge': ('session', 'build', 'build_model', 'offline', 'simulate'),
             'qwen-bf16-probe': ('launch_build', 'build', 'protocol', 'inbody_cpu', 'launch', 'mutations', 'pack'),
             'glm53-seam': ('seam',),
             'glm53-boundary': ('boundary_build',),
             'glm53-model-build': ('model_build',),
             'glm53-run-all': ('run_all',),
             'qwen-vllm': ('vllm_build', 'decode')}


@dataclass(frozen=True)
class HarnessCommandRequest:
    harness: str
    module: str
    argv: tuple[str, ...]          # the command's arguments, parsed by the harness exactly as before
    root: str                      # repository whose declarations and native assets the harness uses
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.module not in HARNESSES.get(self.harness, ()):
            raise ValueError(f'unknown harness command {self.harness!r}/{self.module!r}')
        if isinstance(self.argv, str) or any(type(arg) is not str for arg in self.argv):
            raise ValueError('argv requires a sequence of strings')
        object.__setattr__(self, 'argv', tuple(self.argv))
        if type(self.root) is not str or not Path(self.root).is_absolute():
            raise ValueError('root must be an absolute path')
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported harness command schema')
