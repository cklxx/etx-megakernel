"""Seal the source scope used by staged experiment providers."""
from pathlib import Path

from etx.experiments.sealed_protocol import sha

ROOT = Path(__file__).resolve().parents[2]


def snapshot():
    paths = set()
    for folder in ('etx', 'examples/experiment_program', 'examples/dsv2lite',
                   'examples/vllm_llm', 'examples/llm'):
        for path in (ROOT / folder).rglob('*'):
            relative = path.relative_to(ROOT)
            if not path.is_file() or path.suffix not in ('.py', '.hip', '.h', '.hpp', '.cpp', '.cuh', '.sh', '.json', '.yaml', '.yml'):
                continue
            # etx/build is library source, unlike generated build directories.
            if any(part.startswith(('results', '__pycache__')) or
                   (part.startswith('build') and not (relative.parts[0] == 'etx' and i == 1 and part == 'build'))
                   for i, part in enumerate(relative.parts[:-1])):
                continue
            paths.add(path)
    paths.add(ROOT / 'examples/sessions/experiment_program.py')
    paths.update((ROOT / 'examples/llm/configs').glob('*.json'))
    return {str(path.resolve()): sha(path) for path in sorted(paths)}
