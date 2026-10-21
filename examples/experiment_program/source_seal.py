"""Seal the local implementation needed by a relocated prepared session."""
from pathlib import Path
from .protocol import sha
ROOT=Path(__file__).resolve().parents[2]


def snapshot():
    paths=set()
    for folder in ('etx','examples/experiment_program','examples/dsv2lite','examples/vllm_llm','examples/llm'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.hip','.h','.hpp','.cpp','.cuh','.sh','.json','.yaml','.yml') and not any(x.startswith(('results','build','__pycache__')) for x in p.relative_to(ROOT).parts):paths.add(p)
    paths.add(ROOT/'examples/sessions/experiment_program.py')
    paths.update((ROOT/'examples/llm/configs').glob('*.json'))
    return {str(p.resolve()):sha(p) for p in sorted(paths)}
