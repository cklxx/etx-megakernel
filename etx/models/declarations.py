"""Import a model's declaration modules (examples/<model>/...) from an explicit repository root.

Model packages in etx/ never import examples ambiently: the caller names the repository tree whose
declarations apply (a staged session archive, a checkout). The tree is put first on sys.path for the
import only; a different root already imported is refused rather than mixed.
"""
import importlib
from pathlib import Path
import sys
import threading


_LOCK = threading.Lock()


def declaration(root, name):
    """Return module `examples.<name>` from repository `root` (absolute)."""
    root = Path(root)
    if not root.is_absolute() or not (root / 'examples').is_dir():
        raise ValueError(f'repository root with examples/ required, got {root}')
    full = 'examples.' + name
    with _LOCK:
        loaded = sys.modules.get('examples')
        if loaded is not None:
            origin = Path(next(iter(getattr(loaded, '__path__', [''])))).resolve()
            if origin != (root / 'examples').resolve():
                raise ValueError(f'declarations already imported from {origin.parent}, not {root}')
        sys.path.insert(0, str(root))
        try:
            return importlib.import_module(full)
        finally:
            sys.path.remove(str(root))
