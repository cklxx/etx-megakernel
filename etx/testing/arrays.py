"""Array-file fixtures for external tests, which cannot import numpy themselves."""
from pathlib import Path


def write_npz(path, **arrays):
    """Write `arrays` (name -> sequence) to an uncompressed .npz at `path`, as numpy.savez does."""
    import numpy as np
    np.savez(Path(path), **{name: np.asarray(values) for name, values in arrays.items()})
    return Path(path)
