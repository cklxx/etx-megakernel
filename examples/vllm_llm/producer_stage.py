"""Attach a supplied, independent venv to a fresh archive. No install or download."""
from pathlib import Path
import argparse
import subprocess

PROBE = ('import sys,yaml,numpy; from pathlib import Path; '
         'assert sys.version_info >= (3,11), "Python >=3.11 required"; '
         'assert sys.prefix != sys.base_prefix, "not a venv"; '
         'assert all(Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()) '
         'for m in (yaml,numpy)), "dependencies outside venv"; '
         'print("STAGE interpreter/dependencies OK", sys.prefix)')


def attach(root, venv):
    root, venv = Path(root).resolve(), Path(venv).expanduser().resolve(strict=True)
    if venv.is_relative_to(root):
        raise ValueError('use an independent venv outside the staged checkout')
    # Probe before creating the link, then probe through the actual staged path.
    subprocess.run([str(venv/'bin/python'), '-I', '-B', '-c', PROBE], check=True, timeout=20)
    link = root/'.venv'
    if link.is_symlink() and link.resolve() == venv:
        pass
    elif link.exists() or link.is_symlink():
        raise ValueError('refusing to replace existing .venv')
    else:
        link.symlink_to(venv, target_is_directory=True)
    subprocess.run([str(link/'bin/python'), '-I', '-B', '-c', PROBE], check=True, timeout=20)
    print(f'STAGE ready root={root} venv={venv}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('venv', type=Path)
    args = ap.parse_args()
    attach(Path(__file__).resolve().parents[2], args.venv)


if __name__ == '__main__':
    main()
