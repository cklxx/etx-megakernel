"""One-command CPU acceptance. Reports external skips separately from passes."""
from __future__ import annotations

import argparse
from pathlib import Path

from etx.external import cpu_acceptance_fixtures, run_cpu_acceptance


def run(out, **options):
    """Compatibility entry point; lifecycle is owned by the experiment scaffold."""
    return run_cpu_acceptance(out, **options)


def fixtures(out):
    """Compatibility entry point for the fixture worker."""
    return cpu_acceptance_fixtures(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--require-external", action="store_true")
    parser.add_argument("--local-matrix", action="store_true", help="run the 16-entry production-operator compilation matrix")
    compiler = parser.add_mutually_exclusive_group()
    compiler.add_argument("--rocm-image", help="locally installed ROCm image for offline compile/link; no GPU execution")
    compiler.add_argument("--rocm-root", type=Path, help="installed or extracted ROCm root (contains bin/hipcc)")
    parser.add_argument("--fixtures", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.fixtures:
        return fixtures(args.fixtures)
    if args.out is None:
        parser.error("--out is required")
    try:
        return run(args.out, formal=args.formal, require_external=args.require_external, rocm_image=args.rocm_image,
                   rocm_root=args.rocm_root, local_matrix=args.local_matrix)
    except (OSError, ValueError) as exc:
        print(f"CPU CHECK FAILED: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
