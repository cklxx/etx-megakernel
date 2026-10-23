"""Translator for harness shells: `python -m etx.tools.harness <harness> <module> --root <repo> [args...]`."""
import sys

from etx.external import HarnessCommandRequest, run_harness_command


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 4 or argv[2] != '--root':
        raise SystemExit('usage: python -m etx.tools.harness <harness> <module> --root <repo> [args...]')
    return run_harness_command(HarnessCommandRequest(argv[0], argv[1], tuple(argv[4:]), argv[3]))


if __name__ == '__main__':
    raise SystemExit(main())
