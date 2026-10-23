"""Translator for tools/session.sh: `python -m etx.tools.session --root <repo> <action> <ip> [args...]`."""
import sys

from etx.external import SessionTransportRequest, run_session_transport


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ['--root'] or len(argv) < 2:
        raise SystemExit('usage: python -m etx.tools.session --root <repo> <action> <ip> [args...]')
    return run_session_transport(SessionTransportRequest(tuple(argv[2:]), argv[1]))


if __name__ == '__main__':
    raise SystemExit(main())
