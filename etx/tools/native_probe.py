"""Translator for bench/calib/run_calib.sh: `python -m etx.tools.native_probe calibration [arch]`."""
import os
from pathlib import Path
import sys

from etx.external import NativeProbeRequest, run_native_probe


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ['calibration']:
        raise SystemExit('usage: python -m etx.tools.native_probe calibration [arch]')
    root = str(Path(__file__).resolve().parents[2])
    # Extra arguments were ignored by the historical shell; still ignored.
    arguments = dict(arch=argv[1] if len(argv) > 1 else 'gfx942', compiler=os.environ.get('ETX_MODEL_COMPILER', 'hipcc'))
    return run_native_probe(NativeProbeRequest('calibration', 'run', root, arguments))


if __name__ == '__main__':
    raise SystemExit(main())
