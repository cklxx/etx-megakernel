"""CPU-only compile/link of an `etx compile` directory; translator for etx.external.check_codegen.

python -m etx.tools.device_check <codegen-dir> <rocm-directory-or-image> [--logs DIR]
"""
import argparse

from etx.external import check_codegen


def main(argv=None):
    parser = argparse.ArgumentParser(prog='etx-device-check', description=__doc__)
    parser.add_argument('codegen')
    parser.add_argument('toolchain')
    parser.add_argument('--logs')
    args = parser.parse_args(argv)
    return check_codegen(args.codegen, toolchain=args.toolchain, logs=args.logs)['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
