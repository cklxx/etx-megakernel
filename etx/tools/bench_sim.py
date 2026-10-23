"""Reproducible CPU protocol benchmarks. No tensor execution or GPU timings."""
import argparse
import json
from pathlib import Path

from etx.external import run_protocol_experiment, report_protocol_experiment


def run(config, out):
    return run_protocol_experiment(config, output=out)


def report(out):
    return report_protocol_experiment(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    try:
        return run(json.loads(args.config.read_text()), args.out) if args.command == "run" else report(args.out)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"BENCH FAILED: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
