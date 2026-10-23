"""Fail-closed implementation of the prepared ds_inplace.py GPU session: every lever measured as whole tokens.

Every arm -- its levers, flags, plan switches, invocation order, references and pre-registered rule -- is ONE record in
examples/dsv2lite/arms.py; the decisions are etx/session/experiment.py's, on integer nanoseconds.
  build <arm> [--stamped]   the arm's build (fleet: fleet's own)
  pairs <arm>     5 interleaved rounds in the arm's order (reversed on odd rounds), every invocation 32/32 + 27/27
  trace <arm>     one stamped token of the arm's STAMPS2 build: the measured per-stage table
  layers <arm> --reference-dir <dir>   one full single-request invocation against that request's own reference,
                                      with ETX_DS_LAYER_METRICS=1; the same correctness gates still apply
  report          every pairs report found + measured per-stage savings next to stage_table.json's expected ones
No invocation here provisions a GPU or downloads; inputs are prepared by the prep_* jobs. Unfiltered output is kept.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from etx.external import ExperimentSessionRequest, experiment_session_arms, run_experiment_session

# Implementation: etx/models/dsv2lite/session.py behind etx.external.run_experiment_session.


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("operation", choices=("source", "preflight", "build", "pairs", "trace", "layers", "d4trace", "report"))
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--stamped", action="store_true")
    ap.add_argument("--reference-dir", type=Path, help="layers only: request's own fleet_cache.bin and goldens")
    a = ap.parse_args(argv)
    if a.reference_dir is not None and a.operation != "layers":
        ap.error("--reference-dir is only valid for layers")
    if a.operation == "layers" and a.reference_dir is None:
        ap.error("layers requires --reference-dir")
    root = str(Path(__file__).resolve().parents[2])
    allowed = experiment_session_arms("dsv2lite", root)
    if a.operation in allowed:
        if a.arm not in allowed[a.operation]:
            ap.error(f"{a.operation} arm must be one of {allowed[a.operation]}")
    # The historical environment names the session record and its inputs; the request carries them explicitly.
    run_experiment_session(ExperimentSessionRequest(
        "dsv2lite", a.operation, os.path.abspath(os.environ["ETX_SESS_LOGDIR"]), root,
        dict(fleet=os.path.abspath(os.environ["ETX_DSV2_FLEET"]), data=os.path.abspath(os.environ["ETX_DSV2_DATA"])),
        arm=a.arm, stamped=a.stamped, reference_dir=str(a.reference_dir.resolve()) if a.reference_dir else None))


if __name__ == "__main__":
    main()
