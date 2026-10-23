"""Prepare new producer-bound references. Never certify an old unbound payload in place."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import sys
from etx.models.declarations import declaration as _declaration
from . import paths

from .batch_goldens import FILES
from .batch_goldens import binding
from .batch_goldens import read_binding
from .batch_goldens import request_dir
from .batch_goldens import sha
from .batch_goldens import main as goldens
FLEET_SRC_SHA256 = _declaration(paths.ROOT, "dsv2lite.pinned").FLEET_SRC_SHA256
fleet_export = _declaration(paths.ROOT, "dsv2lite.pinned").fleet_export
source_sha = _declaration(paths.ROOT, "dsv2lite.pinned").source_sha
from etx.models.dsv2lite.session import Session, write_json, PINNED_INPUTS

RECEIPT = "golden.producer.json"


def numerical_runtime() -> dict:
    import importlib.metadata
    import torch
    if not torch.cuda.is_available():
        raise ValueError("reference producer requires the native GPU runtime; not a CPU reference")
    prop = torch.cuda.get_device_properties(0)
    return dict(packages={p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "tokenizers", "numpy", "safetensors")},
                hip=torch.version.hip, device=prop.name, arch=getattr(prop, "gcnArchName", None),
                default_dtype=str(torch.get_default_dtype()),
                matmul_precision=torch.get_float32_matmul_precision())


def recipe(fleet: Path, model: Path, r: int) -> dict:
    """Include the exact local model, including remote-code files. Cache lookup needs no GPU."""
    if source_sha(fleet / "src") != FLEET_SRC_SHA256:
        raise ValueError("reference producer is not the pinned Fleet export")
    files = sorted(p for p in model.rglob("*") if p.is_file() and
                   (p.suffix in (".json", ".safetensors", ".py") or p.name.startswith("tokenizer")))
    if not files or not (model / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("complete local model snapshot required")
    return dict(schema=1, numerical_runtime=numerical_runtime(),
                numerical_recipe=dict(dtype="bfloat16", attention="eager", context=1024, decode=32, seed=0,
                                      deterministic_algorithms="warn_only", eval=True),
                fleet_src_sha256=FLEET_SRC_SHA256, rotation=97*r,
                wrapper_sha256=sha(paths.HERE / "reference_rotated.py"),
                producer_sha256=sha(fleet / "src/host/reference_run.py"),
                model_sha256={str(p.relative_to(model)): sha(p) for p in files})


def verify(d: Path, expected: dict) -> dict:
    current = binding(d)
    receipt = json.loads((d / RECEIPT).read_text())
    if read_binding(d) != current or receipt.get("recipe") != expected or receipt.get("binding") != current:
        raise ValueError(f"producer binding refused: {d}")
    if receipt.get("exit_code") != 0 or not receipt.get("command") or not receipt.get("producer_log_sha256"):
        raise ValueError(f"producer execution receipt missing: {d}")
    if receipt["producer_log_sha256"] != sha(d / "golden.producer.raw.log"):
        raise ValueError(f"producer raw log differs: {d}")
    return receipt


def prepare(s: Session, model: Path, count: int, regenerate=False) -> dict:
    """Write only a fresh result tree; old cache files are never changed or retroactively certified."""
    dest = s.logs / "references"
    if dest.exists():
        raise ValueError("references already exists; use a fresh result directory")
    dest.mkdir()
    result = dict(status="INCOMPLETE", original_payload_recovered=False, requests=[], data=str(dest))
    try:
        fleet = fleet_export(s.fleet)
        base = recipe(fleet, model, 0)  # hash multi-GB model once
        for r in range(count):
            expected = dict(base, rotation=97*r)
            old, new = request_dir(s.data, r), request_dir(dest, r)
            try:
                receipt = verify(old, expected)
            except (OSError, ValueError, KeyError) as exc:
                if not regenerate:
                    raise ValueError(f"request {r}: no reusable producer-bound tuple; regeneration required: {exc}") from exc
                new.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, "examples/dsv2lite/reference_rotated.py", "--fleet", str(fleet),
                       "--rotate", str(97*r), "--", "--model", str(model), "--context", "1024", "--decode", "32", "--out", str(new / "golden.npz")]
                name = f"reference_r{r}"
                s.run(name, cmd, env=dict(HIP_VISIBLE_DEVICES=os.environ.get("GPU", "0")), timeout=180)
                # Only this successful producer call may create the receipt.
                goldens([str(new), "--certify"])
                receipt = dict(recipe=expected, binding=binding(new), command=cmd, exit_code=0,
                               producer_log_sha256=sha(s.logs / f"{name}.raw.log"))
                shutil.copyfile(s.logs / f"{name}.raw.log", new / "golden.producer.raw.log")
                write_json(new / RECEIPT, receipt)
                verify(new, expected)
                mode = "new_inputs"
            else:
                new.mkdir(parents=True, exist_ok=True)
                for f in (*FILES, "golden.binding.json", RECEIPT, "golden.producer.raw.log"):
                    (new / f).symlink_to((old / f).resolve())
                mode = "reused_producer_bound"
            result["requests"].append(dict(request=r, mode=mode, receipt=receipt))
            print(f"REFERENCES request={r} mode={mode}", flush=True)
        # The weight pack is independent of regenerated request state. Keep its measured identity gate.
        result["weights_sha256"] = {}
        for f in ("weights.bin", "weights.bin.manifest"):
            result["weights_sha256"][f] = sha(s.data / f)
            if result["weights_sha256"][f] != PINNED_INPUTS[f]:
                raise ValueError(f"weight input differs from pinned baseline: {f}")
            (dest / f).symlink_to((s.data / f).resolve())
        goldens([str(dest), "--requests", str(count)])
        result["status"] = "READY"
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        write_json(s.logs / "references_ready.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--requests", required=True, type=int, choices=(4, 8))
    ap.add_argument("--regenerate-missing", action="store_true")
    a = ap.parse_args(argv)
    prepare(Session(), a.model.expanduser(), a.requests, a.regenerate_missing)


if __name__ == "__main__":
    main()
