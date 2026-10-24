"""One-command CPU acceptance. Reports external skips separately from passes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from etx.tools.validation.bench_sim import ROOT, provenance, write_json, digest
from etx.session import proc

# Known optional tests need artifacts/toolchains beyond the source-only CPU suite.
EXTERNAL_TESTS = {
    # CPU-only CUDA lowering compile requires clang's nvptx64 backend.
    "tests.test_prims.test_sm_90_lowerings_compile",
    # Import toolchain: ETX_LLVM, ETX_VLLM and ETX_ROCM_HEADERS.
    "tests.test_importer.test_wvsplitk_import_links_and_inlines",
    "tests.test_importer.test_triton_downgrade_hex_literals_parse_back",
    "tests.test_codegen.test_generated_hip_compiles",
    "tests.test_ref_numpy.test_ref_numpy_matches_hf",
    "tests.test_coh_model.test_coh_range",
    "tests.test_coh_model.test_coh_off_is_identity",
    "tests.test_coh_model.test_coh_not_applied_unfused",
    "tests.test_phase_model.test_reproduces_the_measured_builds",
    "tests.test_phase_model.test_folding_narrow_phases_away_is_worth_almost_nothing",
    # Imported builds: ETX_CONTRACT_BUILDS and ETX_LLVM.
    "tests.test_acquire_contract.test_real_imported_build[qwen3-30b-a3b_xcd_qknr-48]",
    "tests.test_acquire_contract.test_real_imported_build[mixtral-8x7b_imported-32]",
    # Round48 archive: ETX_DATA_ROOT.
    "tests.test_pairs.test_round48_interleave_log_is_refused",
    # Round11 archive gates: skipped without ETX_DATA_ROOT, failures when ETX_REQUIRE_DATA=1.
    "tests.test_layer_budget.test_round11_host_correspondence",
    "tests.test_bandwidth_phases.test_round11_spec_and_ranked_total",
    "tests.test_floors.test_round11_table_matches_the_raw_trace",
    # Hop evidence: the round11 commit is absent from a shallow clone; the vLLM lines need VLLM_SRC.
    "tests.test_floors.test_tile_evidence_lines_at_the_round11_commit",
    "tests.test_floors.test_vllm_evidence_lines",
    # ISA recount needs ETX_ISA_S and ETX_ISA_MAP.
    "tests.test_floors.test_isa_hop_table_matches_the_matching_build",
    # Host SIMT harness needs a C++20 compiler with <barrier>.
    "tests.test_topk_key32.test_cpp_block_on_a_host_simt_wave",
    # G0 source baseline needs an external vLLM checkout via ETX_VLLM_SOURCE_ROOT.
    "tests.test_vlbest_session.test_g0_local_source_baseline",
}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


ARCHIVE_INDEX_SHA256 = "d4031ea630a8df134c57c43f269a2f090ef0396f6167dec82b4d0585ee2b5f71"


def verify_archive(directory, expected_index=ARCHIVE_INDEX_SHA256):
    """Verify the pinned inventory and every member before using calibration data."""
    index = directory / "SHA256SUMS"
    if file_hash(index) != expected_index:
        raise ValueError(f"archive inventory checksum mismatch: {index}")
    verified = {str(index): expected_index}
    members = set()
    root = directory.resolve()
    for line in index.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if not match:
            raise ValueError(f"invalid archive checksum entry: {line!r}")
        expected, name = match.groups()
        path = (directory / name).resolve()
        if not path.is_relative_to(root) or path in members or path == index.resolve():
            raise ValueError(f"unsafe or duplicate archive entry: {name}")
        members.add(path)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"archive member checksum mismatch or missing file: {name}")
        verified[str(path)] = expected
    if not members:
        raise ValueError("empty archive inventory")
    # An unlisted file could still be loaded by a regression test. Resolving the enumeration to its
    # targets hides an unlisted symlink alias, and rglob does not descend into directory symlinks at
    # all, so no symlink is allowed in the archive instead.
    entries = list(directory.rglob("*"))
    links = [p for p in entries if p.is_symlink()]
    if links:
        raise ValueError(f"archive contains a symlink: {links[0]}")
    present = {p.resolve() for p in entries if p.is_file() and p != index}
    if present != members:
        raise ValueError("archive contains unlisted files")
    return verified


def dependencies():
    result = {"packages": {}, "artifacts": {}, "checkouts": {}}
    for name in ("numpy", "pytest", "pyyaml", "torch", "transformers", "safetensors"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    model = Path(os.environ.get("ETX_REFERENCE_MODEL", "~/models/qwen2.5-0.5b")).expanduser()
    for path in sorted(model.glob("*")):
        if path.suffix in (".json", ".safetensors", ".txt") and path.is_file():
            result["artifacts"][str(path)] = file_hash(path)
    archived = ROOT / "results_2026-09-30"
    if archived.exists():
        result["artifacts"].update(verify_archive(archived))
    for key in ("ETX_VLLM",):
        if os.environ.get(key):
            path = Path(os.environ[key]).expanduser()
            commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
            result["checkouts"][key] = {"path": str(path), "commit": commit.stdout.strip() if commit.returncode == 0 else None}
    return result


def test_results(path):
    root = ET.parse(path).getroot()
    result = {"passed": 0, "failed": [], "skipped": [], "unexpected_skips": []}
    for case in root.iter("testcase"):
        name = case.attrib.get("classname", "") + "." + case.attrib["name"]
        skipped = case.find("skipped")
        if skipped is not None:
            item = {"test": name, "reason": skipped.attrib.get("message", "")}
            result["skipped"].append(item)
            if name not in EXTERNAL_TESTS:
                result["unexpected_skips"].append(item)
        elif case.find("failure") is not None or case.find("error") is not None:
            result["failed"].append(name)
        else:
            result["passed"] += 1
    return result


def command_step(name, command, out, *, timeout=300, cwd=ROOT, env=None):
    log = out / f"{name}.log"
    print(f"START {name}: {' '.join(map(str, command))}", flush=True)
    started = time.monotonic()
    with log.open("w") as stream:
        try:
            code = proc.run(command, sink=stream, deadline_s=timeout, poll_s=10, cwd=cwd, env=env, log=stream,
                            on_poll=lambda s: print(f"RUNNING {name}: {s:.0f}s; log={log}", flush=True))
            if code == proc.DEADLINE_RC:
                stream.write(f"\nTIMEOUT after {timeout}s\n")
        except OSError as exc:
            stream.write(f"{type(exc).__name__}: {exc}\n")
            code = 127
    print(f"END {name}: exit={code}", flush=True)
    return {"step": name, "returncode": code, "log": log.name, "seconds": time.monotonic() - started}


def pytest_environment():
    env = os.environ.copy()
    # Acceptance always executes the repository suite with built-in pytest plus
    # our inventory plugin. Ambient selection/plugin settings are not a protocol.
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def run(out, *, formal=False, require_external=False, rocm_image=None, rocm_root=None, local_matrix=False):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"kind": "cpu_acceptance", "schema_version": 1, "provenance": provenance(),
                "gpu_executed": False, "cpu_core_passed": False, "all_external_checks_passed": False,
                "status": "incomplete", "steps": [], "external_pending": [], "artifacts": {}}
    write_json(out / "acceptance.json", manifest)
    try:
        manifest["dependencies"] = dependencies()
    except (OSError, ValueError) as exc:
        manifest.update(status="failed", dependency_error=str(exc))
        write_json(out / "acceptance.json", manifest)
        (out / "README.md").write_text(f"# CPU acceptance\n\nStatus: **failed**.\n\n{exc}\n")
        print(f"CPU CHECK FAILED: {exc}", flush=True)
        return 1
    steps = manifest["steps"]
    # Run these in a child so assertion/import/shape failures retain a log and
    # cannot prevent the remaining independent checks from running.
    steps.append(command_step("numerical-and-specs", [sys.executable, "-m", "etx.tools.cpu_check", "--fixtures", out], out))
    scripts = sorted(ROOT.rglob("*.sh"))
    scripts = [p for p in scripts if not any(x in p.relative_to(ROOT).parts for x in (".venv", ".git", "build"))]
    for script in scripts:
        # bash -n parses only: never provision a GPU or execute a session script.
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        with (out / "shell-syntax.log").open("a") as stream:
            stream.write(f"{script.relative_to(ROOT)} exit={result.returncode}\n{result.stdout}{result.stderr}")
        if result.returncode:
            steps.append({"step": f"shell:{script.relative_to(ROOT)}", "returncode": result.returncode})
    steps.append({"step": "shell-syntax", "returncode": 0 if not any(s["step"].startswith("shell:") for s in steps) else 1,
                  "scripts": len(scripts), "log": "shell-syntax.log"})
    for example in ("splitk_sum", "moe_layer", "gemm_rs"):
        steps.append(command_step(f"codegen-{example}", [sys.executable, "-m", "etx", "compile",
            ROOT / f"examples/{example}.py", "--arch", "gfx942", "--out", out / "codegen" / example], out))
    steps.append(command_step("pytest", [sys.executable, "-m", "pytest", "-q", "-ra", "-o", "addopts=",
        "-p", "etx.tools.pytest_acceptance", f"--acceptance-inventory={out / 'test-inventory.json'}",
        f"--junitxml={out / 'pytest.xml'}", ROOT / "tests"], out, timeout=1800, env=pytest_environment()))
    inventory = out / "test-inventory.json"
    complete = inventory.exists() and json.loads(inventory.read_text()).get("complete") is True
    steps.append({"step": "test-completeness", "returncode": int(not complete)})
    if (out / "pytest.xml").exists():
        manifest["tests"] = test_results(out / "pytest.xml")
        manifest["external_pending"].extend(manifest["tests"]["skipped"])
    else:
        manifest["tests"] = {"passed": 0, "failed": ["missing pytest results"], "unexpected_skips": [], "skipped": []}
    steps.append(command_step("protocol", [sys.executable, "-m", "etx.tools.bench_sim", "run", "--config",
                                          ROOT / "bench/configs/protocol-gfx942-v1.json", "--out", out / "protocol"], out))
    if (out / "protocol/summary.json").exists():
        before = (out / "protocol/summary.json").read_bytes()
        steps.append(command_step("recompute", [sys.executable, "-m", "etx.tools.bench_sim", "report", out / "protocol"], out))
        steps.append({"step": "recompute-identical", "returncode": int(before != (out / "protocol/summary.json").read_bytes())})
    if formal:
        steps.append(command_step("formal", ["bash", ROOT / "formal/check.sh", "old", out / "formal"], out, timeout=600))
    else:
        manifest["external_pending"].append({"test": "formal", "reason": "not requested; use --formal with LEAN and TLC_JAR"})
    if rocm_image or rocm_root:
        steps.append(command_step("device-compile", ["bash", ROOT / "tools/cpu_device_check.sh", out / "codegen", rocm_image or rocm_root.resolve()],
                                  out, timeout=600))
    else:
        manifest["external_pending"].append({"test": "device-compile", "reason": "use --rocm-root or --rocm-image with an installed toolchain"})
    if local_matrix:
        steps.append(command_step("local-matrix", ["env", "-u", "LOCAL_MATRIX", f"ETX_PYTHON={sys.executable}",
                                  "bash", ROOT / "examples/vllm_llm/local_matrix.sh", out / "local-matrix"],
                                  out, timeout=1200))
    else:
        manifest["external_pending"].append({"test": "local-matrix", "reason": "use --local-matrix with matching LLVM, ROCm headers/device libraries and vLLM sources"})
    tests = manifest["tests"]
    manifest["cpu_core_passed"] = all(s["returncode"] == 0 for s in steps) and not tests["unexpected_skips"] and not tests["failed"]
    manifest["all_external_checks_passed"] = manifest["cpu_core_passed"] and not manifest["external_pending"]
    manifest["status"] = "passed" if manifest["cpu_core_passed"] else "failed"
    if require_external and not manifest["all_external_checks_passed"]:
        manifest["status"] = "failed"
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("acceptance.json", "README.md"):
            manifest["artifacts"][str(path.relative_to(out))] = digest(path.read_bytes())
    write_json(out / "acceptance.json", manifest)
    lines = ["# CPU acceptance", "", f'Status: **{manifest["status"]}**.',
             f'CPU core: {manifest["cpu_core_passed"]}; all external checks: {manifest["all_external_checks_passed"]}.',
             "GPU execution and performance validation: not performed.", "",
             f'Tests: {tests["passed"]} passed, {len(tests["failed"])} failed, {len(tests["skipped"])} skipped.', "",
             "| Step | Exit |", "|---|---:|"]
    lines += [f'| {s["step"]} | {s["returncode"]} |' for s in steps]
    lines += ["", "## External checks still pending", ""]
    lines += [f'- {p["test"]}: {p["reason"]}' for p in manifest["external_pending"]] or ["None."]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return int(manifest["status"] != "passed")


def fixtures(out):
    from etx.bench.numerical import acceptance_suite
    from etx.importer.launch_spec import model_specs
    result = acceptance_suite()
    write_json(out / "numerical.json", result)
    specs = {name: model_specs(ROOT / f"examples/llm/configs/{name}.json")
             for name in ("qwen3-30b-a3b", "mixtral-8x7b")}
    write_json(out / "operator-specs.json", specs)
    from etx.importer.launch_spec import registry, verify_capture, CAPTURE_ROOT
    from etx.importer.triton_import import import_triton
    imported = out / "codegen/imports"
    imported.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for export in registry():
        capture = verify_capture(export)
        source, info, meta, notes = import_triton(CAPTURE_ROOT / export, "fused_moe_kernel", f"etx_imp_{export}", 1024, 1)
        (imported / f"{export}.ll").write_text(source)
        metadata[export] = info.to_json() | {"capture_spec": capture, "downgrade": notes}
    write_json(imported / "imports.json", metadata)
    return int(not result["passed"])


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
