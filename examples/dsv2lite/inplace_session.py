"""Fail-closed implementation of the prepared ds_inplace.py GPU job protocol.

No invocation here provisions a GPU or downloads/regenerates reference data.
Unfiltered subprocess output is streamed and retained in the runner's directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import shlex
import statistics
import subprocess
import sys
import time

from etx.tools import timeline as tl

PINNED_SRC_SHA256 = "28f5cbb4f39524df318883522341c48ca4162c18b71a3a6223c52eabb40ce4b3"
PATCHES = {"default": "", "stamps": "", "Q": "gemv_no_terminal", "gamma": "norm_gamma_early",
           "tree": "norm_tree", "both": "gemv_no_terminal norm_tree"}
PHASES = ("qkv_5", "router_5")
DATA_FILES = ("weights.bin", "weights.bin.manifest", "fleet_cache.bin", "golden_tokens.txt", "golden_hidden.bin")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def source_sha(src: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(src.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(src).as_posix().encode() + b"\0" + bytes.fromhex(sha(path)))
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def paired_screen(qkv: list[float], router: list[float] | None = None) -> dict:
    if len(qkv) != 5 or (router is not None and len(router) != 5):
        raise ValueError("screen requires five complete pairs in every selected phase")
    gains = [27 * q + (26 * router[i] if router is not None else 0) for i, q in enumerate(qkv)]
    if not all(math.isfinite(v) for v in gains):
        raise ValueError("nonfinite screen gain")
    median = statistics.median(gains)
    spread = max(gains) - min(gains)
    go = min(gains) > 0 and median >= 50 and median > 2 * spread
    return dict(go=go, gains_us=gains, median_us=median, spread_us=spread,
                hundred_us_route=go and median >= 100)


def full_gain(values: list[float]) -> dict:
    if len(values) != 5 or not all(math.isfinite(v) for v in values):
        raise ValueError("full decision requires five finite paired gains")
    median = statistics.median(values); spread = max(values) - min(values)
    return dict(candidate_gain_ms=values, median_ms=median, spread_ms=spread,
                pass_70us=min(values) > 0 and median >= .070 and median > 2 * spread)


def full_result(log: str) -> dict:
    if len(re.findall(r"^tokens: 32/32 match HF greedy$", log, re.M)) != 1:
        raise ValueError("full invocation missing exact 32/32 free-greedy gate")
    if len(re.findall(r"consecutive layers within the gate: 27 of 27", log)) != 1:
        raise ValueError("full invocation missing 27/27 layer gate")
    warmups = re.findall(r"run \d+ \(warm-up\).*?: [^\n]*?(\d+)/(\d+) tokens match", log)
    if warmups != [("32", "32")]:
        raise ValueError("full invocation missing one 32/32 warmup")
    if re.search(r"MISMATCH|\[FAIL\]|no reference token", log):
        raise ValueError("full invocation contains a correctness failure")
    device = re.search(r"per-token latency .*?: median ([\d.]+) ms", log)
    wall = re.search(r"step wall \(launch to end\) ([\d.]+) ms/token", log)
    if wall is None:
        wall = re.search(r"wall time of the launches: .*? = ([\d.]+) ms/token", log)
    if device is None or wall is None:
        raise ValueError("full invocation missing device/wall timing")
    result = dict(device_ms=float(device[1]), wall_ms=float(wall[1]), tokens=32, layers=27)
    if not all(math.isfinite(result[k]) and result[k] > 0 for k in ("device_ms", "wall_ms")):
        raise ValueError("invalid full timing")
    return result


def phase_span(plan: Path, raw: Path, phase: str) -> float:
    trace = tl.build_trace(tl.load_plan(str(plan)), tl.read_raw(str(raw)))
    expected = f"iso0x0_r0_{phase}"
    tasks = [task for task in trace.tasks if task.name == expected]
    if len(tasks) != 296 or len(trace.tasks) != 296:
        raise ValueError(f"expected one 296-task {expected} phase, got {len(tasks)}/{len(trace.tasks)}")
    if any(t.ready < 0 or t.done <= t.ready for t in tasks):
        raise ValueError("missing or invalid phase trace timestamps")
    return max(t.done for t in tasks) - min(t.ready for t in tasks)


class Session:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[2]
        self.logs = Path(os.environ["ETX_SESS_LOGDIR"]).resolve()
        self.fleet = Path(os.environ.get("ETX_DSV2_FLEET", "/tmp/etx-dsv2/fleet-3fb67a6")).resolve()
        self.data = Path(os.environ["ETX_DSV2_DATA"]).resolve()
        self.logs.mkdir(parents=True, exist_ok=True)

    def probe(self, log, name: str):
        """Preserve stuck-state diagnostics; a failed probe never hides the original failure."""
        commands = [["ps", "-eo", "pid,ppid,etime,state,args"]]
        smi = Path("/opt/rocm/bin/rocm-smi")
        if "GPU" in os.environ and smi.exists():
            commands.append([str(smi), "--showuse"])
        for command in commands:
            try:
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
                value = f"PROBE invocation={name} command={shlex.join(command)} rc={result.returncode}\n{result.stdout}\n"
            except (OSError, subprocess.TimeoutExpired) as exc:
                value = f"PROBE invocation={name} failed: {exc}\n"
            log.write(value); log.flush(); print(value, end="", flush=True)

    def run(self, name: str, command: list[str], *, env: dict | None = None, timeout=180) -> tuple[str, float]:
        """Fresh invocation, full stderr/stdout, independent deadline, 30-second progress."""
        path = self.logs / f"{name}.raw.log"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite prior invocation: {path}")
        clean = {k: v for k, v in os.environ.items() if not k.startswith(("ETX_", "FLEET_"))}
        # Prevent caller variant overrides from contaminating default or a candidate.
        clean.update(PYTHONUNBUFFERED="1")
        clean.update(env or {})
        command = ["timeout", "--kill-after=5s", f"{timeout}s", "stdbuf", "-oL", "-eL", *map(str, command)]
        text = ""; warm_start = None; started = heartbeat = last_output = last_probe = time.monotonic()
        with path.open("w") as log:
            line = "$ " + shlex.join(command) + "\n"
            print(line, end="", flush=True); log.write(line)
            with subprocess.Popen(command, cwd=self.root, env=clean, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
                selector = selectors.DefaultSelector(); selector.register(proc.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    for key, _ in selector.select(timeout=1):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj); continue
                        part = chunk.decode(errors="replace"); text += part
                        last_output = time.monotonic()
                        log.write(part); log.flush(); print(part, end="", flush=True)
                        if warm_start is None and "SNAPSHOT ready " in text:
                            warm_start = time.monotonic()
                    now = time.monotonic()
                    if now - heartbeat >= 30:
                        print(f"PROGRESS invocation={name} elapsed_s={now-started:.1f} deadline_s={timeout}", flush=True)
                        heartbeat = now
                    if now - last_output >= 150 and now - last_probe >= 150:
                        self.probe(log, name); last_probe = time.monotonic()
                rc = proc.wait()
                if rc in (124, 137):
                    message = f"DEADLINE invocation={name} rc={rc} deadline_s={timeout}; NOT COMPLETE\n"
                    log.write(message); print(message, end="", flush=True)
                    self.probe(log, name)
        if rc:
            raise RuntimeError(f"invocation {name} failed rc={rc}; full log {path}")
        return text, time.monotonic() - (warm_start or started)

    def args(self, tokens=32) -> list[str]:
        return ["--weights", str(self.data / "weights.bin"), "--cache", str(self.data / "fleet_cache.bin"),
                "--golden", str(self.data / "golden_tokens.txt"), "--hidden", str(self.data / "golden_hidden.bin"),
                "--tokens", str(tokens), "--context", "1024", "--repeat", "2" if tokens == 32 else "1"]

    def out(self, arm: str, mode="full", phase="") -> Path:
        return self.logs / f"b_{arm}_{mode}{'_' + phase if phase else ''}"

    def source_identity(self) -> str:
        """The pushed tree has no .git: stage_inplace.sh writes the clean commit it pushed to SOURCE_IDENTITY."""
        path = self.root / "SOURCE_IDENTITY"
        text = path.read_text() if path.is_file() else ""
        if not re.search(r"^commit [0-9a-f]{40}$", text, re.M) or not re.search(r"^dirty 0$", text, re.M):
            raise RuntimeError(f"missing or dirty {path}: stage the box with examples/dsv2lite/stage_inplace.sh")
        return text

    def source(self):
        """Everything the builds need, checked before any build; data is checked later, by preflight."""
        if (self.fleet / ".git").exists():
            raise RuntimeError("fleet input must be a pinned export, never a repository working tree")
        if (self.fleet / "src").is_symlink() or any(p.is_symlink() for p in (self.fleet / "src").rglob("*")):
            raise RuntimeError("fleet source export must not follow symlinks into another tree")
        if source_sha(self.fleet / "src") != PINNED_SRC_SHA256:
            raise RuntimeError("fleet src is not the exact pinned 3fb67a6 export")
        identity = self.source_identity()
        write_json(self.logs / "source.json", dict(fleet_commit="3fb67a6", fleet_src_sha256=PINNED_SRC_SHA256,
                                                   etx_identity=identity))
        self.run("environment", ["bash", "-c", "set -eu\nhipcc --version\nrocminfo\ncat SOURCE_IDENTITY"], timeout=60)
        print(f"SOURCE OK fleet export {PINNED_SRC_SHA256[:16]} etx {identity.split()[1][:12]}", flush=True)

    def preflight(self):
        manifest = {}
        for line in (self.data / "SHA256SUMS").read_text().splitlines():
            match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
            if match:
                manifest[match[2].removeprefix("./")] = match[1]
        actual = {}
        for name in DATA_FILES:
            path = self.data / name
            if name not in manifest or not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"missing input or trusted checksum: {path}")
            print(f"PREFLIGHT hashing {path} bytes={path.stat().st_size}", flush=True)
            actual[name] = sha(path)
            if actual[name] != manifest[name]:
                raise RuntimeError(f"input checksum mismatch: {path}")
        if len((self.data / "golden_tokens.txt").read_text().split()) < 33:
            raise RuntimeError("golden needs 33 tokens for 32 decode comparisons")
        if (self.data / "golden_hidden.bin").stat().st_size != 27 * 2048 * 4:
            raise RuntimeError("hidden golden is not 27 x 2048 fp32")
        write_json(self.logs / "inputs.json", dict(fleet_commit="3fb67a6", fleet_src_sha256=PINNED_SRC_SHA256,
                                                  data=str(self.data), sha256=actual))
        print("PREFLIGHT OK inputs match their SHA256SUMS", flush=True)

    def build(self, arm: str):
        if arm == "fleet":
            out = self.out(arm); out.mkdir(parents=True)
            self.run("build_fleet_graph", [sys.executable, self.fleet / "src/host/taskgraph.py", "--kv-chunks", "16",
                                           "--emit", out / "taskgraph_d16.bin"], timeout=60)
            self.run("build_fleet_compile", ["hipcc", "--offload-arch=gfx942", "-O3", "-std=c++17",
                                              "-DFLEET_NT_WEIGHTS=1", "-Rpass-analysis=kernel-resource-usage",
                                              "-I" + str(self.fleet / "src"), self.fleet / "src/host/fleet_launch.hip",
                                              self.fleet / "src/kernels/fleet_kernel.hip", "-o", out / "run"], timeout=600)
        else:
            specs = [("full", "")]
            if arm == "default":
                specs += [("capture", ph) for ph in PHASES]
            if arm != "stamps":
                specs += [("replay", ph) for ph in (("qkv_5",) if arm == "Q" else PHASES)]
            for mode, phase in specs:
                out = self.out(arm, mode, phase)
                env = dict(ETX_OUT=str(out), ETX_FLEET_PATCH=PATCHES[arm], ETX_BUILD_VERBOSE="1")
                flags = ["-Rpass-analysis=kernel-resource-usage"]
                if mode != "full":
                    env["ETX_DSV2_SNAPSHOT"] = f"{mode}:0:{phase}"
                    flags += ["-DETX_DSV2_SNAPSHOT", "-DETX_EV_STRIDE=32", "-DETX_ACQUIRE_DOMAIN_RELAXED"]
                if arm == "stamps":
                    flags.append("-DETX_DSV2_STAMPS2")
                text, _ = self.run(f"compile_{arm}_{mode}_{phase or 'all'}", ["bash", "examples/dsv2lite/build.sh", self.fleet, *flags],
                                   env=env, timeout=600)
                from examples.dsv2lite.build_resources import parse
                resources = parse(text)
                write_json(out / "resources.json", resources)
                print(f"BUILD arm={arm} mode={mode} phase={phase or 'all'} resources={resources}", flush=True)
        print(f"BUILD arm={arm} OK", flush=True)

    def full_invocation(self, arm: str, name: str) -> dict:
        command = [self.out(arm) / "run", *self.args()]
        if arm == "fleet":
            command += ["--graph", self.out(arm) / "taskgraph_d16.bin"]
        text, _ = self.run(name, command)
        return full_result(text)

    def pairs(self, arm: str, baseline=False):
        if not baseline:
            screen = json.loads((self.logs / f"screen_{arm}.json").read_text())
            if not screen["go"]:
                raise RuntimeError(f"refusing unscreened full arm {arm}")
        name = "baseline" if baseline else f"full_{arm}"
        pairs = []
        for i in range(5):
            result = {}
            order = ("default", arm) if baseline else ("default", arm, "fleet")
            if i % 2:
                order = tuple(reversed(order))
            for current in order:
                result[current] = self.full_invocation(current, f"{name}_p{i}_{current}")
            pairs.append(result)
        report = dict(arm=arm, pairs=pairs, device_condition="one MI300X, batch 1, context 1024, 32 tokens, repeat 2",
                      wall_condition="ETX per-token launch; fleet shipped default 32 tokens per launch")
        for key in ("device_ms", "wall_ms"):
            report[key] = {}
            for reference in (("default",) if baseline else ("default", "fleet")):
                values = [p[reference][key] - p[arm][key] for p in pairs]
                report[key][reference] = full_gain(values)
        write_json(self.logs / f"{name}.json", report)
        print(f"{'BASELINE' if baseline else 'FULL'} arm={arm} complete 5 pairs, every invocation 32/32 +27/27 "
              f"device={report['device_ms']} wall={report['wall_ms']}", flush=True)

    def snapshot_env(self, mode: str, phase: str, name: str) -> dict:
        return dict(ETX_DSV2_SNAPSHOT_MODE=mode, ETX_DSV2_SNAPSHOT_PHASE=phase,
                    ETX_DSV2_SNAPSHOT_FILE=str(self.logs / f"input_{phase}.bin"),
                    ETX_DSV2_SNAPSHOT_OUTPUT=str(self.logs / f"{name}.output.bin"),
                    ETX_TRACE="1", ETX_TRACE_RAW=str(self.logs / f"{name}.trace.bin"))

    def capture(self):
        snapshots = {}
        for phase in PHASES:
            name = f"capture_{phase}"
            text, _ = self.run(name, [self.out("default", "capture", phase) / "run", *self.args(1)],
                               env=self.snapshot_env("capture", phase, name))
            if f"SNAPSHOT complete mode=capture phase={phase}" not in text:
                raise RuntimeError("snapshot capture incomplete")
            snapshots[phase] = sha(self.logs / f"input_{phase}.bin")
        write_json(self.logs / "snapshot_inputs.json", snapshots)

    def screen(self, arm: str):
        phases = ("qkv_5",) if arm == "Q" else PHASES
        snapshots = json.loads((self.logs / "snapshot_inputs.json").read_text())
        deltas = {phase: [] for phase in phases}; pairs = []; warm_seconds = 0.0
        for i in range(5):
            samples = {}
            for current in (("default", arm) if i % 2 == 0 else (arm, "default")):
                samples[current] = {}
                for phase in phases:
                    if sha(self.logs / f"input_{phase}.bin") != snapshots[phase]:
                        raise RuntimeError("saved phase input changed")
                    name = f"screen_{arm}_p{i}_{current}_{phase}"
                    out = self.out(current, "replay", phase)
                    text, warm = self.run(name, [out / "run", *self.args(1)], env=self.snapshot_env("replay", phase, name))
                    warm_seconds += warm
                    if warm_seconds > 300:
                        raise RuntimeError("five warm-GPU-minute decider cap exceeded")
                    if f"SNAPSHOT complete mode=replay phase={phase}" not in text:
                        raise RuntimeError("snapshot replay incomplete")
                    span = phase_span(out / "plan.json", self.logs / f"{name}.trace.bin", phase)
                    samples[current][phase] = dict(span_us=span, output_sha256=sha(self.logs / f"{name}.output.bin"))
            for phase in phases:
                ctl, cand = samples["default"][phase], samples[arm][phase]
                if ctl["output_sha256"] != cand["output_sha256"]:
                    raise RuntimeError(f"phase output byte mismatch arm={arm} phase={phase} pair={i}")
                deltas[phase].append(ctl["span_us"] - cand["span_us"])
            pairs.append(samples)
        result = paired_screen(deltas["qkv_5"], deltas.get("router_5"))
        result.update(arm=arm, pairs=pairs, delta_phase_us=deltas, snapshot_sha256=snapshots, warm_seconds=warm_seconds)
        write_json(self.logs / f"screen_{arm}.json", result)
        print(f"SCREEN arm={arm} GO={int(result['go'])} median_us={result['median_us']:.6f} "
              f"spread_us={result['spread_us']:.6f} gains_us={result['gains_us']} "
              f"hundred_us_route={int(result['hundred_us_route'])}", flush=True)

    def trace(self):
        env = dict(ETX_TRACE="1", ETX_TRACE_RAW=str(self.logs / "default.trace.bin"),
                   ETX_DS_PROLOGUE=str(self.logs / "default.stamps.bin"))
        text, _ = self.run("trace_default", [self.out("stamps") / "run", *self.args()], env=env)
        full_result(text)
        self.run("trace_ledger", [sys.executable, "-m", "examples.dsv2lite.ledger", "--etx",
                                  self.out("stamps") / "plan.json", self.logs / "default.trace.bin",
                                  self.logs / "default.stamps.bin", "--fleet-src", self.fleet / "src"], timeout=60)
        print("TRACE default completed 32/32 +27/27; instrumentation timing excluded from comparisons", flush=True)

    def report(self):
        result = {}
        for arm in ("Q", "gamma", "tree", "both"):
            screen = json.loads((self.logs / f"screen_{arm}.json").read_text())
            full = self.logs / f"full_{arm}.json"
            if screen["go"] and not full.exists():
                raise RuntimeError(f"passing arm {arm} is NOT COMPLETE (no full five-pair result)")
            result[arm] = dict(screen=screen, full=json.loads(full.read_text()) if full.exists() else "REJECTED BY SCREEN")
        write_json(self.logs / "report.json", result)
        print("REPORT complete; raw and JSON evidence retained; screen estimates are not full-token claims", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("operation", choices=("source", "preflight", "build", "baseline", "capture", "screen", "full", "trace", "report"))
    ap.add_argument("arm", nargs="?", choices=("default", "fleet", "stamps", "Q", "gamma", "tree", "both"))
    args = ap.parse_args(argv)
    session = Session()
    if args.operation in ("build", "screen", "full") and args.arm is None:
        ap.error(f"{args.operation} requires arm")
    if args.operation == "baseline":
        session.pairs("fleet", baseline=True)
    elif args.operation == "full":
        session.pairs(args.arm)
    else:
        method = getattr(session, args.operation)
        method(args.arm) if args.operation in ("build", "screen") else method()


if __name__ == "__main__":
    main()
