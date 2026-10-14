"""Fail-closed implementation of the prepared ds_inplace.py GPU session: every lever measured as whole tokens.

Every arm -- its levers, flags, plan switches, invocation order, references and pre-registered rule -- is ONE record in
examples/dsv2lite/arms.py; the decisions are etx/session/experiment.py's, on integer nanoseconds.
  build <arm> [--stamped]   the arm's build (fleet: fleet's own)
  pairs <arm>     5 interleaved rounds in the arm's order (reversed on odd rounds), every invocation 32/32 + 27/27
  trace <arm>     one stamped token of the arm's STAMPS2 build: the measured per-stage table
  report          every pairs report found + measured per-stage savings next to stage_table.json's expected ones
No invocation here provisions a GPU or downloads; inputs are prepared by the prep_* jobs. Unfiltered output is kept.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from etx.session import experiment, proc
from etx.session.record import sha256 as sha
from examples.dsv2lite.arms import ARMS, CONDITION, build_env, full_result
from examples.dsv2lite.pinned import FLEET_SRC_SHA256 as PINNED_SRC_SHA256
from examples.dsv2lite.pinned import source_sha

DATA_FILES = ("weights.bin", "weights.bin.manifest", "fleet_cache.bin", "golden_tokens.txt", "golden_hidden.bin")
# The measured inputs (2026-10-13 session, committed) ARE the inputs' identity: preflight compares the bytes on the box
# with these and nothing else (no checksum file regenerated on the box can vouch for them: review 07, R09). The model
# download does not pin a revision; these hashes identify it.
PINNED_INPUTS = json.loads((Path(__file__).resolve().parent / "results/2026-10-13-ds_inplace/inputs.json").read_text())["sha256"]


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


D4_STAGE_NOTE = ("D4 arms: topk = the ROUTED XCDs' routing prologue (only routed gate_up tasks write stamps); moe_up and "
                 "moe_down are the merged routed + shared-XCD phases (shgu / sdown join gate_up / down), so their "
                 "bodies include the shared XCDs while the topk split does not. Span, stage sums and verdicts are exact.")


class Session:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[2]
        self.logs = Path(os.environ["ETX_SESS_LOGDIR"]).resolve()
        self.fleet = Path(os.environ["ETX_DSV2_FLEET"]).resolve()
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

    def run(self, name: str, command: list[str], *, env: dict | None = None, timeout=180) -> str:
        """Fresh invocation, full stderr/stdout, independent deadline, 30-second progress, probe after 150 s silence.
        etx/session/proc.py supervises it: at the deadline, and after a normal exit, nothing it started survives."""
        path = self.logs / f"{name}.raw.log"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite prior invocation: {path}")
        clean = {k: v for k, v in os.environ.items() if not k.startswith(("ETX_", "FLEET_"))}
        clean.update(PYTHONUNBUFFERED="1")         # no caller override may leak into default or a candidate
        clean.update(env or {})
        command = ["stdbuf", "-oL", "-eL", *map(str, command)]
        text, clock = [], dict(heartbeat=0.0, output=0.0, probe=0.0)
        with path.open("w") as log:
            def sink(part):
                text.append(part); clock["output"] = time.monotonic()
                log.write(part); log.flush(); print(part, end="", flush=True)

            def on_poll(elapsed):
                now = time.monotonic()
                if elapsed - clock["heartbeat"] >= 30:
                    print(f"PROGRESS invocation={name} elapsed_s={elapsed:.1f} deadline_s={timeout}", flush=True)
                    clock["heartbeat"] = elapsed
                if now - max(clock["output"], started) >= 150 and now - max(clock["probe"], started) >= 150:
                    self.probe(log, name); clock["probe"] = time.monotonic()

            line = "$ " + shlex.join(command) + "\n"
            print(line, end="", flush=True); log.write(line)
            started = time.monotonic()
            rc = proc.run(command, sink=sink, deadline_s=timeout, on_poll=on_poll, env=clean, cwd=self.root, log=log)
            if rc == proc.DEADLINE_RC:
                message = f"DEADLINE invocation={name} rc={rc} deadline_s={timeout}; NOT COMPLETE\n"
                log.write(message); print(message, end="", flush=True)
                self.probe(log, name)
        if rc:
            raise RuntimeError(f"invocation {name} failed rc={rc}; full log {path}")
        return "".join(text)

    def args(self) -> list[str]:
        return ["--weights", str(self.data / "weights.bin"), "--cache", str(self.data / "fleet_cache.bin"),
                "--golden", str(self.data / "golden_tokens.txt"), "--hidden", str(self.data / "golden_hidden.bin"),
                "--tokens", "32", "--context", "1024", "--repeat", "2"]

    def out(self, arm: str, stamped: bool = False) -> Path:
        return self.logs / f"b_{arm}{'_stamps' if stamped else ''}"

    def source_identity(self) -> str:
        """`tools/session.sh stage` extracts the archive of a clean commit, after checking its hash, into a NEW
        directory named by that hash: the tree that runs is that archive and nothing else (R08 by construction)."""
        path = self.root / "SOURCE_IDENTITY"
        text = path.read_text() if path.is_file() else ""
        m = re.search(r"^archive ([0-9a-f]{64})$", text, re.M)
        if not re.search(r"^commit [0-9a-f]{40}$", text, re.M) or not m or not self.root.name.startswith(m[1] + "-"):
            raise RuntimeError(f"{self.root} is not a staged archive (SOURCE_IDENTITY): run tools/session.sh stage")
        return text

    def source(self):
        """Everything the builds need, checked before any build; data is checked later, by preflight."""
        if (self.fleet / ".git").exists():
            raise RuntimeError("fleet input must be a pinned export, never a repository working tree")
        if (self.fleet / "src").is_symlink() or any(p.is_symlink() for p in (self.fleet / "src").rglob("*")):
            raise RuntimeError("fleet source export must not follow symlinks into another tree")
        if source_sha(self.fleet / "src") != PINNED_SRC_SHA256:
            raise RuntimeError("fleet src is not the exact pinned 3fb67a6 export")
        from examples.dsv2lite.patch_fleet import levers
        missing = sorted({p for arm in ARMS.values() for p in arm.levers} - set(levers()))
        if missing:
            raise RuntimeError(f"levers not on this branch: {missing}")
        identity = self.source_identity()
        write_json(self.logs / "source.json", dict(fleet_commit="3fb67a6", fleet_src_sha256=PINNED_SRC_SHA256,
                                                   etx_identity=identity, arms={k: list(a.levers) for k, a in ARMS.items()}))
        self.run("environment", ["bash", "-c", "set -eu\nhipcc --version\nrocminfo\ncat SOURCE_IDENTITY"], timeout=60)
        print(f"SOURCE OK fleet export {PINNED_SRC_SHA256[:16]} etx {identity.split()[1][:12]}", flush=True)

    def preflight(self):
        actual = {}
        for name in DATA_FILES:
            path = self.data / name
            if not path.is_file():
                raise RuntimeError(f"missing input: {path}")
            print(f"PREFLIGHT hashing {path} bytes={path.stat().st_size}", flush=True)
            actual[name] = sha(path)
            if actual[name] != PINNED_INPUTS[name]:
                raise RuntimeError(f"input {name} is not the measured one ({actual[name][:16]} vs pinned "
                                   f"{PINNED_INPUTS[name][:16]}): a different experiment; refused")
        write_json(self.logs / "inputs.json", dict(data=str(self.data), sha256=actual))
        print("PREFLIGHT OK inputs are the pinned measured inputs", flush=True)

    def build(self, arm: str, stamped: bool = False):
        out = self.out(arm, stamped)
        if ARMS[arm].builder == "fleet":
            out.mkdir(parents=True)
            self.run("build_fleet_graph", [sys.executable, self.fleet / "src/host/taskgraph.py", "--kv-chunks", "16",
                                           "--emit", out / "taskgraph_d16.bin"], timeout=60)
            self.run("build_fleet_compile", ["hipcc", "--offload-arch=gfx942", "-O3", "-std=c++17",
                                              "-DFLEET_NT_WEIGHTS=1", "-Rpass-analysis=kernel-resource-usage",
                                              "-I" + str(self.fleet / "src"), self.fleet / "src/host/fleet_launch.hip",
                                              self.fleet / "src/kernels/fleet_kernel.hip", "-o", out / "run"], timeout=600)
        else:
            a = ARMS[arm]               # exactly the build rocm_build_check.sh --arm resolves
            env = dict(build_env(a, stamped), ETX_OUT=str(out), ETX_BUILD_VERBOSE="1")
            flags = shlex.split(env["ETX_CFLAGS"])
            text = self.run(f"compile_{arm}{'_stamps' if stamped else ''}",
                            ["bash", "examples/dsv2lite/build.sh", self.fleet, *flags], env=env, timeout=900)
            from examples.dsv2lite.build_resources import parse
            resources = parse(text)
            write_json(out / "resources.json", resources)
            if resources["scratch"] > a.scratch_allow:
                raise RuntimeError(f"arm {arm} spills to scratch on this box beyond its allowance "
                                   f"({a.scratch_allow} B/lane): {resources}")
            print(f"BUILD arm={arm} stamped={int(stamped)} plan_sha256={sha(out / 'plan.json')} resources={resources}", flush=True)
        print(f"BUILD arm={arm} OK", flush=True)

    def full_invocation(self, arm: str, name: str) -> dict:
        command = [self.out(arm) / "run", *self.args()]
        if arm == "fleet":
            command += ["--graph", self.out(arm) / "taskgraph_d16.bin"]
        return full_result(self.run(name, command))

    def pairs(self, arm: str):
        """Five interleaved rounds; order reversed on odd rounds; every invocation a fresh process with all gates."""
        if (self.logs / f"pairs_{arm}.json").exists():
            raise RuntimeError(f"pairs_{arm}.json already exists: use a fresh ETX_SESS_LOGDIR")
        a = ARMS[arm]
        rounds = experiment.measure(a, self.full_invocation, lambda b: (self.out(b) / "run").exists())
        report = experiment.report(a, rounds, CONDITION)
        write_json(self.logs / f"pairs_{arm}.json", report)
        v = report["verdict"]
        print(f"VERDICT arm={arm} {v['outcome']} rule={v['rule']!r}", flush=True)
        for ref, d in report["device_ms"].items():
            print(f"PAIRS arm={arm} device gain vs {ref} median {d['median_us']:.1f} us spread {d['spread_us']:.1f} "
                  f"every_positive={int(d['every_positive'])}", flush=True)

    def trace(self, arm: str):
        """One stamped run of arm's STAMPS2 build (trace + prologue stamps), then the per-stage table of that token."""
        name = f"trace_{arm}"
        env = dict(ETX_TRACE="1", ETX_TRACE_RAW=str(self.logs / f"{name}.trace.bin"),
                   ETX_DS_PROLOGUE=str(self.logs / f"{name}.stamps.bin"))
        text = self.run(name, [self.out(arm, stamped=True) / "run", *self.args()], env=env)
        full_result(text)
        from examples.dsv2lite import stage_table as st
        us, n, span = st.measured(self.out(arm, stamped=True) / "plan.json", self.logs / f"{name}.trace.bin",
                                  self.logs / f"{name}.stamps.bin")
        rec = dict(arm=arm, span_us=span, stages_us=us, instances=n)
        if arm in ("alld4r0", "alld4r0l1"):     # review 07, R11: what the D4 split means (totals and verdicts unaffected)
            rec["attribution_note"] = D4_STAGE_NOTE
        write_json(self.logs / f"stages_{arm}.json", rec)
        print(f"TRACE arm={arm} span {span:.1f} us; stages " + " ".join(f"{k}={us[k]:.1f}" for k in st.ORDER), flush=True)

    def d4trace(self):
        """One traced token of the d4 build: does every MoE layer's shared gate_up start before its routing completes?"""
        name = "trace_d4"
        env = dict(ETX_TRACE="1", ETX_TRACE_RAW=str(self.logs / f"{name}.trace.bin"))
        full_result(self.run(name, [self.out("d4") / "run", *self.args()], env=env))
        from examples.dsv2lite.d4_placement import trace_lead
        r = trace_lead(self.out("d4") / "plan.json", self.logs / f"{name}.trace.bin")
        write_json(self.logs / "d4_trace.json", r)
        print(f"D4TRACE shared gate_up ready before routing done in {r['before_routing']}/{r['layers']} MoE layers, "
              f"median lead {r['median_lead_us']:.1f} us, min {r['min_lead_us']:.1f} us", flush=True)

    def report(self):
        from examples.dsv2lite import stage_table as st
        out = {f.stem.removeprefix("pairs_"): json.loads(f.read_text()) for f in sorted(self.logs.glob("pairs_*.json"))}
        if "all" not in out:
            raise RuntimeError("the deciding arm `all` is NOT COMPLETE")
        expected = json.loads((self.root / "examples/dsv2lite/results/stages/stage_table.json").read_text())
        rows = []
        stages = {a: json.loads((self.logs / f"stages_{a}.json").read_text()) for a in ("default", "all")
                  if (self.logs / f"stages_{a}.json").exists()}
        for r in expected["rows"]:
            s = r["stage"]
            row = dict(stage=s, expected_saving_us=-r["delta_us"])
            if len(stages) == 2:
                row.update(default_us=stages["default"]["stages_us"].get(s), all_us=stages["all"]["stages_us"].get(s))
                row["measured_saving_us"] = row["default_us"] - row["all_us"]
            rows.append(row)
        out["stages"] = rows
        write_json(self.logs / "report.json", out)
        a = out["all"]["device_ms"]["default"]
        print(f"REPORT all: median {a['median_us']:.1f} us spread {a['spread_us']:.1f} {out['all']['verdict']['outcome']}; "
              f"stages measured: {len(stages) == 2}; raw logs and JSON kept in {self.logs}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("operation", choices=("source", "preflight", "build", "pairs", "trace", "d4trace", "report"))
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--stamped", action="store_true")
    a = ap.parse_args(argv)
    session = Session()
    allowed = dict(build=[n for n, x in ARMS.items() if x.builder in ("etx", "fleet")], pairs=[n for n, x in ARMS.items() if x.order],
                   trace=[n for n, x in ARMS.items() if x.stamped])
    if a.operation in allowed:
        if a.arm not in allowed[a.operation]:
            ap.error(f"{a.operation} arm must be one of {allowed[a.operation]}")
        getattr(session, a.operation)(a.arm, *([a.stamped] if a.operation == "build" else []))
    else:
        getattr(session, a.operation)()


if __name__ == "__main__":
    main()
