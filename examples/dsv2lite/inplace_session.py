"""Fail-closed implementation of the prepared ds_inplace.py GPU session: every lever measured as whole tokens.

Arms (ARMS): default, fleet (shipped 3fb67a6), each lever alone, and `all` (every lever that passed its local gate).
  baseline        5 interleaved pairs default / fleet
  full <arm>      5 interleaved pairs default / arm (/ fleet for `all`), every invocation 32/32 tokens + 27/27 layers
  trace <arm>     one stamped token of the STAMPS2 build of default or `all`: the measured per-stage table
  report          verdicts + measured per-stage table next to the expected one (stage_table.py)
Decision (fixed before the run, for `all` only): every paired device gain > 0, median >= 70 us, median > 2 x spread.
Single levers are attribution: median and spread are reported, no verdict (most are below 5 pairs' resolution).
No invocation here provisions a GPU or downloads; inputs are prepared by the prep_* jobs. Unfiltered output is kept.
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

from examples.dsv2lite.acquire import RELAXED
from examples.dsv2lite.pinned import FLEET_SRC_SHA256 as PINNED_SRC_SHA256
from examples.dsv2lite.pinned import source_sha

LEVERS_ALL = ("gemv_no_terminal_all", "norm_tree_all", "argmax_tree", "topk_rank", "attn_prologue_live")
ARMS = {"default": (), "Q": ("gemv_no_terminal",), "Qall": ("gemv_no_terminal_all",), "N": ("norm_tree",),
        "Nall": ("norm_tree_all",), "argmax": ("argmax_tree",), "topk": ("topk_rank",),
        "prep": ("attn_prologue_live",), "all": LEVERS_ALL}
# Arms that differ by a build flag instead of a lever patch. "relaxed" is D1's comparison: the safe default against
# the old sc0 acquire, named explicitly (examples/dsv2lite/acquire.py).
ARM_FLAGS = {"relaxed": [RELAXED]}
ARMS["relaxed"] = ()
# D2: fleet's tiles load every payload coherently (lever coherent_loads) and the plan drops their acquire invalidates
# (ETX_DSV2_COH=1); the build refuses one without the other (coh_check.py). Measured against BOTH default (sc1) and
# relaxed in the same rounds: docs/DSV2-D1-D5.md, D2.
ARMS["coh"] = ("coherent_loads",)
ARM_ENV = {"coh": {"ETX_DSV2_COH": "1"}}
D2_MIN_TOL_US = 20.0
# D4: shared experts during routing, routed slices on the shared XCDs (d4_placement.py; the plan is checked at build
# time). Plan switch + tile switch; pairs against default, its own rule, and one traced token for criterion (a).
ARMS["d4"] = ()
ARM_FLAGS["d4"] = ["-DETX_DSV2_D4=1"]
ARM_ENV["d4"] = {"ETX_DSV2_D4": "1"}
D4_KEEP_US = 30.0
# measurements only (docs/DSV2-D1-D5.md, "Next two arms"): the unsafe acquire on the best arm, and D4 without slices
ARMS["allrelaxed"] = LEVERS_ALL
ARM_FLAGS["allrelaxed"] = [RELAXED]
ARMS["d4r0"] = ()
ARM_FLAGS["d4r0"] = ["-DETX_DSV2_D4=1"]
ARM_ENV["d4r0"] = {"ETX_DSV2_D4": "1", "ETX_DSV2_D4_ROWS": "0"}
# D2b (docs/DSV2-D1-D5.md): l1 alone under D2's rule; alld4r0l1 = the best arm + D2b, KEEP rule against alld4r0
ARMS["l1"] = ("l1_loads",)
ARM_ENV["l1"] = {"ETX_DSV2_COH": "l1"}
ARMS["alld4r0l1"] = (*LEVERS_ALL, "l1_loads")
ARM_FLAGS["alld4r0l1"] = ["-DETX_DSV2_D4=1"]
ARM_ENV["alld4r0l1"] = {"ETX_DSV2_D4": "1", "ETX_DSV2_D4_ROWS": "0", "ETX_DSV2_COH": "l1"}
# D3 (docs/DSV2-D1-D5.md): four 352-row K-chunks on top of the five levers, 2-way against `all`
ARMS["alld3"] = (*LEVERS_ALL, "kchunk352")
ARM_FLAGS["alld3"] = ["-DETX_DSV2_KCHUNKS=1"]
ARM_ENV["alld3"] = {"ETX_DSV2_KCHUNKS": "4"}
ARMS["alld4r0"] = LEVERS_ALL                       # candidate: the best arm + D4 placement without slices
ARM_FLAGS["alld4r0"] = ["-DETX_DSV2_D4=1"]
ARM_ENV["alld4r0"] = dict(ARM_ENV["d4r0"])
# D4 slices: paired against both default and the measured best arm, alld4r0. The KEEP decision is against alld4r0.
ARMS["alld4s"] = LEVERS_ALL
ARM_FLAGS["alld4s"] = ["-DETX_DSV2_D4=1"]
ARM_ENV["alld4s"] = {"ETX_DSV2_D4": "1", "ETX_DSV2_D4_ROWS": "32"}
SINGLE = ("Q", "Qall", "N", "Nall", "argmax", "topk", "prep", "relaxed")
STAMPED = ("default", "all")
TRACEABLE = (*STAMPED, "alld4r0", "alld4r0l1")    # arms whose stamped build a session may trace (the new arms' stages)
DATA_FILES = ("weights.bin", "weights.bin.manifest", "fleet_cache.bin", "golden_tokens.txt", "golden_hidden.bin")
GO_US = 70.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def paired(gains_us: list[float]) -> dict:
    """Five paired gains (reference minus candidate, us): median, full-range spread, and the decision rule."""
    if len(gains_us) != 5 or not all(math.isfinite(v) for v in gains_us):
        raise ValueError("a paired result needs five finite gains")
    median = statistics.median(gains_us)
    spread = max(gains_us) - min(gains_us)
    return dict(gains_us=gains_us, median_us=median, spread_us=spread, every_positive=min(gains_us) > 0,
                go=min(gains_us) > 0 and median >= GO_US and median > 2 * spread)


def d2_verdict(rounds: list[dict], cand: str = "coh") -> dict:
    """D2's rule (docs/DSV2-D1-D5.md, fixed before the code), device basis, the same five rounds:
    (a) coh vs default (sc1): every pair faster and median >= 70 us;
    (b) coh vs relaxed: |median(coh - relaxed)| <= max(2 x spread of the relaxed-vs-default pairs, 20 us)."""
    a = paired([1000 * (r["default"]["device_ms"] - r[cand]["device_ms"]) for r in rounds])
    rel = paired([1000 * (r["default"]["device_ms"] - r["relaxed"]["device_ms"]) for r in rounds])
    b = [1000 * (r[cand]["device_ms"] - r["relaxed"]["device_ms"]) for r in rounds]
    b_median = statistics.median(b)
    tol = max(2 * rel["spread_us"], D2_MIN_TOL_US)
    a_ok = a["every_positive"] and a["median_us"] >= GO_US
    b_ok = abs(b_median) <= tol
    return dict(a=a, relaxed_vs_default=rel, b_gains_us=b, b_median_us=b_median, b_tol_us=tol, a_ok=a_ok, b_ok=b_ok,
                go=a_ok and b_ok)


def d4_verdict(vs_default: dict) -> dict:
    """D4's speed rule (docs/DSV2-D1-D5.md): every pair faster, median >= 30 us and > 2 x spread. Criterion (a), the
    shared XCDs' gate_up ready before the layer's routing completes, comes from `d4trace`."""
    return dict(vs_default, keep=vs_default["every_positive"] and vs_default["median_us"] >= D4_KEEP_US
                and vs_default["median_us"] > 2 * vs_default["spread_us"])


# Every measured arm's pre-registered rule, as written in docs/DSV2-D1-D5.md before its first measurement (commit of
# the rule in the doc). verdict() records the outcome in ONE field so no arm reports a blank decision.
RULES = {
    "all": ("GO: every pair faster than default, median >= 70 us, > 2 x spread", "go", "default"),
    "coh": ("D2: (a) every pair faster than default, median >= 70 us; (b) |median(coh - relaxed)| <= max(2 x relaxed spread, 20 us)", "d2", None),
    "l1": ("D2 rule applied to D2b: (a) every pair faster than default, median >= 70 us; (b) |median(l1 - relaxed)| <= max(2 x relaxed spread, 20 us)", "d2", None),
    "d4": ("KEEP: every pair faster than default, median >= 30 us, > 2 x spread", "d4", None),
    "alld4r0": ("KEEP: every pair faster than all, median >= 30 us, > 2 x spread (rule commit 7e08d27)", "keep_vs_all", None),
    "alld4r0l1": ("KEEP: every pair faster than alld4r0, median >= 30 us, > 2 x spread (rule commit 8d69e52)", "keep_vs_alld4r0", None),
    "alld4s": ("KEEP: every pair faster than alld4r0, median >= 30 us, > 2 x spread (rule commit ff15c2c, brief 05)", "keep_vs_alld4r0", None),
    "alld3": ("D3 GO only if every pair faster than all (one chunk), median >= 30 us, > 2 x spread; else DROP (rule commit 1f6e3cc)", "keep_vs_all", None),
}


def verdict(arm: str, report: dict) -> dict:
    if arm not in RULES:
        return dict(rule="attribution or measurement only: no decision", outcome="NO RULE (attribution)")
    rule, field, ref = RULES[arm]
    if field == "go":
        ok = report["device_ms"][ref]["median_us"] >= GO_US and report["device_ms"][ref]["every_positive"] and \
            report["device_ms"][ref]["median_us"] > 2 * report["device_ms"][ref]["spread_us"]
    elif field == "d2":
        ok = report["d2"]["go"]
    else:
        ok = report[field]["keep"]
    return dict(rule=rule, outcome=("GO" if field in ("go", "d2") else "KEEP") if ok else "NO-GO", passed=bool(ok))


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
        """Fresh invocation, full stderr/stdout, independent deadline, 30-second progress, probe after 150 s silence."""
        path = self.logs / f"{name}.raw.log"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite prior invocation: {path}")
        clean = {k: v for k, v in os.environ.items() if not k.startswith(("ETX_", "FLEET_"))}
        clean.update(PYTHONUNBUFFERED="1")         # no caller override may leak into default or a candidate
        clean.update(env or {})
        command = ["timeout", "--kill-after=5s", f"{timeout}s", "stdbuf", "-oL", "-eL", *map(str, command)]
        text = ""; started = heartbeat = last_output = last_probe = time.monotonic()
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
        return text

    def args(self) -> list[str]:
        return ["--weights", str(self.data / "weights.bin"), "--cache", str(self.data / "fleet_cache.bin"),
                "--golden", str(self.data / "golden_tokens.txt"), "--hidden", str(self.data / "golden_hidden.bin"),
                "--tokens", "32", "--context", "1024", "--repeat", "2"]

    def out(self, arm: str, stamped: bool = False) -> Path:
        return self.logs / f"b_{arm}{'_stamps' if stamped else ''}"

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
        from examples.dsv2lite.patch_fleet import levers
        missing = sorted({p for arm in ARMS.values() for p in arm} - set(levers()))
        if missing:
            raise RuntimeError(f"levers not on this branch: {missing}")
        identity = self.source_identity()
        write_json(self.logs / "source.json", dict(fleet_commit="3fb67a6", fleet_src_sha256=PINNED_SRC_SHA256,
                                                   etx_identity=identity, arms={k: list(v) for k, v in ARMS.items()}))
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
        write_json(self.logs / "inputs.json", dict(data=str(self.data), sha256=actual))
        print("PREFLIGHT OK inputs match their SHA256SUMS", flush=True)

    def build(self, arm: str, stamped: bool = False):
        out = self.out(arm, stamped)
        if arm == "fleet":
            out.mkdir(parents=True)
            self.run("build_fleet_graph", [sys.executable, self.fleet / "src/host/taskgraph.py", "--kv-chunks", "16",
                                           "--emit", out / "taskgraph_d16.bin"], timeout=60)
            self.run("build_fleet_compile", ["hipcc", "--offload-arch=gfx942", "-O3", "-std=c++17",
                                              "-DFLEET_NT_WEIGHTS=1", "-Rpass-analysis=kernel-resource-usage",
                                              "-I" + str(self.fleet / "src"), self.fleet / "src/host/fleet_launch.hip",
                                              self.fleet / "src/kernels/fleet_kernel.hip", "-o", out / "run"], timeout=600)
        else:
            # plan switches default off; an arm's own switches override them (a dict update, not repeated keywords:
            # the 2026-10-13 d1d5 run lost build_coh / build_d4 to "multiple values for keyword argument")
            env = dict(ETX_OUT=str(out), ETX_FLEET_PATCH=" ".join(ARMS[arm]), ETX_BUILD_VERBOSE="1",
                       ETX_DSV2_COH="0", ETX_DSV2_D4="0", ETX_DSV2_KCHUNKS="1")
            env.update(ARM_ENV.get(arm, {}))
            flags = (["-Rpass-analysis=kernel-resource-usage"] + ARM_FLAGS.get(arm, [])
                     + (["-DETX_DSV2_STAMPS2"] if stamped else []))
            text = self.run(f"compile_{arm}{'_stamps' if stamped else ''}",
                            ["bash", "examples/dsv2lite/build.sh", self.fleet, *flags], env=env, timeout=900)
            from examples.dsv2lite.build_resources import parse
            resources = parse(text)
            write_json(out / "resources.json", resources)
            if resources["scratch"] > 0:
                raise RuntimeError(f"arm {arm} spills to scratch on this box: {resources}")
            print(f"BUILD arm={arm} stamped={int(stamped)} resources={resources}", flush=True)
        print(f"BUILD arm={arm} OK", flush=True)

    def full_invocation(self, arm: str, name: str) -> dict:
        command = [self.out(arm) / "run", *self.args()]
        if arm == "fleet":
            command += ["--graph", self.out(arm) / "taskgraph_d16.bin"]
        return full_result(self.run(name, command))

    def pairs(self, arm: str):
        """Five interleaved rounds; order reversed on odd rounds; every invocation a fresh process with all gates."""
        if arm == "alld4s" and (self.logs / "pairs_alld4s.json").exists():
            raise RuntimeError("alld4s report already exists; choose a fresh ETX_SESS_LOGDIR for new pairs")
        if arm == "baseline":
            order, refs, cand = ("default", "fleet"), ("default",), "fleet"
        elif arm == "all":
            order, refs, cand = ("default", "all", "fleet"), ("default", "fleet"), "all"
            if not (self.out("fleet") / "run").exists():          # the decision is vs default; fleet is the context
                print("PAIRS arm=all: fleet binary missing, measuring against default only", flush=True)
                order, refs = ("default", "all"), ("default",)
        elif arm in ("coh", "l1"):
            order, refs, cand = ("default", "relaxed", arm), ("default", "relaxed"), arm
        elif arm == "alld4r0l1":
            order, refs, cand = ("default", "alld4r0", arm), ("default", "alld4r0"), arm
        elif arm == "alld3":
            order, refs, cand = ("all", arm), ("all",), arm
        elif arm in ("allrelaxed", "alld4r0"):
            order, refs, cand = ("default", "all", arm), ("default", "all"), arm
        elif arm == "alld4s":
            order, refs, cand = ("default", "alld4r0", "alld4s"), ("default", "alld4r0"), "alld4s"
        else:
            order, refs, cand = ("default", arm), ("default",), arm
        rounds = []
        for i in range(5):
            result = {}
            for current in (order if i % 2 == 0 else tuple(reversed(order))):
                result[current] = self.full_invocation(current, f"{arm}_r{i}_{current}")
            rounds.append(result)
        report = dict(arm=arm, rounds=rounds, condition="one MI300X, batch 1, context 1024, 32 tokens, repeat 2")
        for key in ("device_ms", "wall_ms"):
            report[key] = {}
            for ref in refs:
                report[key][ref] = paired([1000 * (r[ref][key] - r[cand][key]) for r in rounds])
        if arm == "alld4r0":
            report["keep_vs_all"] = d4_verdict(report["device_ms"]["all"])
        if arm == "alld4r0l1":
            report["keep_vs_alld4r0"] = d4_verdict(report["device_ms"]["alld4r0"])
        if arm == "alld4s":
            report["keep_vs_alld4r0"] = d4_verdict(report["device_ms"]["alld4r0"])
        if arm == "alld3":
            report["keep_vs_all"] = d4_verdict(report["device_ms"]["all"])
        if arm in ("coh", "l1"):
            report["d2"] = d2_verdict(rounds, arm)
        elif arm == "d4":
            report["d4"] = d4_verdict(report["device_ms"]["default"])
        if arm != "all":
            for basis in report["device_ms"].values():
                basis["go"] = None          # per-reference fields carry no decision; the arm's rule is in `verdict`
        report["verdict"] = verdict(arm, report)
        write_json(self.logs / f"pairs_{arm}.json", report)
        v = report["verdict"]
        print(f"VERDICT arm={arm} {v['outcome']} rule={v['rule']!r}", flush=True)
        base = "default" if "default" in report["device_ms"] else refs[0]     # 2-way arms against `all` (alld3, alld5)
        d = report["device_ms"][base]
        if arm in ("coh", "l1"):
            v = report["d2"]
            print(f"PAIRS arm={arm} D2 (a) vs sc1 median {v['a']['median_us']:.1f} us spread {v['a']['spread_us']:.1f} "
                  f"every_positive={int(v['a']['every_positive'])} ok={v['a_ok']}; (b) coh - relaxed median "
                  f"{v['b_median_us']:.1f} us tolerance {v['b_tol_us']:.1f} ok={v['b_ok']}; GO={v['go']}", flush=True)
        if arm == "d4":
            v = report["d4"]
            print(f"PAIRS arm=d4 D4 median {v['median_us']:.1f} us spread {v['spread_us']:.1f} every_positive="
                  f"{int(v['every_positive'])} KEEP={v['keep']}", flush=True)
        if arm == "alld4s":
            v = report["keep_vs_alld4r0"]
            print(f"PAIRS arm=alld4s device gain vs alld4r0 median {v['median_us']:.1f} us spread "
                  f"{v['spread_us']:.1f} every_positive={int(v['every_positive'])} KEEP={v['keep']}", flush=True)
        print(f"PAIRS arm={arm} device gain vs {base} median {d['median_us']:.1f} us spread {d['spread_us']:.1f} "
              f"every_positive={int(d['every_positive'])} GO={d['go']}", flush=True)

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
        write_json(self.logs / f"stages_{arm}.json", dict(arm=arm, span_us=span, stages_us=us, instances=n))
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
        out = dict()
        for arm in ("baseline", "all", *SINGLE, "coh", "d4"):
            path = self.logs / f"pairs_{arm}.json"
            out[arm] = json.loads(path.read_text()) if path.exists() else "NOT COMPLETE"
        if out["all"] == "NOT COMPLETE":
            raise RuntimeError("the deciding arm `all` is NOT COMPLETE")
        expected = json.loads((self.root / "examples/dsv2lite/results/stages/stage_table.json").read_text())
        rows = []
        stages = {a: json.loads((self.logs / f"stages_{a}.json").read_text()) for a in STAMPED
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
        print(f"REPORT all: median {a['median_us']:.1f} us spread {a['spread_us']:.1f} GO={a['go']}; "
              f"stages measured: {len(stages) == 2}; raw logs and JSON kept in {self.logs}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("operation", choices=("source", "preflight", "build", "pairs", "trace", "d4trace", "report"))
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--stamped", action="store_true")
    a = ap.parse_args(argv)
    session = Session()
    if a.operation == "build":
        if a.arm not in (*ARMS, "fleet"):
            ap.error(f"build arm must be one of {sorted((*ARMS, 'fleet'))}")
        session.build(a.arm, a.stamped)
    elif a.operation == "pairs":
        if a.arm not in ("baseline", "all", *SINGLE, "coh", "d4", "allrelaxed", "d4r0", "alld4r0", "l1", "alld4r0l1",
                         "alld3", "alld4s"):
            ap.error("pairs arm must be baseline, all, coh, d4 or a single lever")
        session.pairs(a.arm)
    elif a.operation == "trace":
        if a.arm not in TRACEABLE:
            ap.error(f"trace arm must be one of {TRACEABLE}")
        session.trace(a.arm)
    else:
        getattr(session, a.operation)()


if __name__ == "__main__":
    main()
