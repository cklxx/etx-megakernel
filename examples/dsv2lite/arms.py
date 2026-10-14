"""The DeepSeek port's arms: ONE table (etx/session/experiment.py, component D of docs/SESSION-ARCH-2026-10-13.md).

The session's builds and pairs (inplace_session.py), its job graph (examples/sessions/ds_*.py), the local device build
(rocm_build_check.sh --arm <name>) and every re-derivation read this table; nothing else names an arm's levers,
flags, plan switches, invocation order, references or rule. Rules are copied verbatim from docs/DSV2-D1-D5.md, where
each was committed before its measurement (commits named in the text).

  python -m examples.dsv2lite.arms shell <arm>     the arm's build as shell assignments (rocm_build_check.sh)
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import sys

from etx.session.experiment import Arm, Rule, measure, ns
from examples.dsv2lite.acquire import RELAXED

CONDITION = "one MI300X, batch 1, context 1024, 32 tokens, repeat 2"
LEVERS_ALL = ("gemv_no_terminal_all", "norm_tree_all", "argmax_tree", "topk_rank", "attn_prologue_live")
BASE_ENV = dict(ETX_DSV2_COH="0", ETX_DSV2_D4="0", ETX_DSV2_KCHUNKS="1", ETX_DSV2_D5="0")   # plan switches, all off
D4 = ("-DETX_DSV2_D4=1",)
D4R0 = dict(ETX_DSV2_D4="1", ETX_DSV2_D4_ROWS="0")
BATCHES = (1, 2, 4, 8)                       # docs/BATCH-2026-10-14.md: the ETX arm of batch B is BATCH_ARMS[B]
BATCH_ARMS = {1: "alld4r0l1kvsmw", 2: "x2", 4: "x4", 8: "x8"}   # B = 1: the kept arm (section 9 amendment, ebb8cc3)
X8_CANDIDATES = ("x8p", "x8m", "x8k", "x8")   # the batch-8 recipe diagnostic (section 10); gates accept these
BATCH_CONDITION = "one MI300X, bf16, batch B in (1, 2, 4, 8), context 1024, 32 greedy tokens per request, repeat 2"


def keep(ref, text):
    return Rule("device_ms", ref, 30, text=text, labels=("KEEP", "NO-GO"))


def d2(cand):
    return Rule("device_ms", "default", 70, spread_factor=None, text=f"D2 rule applied to D2b: (a) every pair faster than default, median >= 70 us; (b) |median({cand} - "
                "relaxed)| <= max(2 x relaxed spread, 20 us)" if cand == "l1" else "D2: (a) every pair faster than "
                "default, median >= 70 us; (b) |median(coh - relaxed)| <= max(2 x relaxed spread, 20 us)",
                equal_to="relaxed", equal_floor_us=20)


_LEVER = dict(Q="gemv_no_terminal", Qall="gemv_no_terminal_all", N="norm_tree", Nall="norm_tree_all",
              argmax="argmax_tree", topk="topk_rank", prep="attn_prologue_live")
SINGLE = (*_LEVER, "relaxed")                             # attribution: each lever alone, no rule
ARMS = {a.name: a for a in [
    Arm("default", stamped=True),
    Arm("fleet", builder="fleet"),                                         # fleet 3fb67a6's own build (inplace_session.Session.build)
    Arm("baseline", order=("default", "fleet"), builder="none"),          # a measurement, not a build
    *(Arm(n, levers=(lever,), order=("default", n)) for n, lever in _LEVER.items()),
    Arm("relaxed", flags=(RELAXED,), order=("default", "relaxed")),     # D1: the old sc0 acquire, by name
    Arm("all", levers=LEVERS_ALL, order=("default", "all", "fleet"), references=("default", "fleet"), optional=("fleet",),
        rule=Rule("device_ms", "default", 70, text="GO: every pair faster than default, median >= 70 us, > 2 x spread"), stamped=True),
    Arm("coh", levers=("coherent_loads",), env=dict(ETX_DSV2_COH="1"), order=("default", "relaxed", "coh"),
        references=("default", "relaxed"), rule=d2("coh")),
    Arm("d4", flags=D4, env=dict(ETX_DSV2_D4="1"), order=("default", "d4"),
        rule=keep("default", "KEEP: every pair faster than default, median >= 30 us, > 2 x spread")),
    Arm("allrelaxed", levers=LEVERS_ALL, flags=(RELAXED,), order=("default", "all", "allrelaxed"), references=("default", "all")),
    Arm("d4r0", flags=D4, env=D4R0, order=("default", "d4r0")),
    Arm("alld4r0", levers=LEVERS_ALL, flags=D4, env=D4R0, order=("default", "all", "alld4r0"), references=("default", "all"),
        rule=keep("all", "KEEP: every pair faster than all, median >= 30 us, > 2 x spread (rule commit 7e08d27)"),
        stamped=True),
    Arm("l1", levers=("l1_loads",), env=dict(ETX_DSV2_COH="l1"), order=("default", "relaxed", "l1"),
        references=("default", "relaxed"), rule=d2("l1")),
    Arm("alld4r0l1", levers=(*LEVERS_ALL, "l1_loads"), flags=D4, env=dict(D4R0, ETX_DSV2_COH="l1"),
        order=("default", "alld4r0", "alld4r0l1"), references=("default", "alld4r0"), stamped=True,
        rule=keep("alld4r0", "KEEP: every pair faster than alld4r0, median >= 30 us, > 2 x spread (rule commit 8d69e52)")),
    Arm("alld3", levers=(*LEVERS_ALL, "kchunk352"), flags=("-DETX_DSV2_KCHUNKS=1",), env=dict(ETX_DSV2_KCHUNKS="4"),
        order=("all", "alld3"), references=("all",), rule=keep("all", "D3 GO only if every pair faster than all (one chunk), "
                                                         "median >= 30 us, > 2 x spread; else DROP (rule commit 1f6e3cc)")),
    Arm("alld4s", levers=LEVERS_ALL, flags=D4, env=dict(ETX_DSV2_D4="1", ETX_DSV2_D4_ROWS="32"),
        order=("default", "alld4r0", "alld4s"), references=("default", "alld4r0"),
        rule=keep("alld4r0", "KEEP: every pair faster than alld4r0, median >= 30 us, > 2 x spread (rule commit ff15c2c, brief 05)")),
    Arm("alld5", levers=LEVERS_ALL, env=dict(ETX_DSV2_D5="1"), order=("all", "alld5"), references=("all",),
        rule=Rule("device_ms", "all", 30, text="DROP if |median gain| <= spread or any pair is slower than all; KEEP only if every pair faster than "
                  "all, median >= 30 us and > 2 x spread; otherwise DROP (D5 rule commit 84eaa1b)",
                  labels=("KEEP", "DROP"))),
    # K1 and the two ablations on the kept arm (docs/DSV2-D1-D5.md, "K1", rules written before the code, 2ec2d29)
    Arm("alld4r0l1kvs", levers=(*LEVERS_ALL, "l1_loads"), flags=(*D4, "-DETX_DSV2_KVA_SHARED=1"),
        env=dict(D4R0, ETX_DSV2_COH="l1", ETX_DSV2_KVA="shared"),
        order=("alld4r0l1", "alld4r0l1kvs"), references=("alld4r0l1",), stamped=True,
        rule=Rule("device_ms", "alld4r0l1", 30, text="K1: KEEP if every pair faster than alld4r0l1, median >= 30 us, "
                  "> 2 x spread; otherwise DROP (rule commit 2ec2d29)", labels=("KEEP", "DROP"))),
    *(Arm(name, levers=tuple(x for x in (*LEVERS_ALL, "l1_loads") if x != lever), flags=D4, env=dict(D4R0, ETX_DSV2_COH="l1"),
          order=("alld4r0l1", name), references=("alld4r0l1",),
          rule=Rule("device_ms", "alld4r0l1", 0, text=f"REMOVE {lever} from the kept recipe if every pair faster than "
                    "alld4r0l1 and median > 2 x spread; otherwise it stays (rule commit 2ec2d29)", labels=("REMOVE", "STAYS")))
      for name, lever in (("alld4r0l1noQ", "gemv_no_terminal_all"), ("alld4r0l1noA", "argmax_tree"))),
    # M1 on the kept arm (docs/DSV2-D1-D5.md, "M1", rule written before the code, 3d27ff8)
    Arm("alld4r0l1mw", levers=(*LEVERS_ALL, "l1_loads", "merge_uv_waves"), flags=D4, env=dict(D4R0, ETX_DSV2_COH="l1"),
        order=("alld4r0l1", "alld4r0l1mw"), references=("alld4r0l1",),
        rule=Rule("device_ms", "alld4r0l1", 30, text="M1: KEEP if every pair faster than alld4r0l1, median >= 30 us, "
                  "> 2 x spread; otherwise DROP (rule commit 3d27ff8)", labels=("KEEP", "DROP"))),
    # KM: K1 and M1 together, against M1 alone (docs/DSV2-D1-D5.md, "KM", rule written before the code, bf92fd4)
    Arm("alld4r0l1kvsmw", levers=(*LEVERS_ALL, "l1_loads", "merge_uv_waves"), flags=(*D4, "-DETX_DSV2_KVA_SHARED=1"),
        env=dict(D4R0, ETX_DSV2_COH="l1", ETX_DSV2_KVA="shared"),
        order=("alld4r0l1", "alld4r0l1mw", "alld4r0l1kvsmw"), references=("alld4r0l1", "alld4r0l1mw"),
        rule=Rule("device_ms", "alld4r0l1mw", 30, text="KM: KEEP if every pair faster than alld4r0l1mw, median >= 30 us, "
                  "> 2 x spread; otherwise the kept arm is alld4r0l1mw (rule commit bf92fd4)", labels=("KEEP", "M1-ONLY"))),
    # batch 2 / 4 / 8 (docs/BATCH-2026-10-14.md, rules written before the code, 0fc38c9): the kept recipe without D4
    # (D4 refuses multiple requests), two workgroups per CU paired by CU, request r on slot r mod 2, no coupling.
    # Measured by examples/dsv2lite/batch_session.py against vLLM at the same B, not by `pairs`: no Rule here.
    # Scratch: x2 0; x4 32 and x8 96 B/lane on the local ROCm 7.2.4 compile (7 / 21 VGPRs holding SGPR spills; the
    # spills grow with the task types, i.e. with the requests). Allowance: twice the local figure, since the box's
    # compiler differs (docs/BATCH-2026-10-14.md, local build evidence). Section 10 gave the arms M1 and K1: on the
    # box (ROCm 7.14, one compile each, 2026-10-14) the new recipe spilled x2 32, x4 64 and x8 144 B/lane, so x2's
    # allowance rises from 0 to 64 with that measurement; x4's 64 and x8's 192 already covered it.
    *(Arm(f"x{n}", levers=(*LEVERS_ALL, "l1_loads", "merge_uv_waves"),
          flags=(f"-DETX_DS_REQUESTS={n}", "-DETX_SLOT_BY_CU") + (("-DETX_DSV2_KVA_SHARED=1",) if k1 else ()),
          env=dict(ETX_DSV2_COH="l1", ETX_DSV2_REQUESTS=str(n), ETX_DSV2_COUPLE="none",
                   **({"ETX_DSV2_KVA": "shared"} if k1 else {})),
          scratch_allow=allow)
      for n, allow, k1 in ((2, 64, True), (4, 64, True))),
    # B = 8 is a diagnostic: with M1 alone the eight-request gate failed (request 5, 3/27 layers), so the three
    # recipes are built side by side and the gate says which one is correct at eight requests. x8p is the plain
    # replication recipe (the pre-section-10 arm), x8m adds M1, x8k adds K1.
    *(Arm(name, levers=(*LEVERS_ALL, "l1_loads", *extra),
          flags=(f"-DETX_DS_REQUESTS=8", "-DETX_SLOT_BY_CU") + (("-DETX_DSV2_KVA_SHARED=1",) if k1 else ()),
          env=dict(ETX_DSV2_COH="l1", ETX_DSV2_REQUESTS="8", ETX_DSV2_COUPLE="none",
                   **({"ETX_DSV2_KVA": "shared"} if k1 else {})),
          scratch_allow=192)
      for name, extra, k1 in (("x8p", (), False), ("x8m", ("merge_uv_waves",), False),
                              ("x8k", (), True), ("x8", ("merge_uv_waves",), True))),
    Arm("allnoQ", levers=tuple(x for x in LEVERS_ALL if x != "gemv_no_terminal_all"),
        order=("default", "all", "allnoQ"), references=("default", "all"),
        rule=Rule("device_ms", "all", 0, text="every pair faster, median > 2 x spread")),
    Arm("default723", builder="rocm:latest", order=("default", "default723"),
        rule=Rule("device_ms", "default", 0, text="every pair faster, median > 2 x spread")),
]}


def build_env(arm: Arm, stamped=False) -> dict:
    if arm.builder != "etx":
        raise ValueError(f"{arm.name} uses {arm.builder}, not the ETX build/check pipeline")
    return {**BASE_ENV, **arm.env, "ETX_FLEET_PATCH": " ".join(arm.levers),
            "ETX_CFLAGS": " ".join(("-Rpass-analysis=kernel-resource-usage", *arm.flags,
                                     *(("-DETX_DSV2_STAMPS2",) if stamped else ())))}


def full_result(log: str) -> dict:
    """What every invocation must show: 32/32 HF-greedy tokens, 27/27 layers, one 32/32 warm-up, device and wall."""
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
    result = dict(device_ms=device[1], wall_ms=wall[1], tokens=32, layers=27)
    if not all(ns(result[k]) > 0 for k in ("device_ms", "wall_ms")):
        raise ValueError("invalid full timing")
    return result


parse = full_result


def checks(d: Path):
    """The re-derivations a DeepSeek record carries besides its pairs: stage tables from trace + stamps + plan."""
    for f in sorted(d.glob("batch_report*.json")):
        yield lambda f=f: (None, f"EXCLUDED {f.name}: sealed bytes only; semantic replay needs raw batch selection, "
                                "invocation failures and routing provenance, which this verifier does not replay")
    archive = archives(d)[0]
    for f in sorted(d.glob("stages_*.json")):
        arm = f.stem.removeprefix("stages_")
        stamps = next((p for p in (d / f"trace_{arm}.stamps.bin", archive / f"trace_{arm}.stamps.bin") if p.exists()), None)
        plan, trace = d / f"b_{arm}_stamps" / "plan.json", d / f"trace_{arm}.trace.bin"
        if not plan.exists() and (d / "PLAN.sha256").exists():
            plan = _regenerated_plan(d)
        if stamps is None or not (plan.exists() and trace.exists()):
            yield lambda arm=arm: (None, f"SKIP stages {arm}: needs its plan, trace and stamps (archive)")
            continue
        yield lambda arm=arm, f=f, plan=plan, trace=trace, stamps=stamps: _stages(arm, f, plan, trace, stamps)


    if (d / "followup/followup.json").exists():
        yield lambda: _followup(d / "followup")


def archives(d: Path) -> list[Path]:
    """Where a legacy result's files archived outside git (the STAMPS2 stamp files) are kept."""
    from examples.dsv2lite.pinned import CACHE
    return [CACHE / f"results_{d.name}"]


# Historical follow-up field names; orders, builds and rules still come from ARMS.
FOLLOWUPS = {"e1": ("allnoQ", {"drop_q_all": "all", "allnoQ_vs_default": "default"}),
             "e2": ("default723", {"compiler_costs": "default"})}


def followup_decision(arm, rounds, ref):
    from dataclasses import replace
    from etx.session.experiment import decide
    d = decide(replace(arm.rule, reference=ref), rounds, arm.cand)
    return dict(diffs_us=d["gains_us"], median_us=d["median_us"], spread_us=d["spread_us"], holds=d["passed"])


def _followup(fu: Path):
    want, bad = json.loads((fu / "followup.json").read_text()), []
    for tag, (name, fields) in FOLLOWUPS.items():
        arm = ARMS[name]
        rounds = measure(arm, lambda b, name: full_result((fu / f"{name}.raw.log").read_text()), tag=tag)
        for key, ref in fields.items():
            d = followup_decision(arm, rounds, ref)
            if any(want[tag][key][k] != v for k, v in d.items()):
                bad.append(key)
    return not bad, f"{'OK  ' if not bad else 'DIFF'} followup: all three stored decisions {bad}"


def _stages(arm, f, plan, trace, stamps):
    from examples.dsv2lite import stage_table as st
    from examples.dsv2lite.phase_table300 import pinned_parser
    pinned_parser()
    us, n, span = st.measured(plan, trace, stamps)
    want = json.loads(f.read_text())
    ok = (want["arm"] == arm and want["instances"] == n and us.keys() == want["stages_us"].keys()
          and abs(span - want["span_us"]) <= 1e-6 and all(abs(us[k] - want["stages_us"][k]) <= 1e-6 for k in us))
    return ok, f"{'OK  ' if ok else 'DIFF'} stages {arm:10s} span {span:.1f} us"


def _regenerated_plan(d: Path, _cache={}) -> Path:
    """A legacy result keeps only PLAN.sha256: the plan is deterministic, so compile it from this commit and check."""
    import hashlib
    import os
    import subprocess
    import tempfile
    if d not in _cache:
        out = Path(tempfile.mkdtemp(prefix="etxplan."))
        subprocess.run([sys.executable, "-m", "etx", "compile", "examples/dsv2lite/graph.py", "--arch", "gfx942",
                        "--out", str(out), "--inline-tiles", "--relay", "on"], check=True, capture_output=True,
                       env={**{k: v for k, v in os.environ.items() if not k.startswith(("ETX_", "FLEET_"))},
                            **build_env(ARMS["default"])})
        same = hashlib.sha256((out / "plan.json").read_bytes()).hexdigest() == (d / "PLAN.sha256").read_text().split()[0]
        if not same:
            raise ValueError(f"regenerated plan differs from {d / 'PLAN.sha256'}")
        _cache[d] = out / "plan.json"
    return _cache[d]


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "shell" or sys.argv[2] not in ARMS:
        raise SystemExit(f"usage: python -m examples.dsv2lite.arms shell <{'|'.join(ARMS)}>")
    a = ARMS[sys.argv[2]]
    print("\n".join(f"export {k}={shlex.quote(v)};" for k, v in build_env(a).items()))
