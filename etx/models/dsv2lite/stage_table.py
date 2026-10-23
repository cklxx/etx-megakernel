"""Per-stage account of the ETX DeepSeek token: measured now, expected after each stage's lever, and the step it adds to.

  python -m examples.dsv2lite.stage_table      # writes examples/dsv2lite/results/stages/stage_table.{json,md}
  main(out=dir)                                # writes them to dir (tests: never the committed files)

Stages (kailun's 14) partition the traced stride-32 token exactly (x1s, 3500.96 us embed -> argmax):
  prologue  embed + every phase of the dense layer 0          tail      layer 26 after its qkv (attn .. down)
  in_norm   qkv's fold + RMSNorm (staging + prologue), L1-26   qkv       the rest of qkv (q / kv_a GEMVs), L1-26
  prep      attention's kv_post + q gather (staging+prologue)  attn      the rest of attention (cache stream, softmax)
  reduce    merge_uv;  o_proj;  router;  L1-25                 topk      gate_up's routing (staging + prologue), L1-25
  moe_up    the rest of gate_up; moe_down = down; L1-25        lm_head, argmax
Every row keeps its own handoff, ramp and slow-task tail (the ledger's categories), so the rows sum to the token. The
staging / prologue split of the stride-32 trace (which has no prologue stamps) uses the stamped 09-27b trace's ratios
per phase [E], as phase_table300.py does; every other current value is [M] trace arithmetic.

Every expected saving is a per-instance number with one of three bases, computed below from the files it cites:
  BYTES  bytes removed / the measured streamed rate (fleet STATUS.md: 4.2 TB/s)          (upper bound: see notes)
  PHASE  a measured cost: hw_bench barrier / LDS probe, the stamped routing and gamma costs (MODEL-dsv2.md:157)
  SIM    the calibrated replay (replay.py): measured task bodies and residuals through etx.sim.xcd; the summary's
         simulated line shrinks each stage's task bodies by its saving and lets the dependencies decide the step.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from etx.models.declarations import declaration as _declaration
from . import paths

from . import ledger as ld
from .phase_table300 import aggregate
from .phase_table300 import pinned_parser
data_sets = _declaration(paths.ROOT, "dsv2lite.pinned").data_sets
fleet_export = _declaration(paths.ROOT, "dsv2lite.pinned").fleet_export
from .replay import default_replay

OUT = paths.ROOT / "examples/dsv2lite/results/stages"   # the committed table
UNTRACED_DEVICE_US = 3473.5        # t1_x1s / t2_x1s medians 3.474 / 3.473 ms (results/2026-09-27d/evstride, last lines)
LIMIT_US = 1238.095                # 5.20 GB / 4.2 TB/s, the byte ceiling (fleet STATUS.md:42-43)
MOE = range(1, 27)                 # layers 1-26 are MoE (graph.py:35-37)


def measured_inputs() -> dict:
    """Every pricing constant, read from the file that measured it."""
    d = data_sets() / "results_2026-09-27d"
    wg = (d / "hw_wg.log").read_text()
    model = (paths.ROOT / "docs/deepseek/MODEL-dsv2.md").read_text()
    status = (fleet_export() / "docs/STATUS.md").read_text()
    m = {
        "barrier_us": float(re.search(r"__syncthreads\s+([\d.]+) ns", wg)[1]) / 1000,
        "lds_probe_us": float(re.search(r"LDS 8 KB write\+read \(2 barriers\)\s+([\d.]+) ns", wg)[1]) / 1000,
        "bperm_us": float(re.search(r"(\d+) serial bpermutes x (\d+) ns = ([\d.]+) us", model)[2]) / 1000,
        "routing_us": float(re.search(r"(\d+) serial bpermutes x (\d+) ns = ([\d.]+) us", model)[3]),
        "routing_bperms": int(re.search(r"(\d+) serial bpermutes x (\d+) ns", model)[1]),
        "gamma_rt_us": float(re.search(r"gamma round trip \(~([\d.]+) us\)", model)[1]),
        "rate_bytes_per_us": float(re.search(r"Streamed read bandwidth, best depth \| ([\d.]+) TB/s", status)[1]) * 1e6,
    }
    # Keep historical citation labels stable; the document is read at its canonical path above.
    m["sources"] = {"barrier_us": "results_2026-09-27d/hw_wg.log, __syncthreads",
                    "lds_probe_us": "results_2026-09-27d/hw_wg.log, LDS 8 KB write+read (2 barriers)",
                    "bperm_us": "docs/MODEL-dsv2.md:157, fleet's routing stamp",
                    "routing_us": "docs/MODEL-dsv2.md:157, fleet's routing stamp",
                    "routing_bperms": "docs/MODEL-dsv2.md:157",
                    "gamma_rt_us": "docs/MODEL-dsv2.md:157",
                    "rate_bytes_per_us": "fleet docs/STATUS.md:42 @3fb67a6, measured streamed rate"}
    return m


@dataclass
class Lever:
    stage: str
    name: str          # the lever's patch (examples/dsv2lite/levers/<name>.py), or the planned name
    per_instance_us: float
    basis: str         # BYTES / PHASE
    formula: str
    risk: str


def levers(m: dict) -> list[Lever]:
    """One row per stage-lever; per-instance savings from the measured inputs and OPS-dsv2 geometry."""
    rate, bar, bp = m["rate_bytes_per_us"], m["barrier_us"], m["bperm_us"]
    t = lambda nbytes: nbytes / rate
    term = {   # never-reduced terminal group per layer: WGs x bytes (OPS-dsv2.md lines quoted)
        "qkv": (296 * 2 * 4 * 2 * 2048 * 2, "296 WG x (q + kv_a) x 4 waves x 2 rows x 4096 B (OPS:51-52)"),
        "o_proj": (296 * 16384, "296 WG x 16,384 B = 4 waves x 8 rows x 512 B; issued 45,056 - useful 28,672 (OPS:86)"),
        "merge": (256 * 8192, "256 WG x 8 rows x 1 KiB; issued 16,384 - useful 8,192 (OPS:82)"),
        "router": (296 * 8192, "296 WG x 2 rows x 4 KiB; issued 16 KiB - useful 8 KiB (OPS:90)"),
        "gate_up": (296 * 32768, "296 WG x 4 waves x one 8 KiB pair; issued 344,064 - useful 311,296 (OPS:95)"),
        "down": (296 * 22528, "296 WG x 4 waves x 2 rows x 2,816 B; distinct 180,224 - useful 157,696 (OPS:99-100)"),
        "lm_head": (296 * 32768, "296 WG x 4 waves x 2 rows x 4096 B, <2,4> (OPS:106)"),
    }
    norm = m["gamma_rt_us"] + 7 * bar
    norm_f = f"gamma RT {m['gamma_rt_us']} us hidden + 10 -> 3 barriers x {bar * 1000:.0f} ns"
    tree = 7 * bar
    tree_f = f"pick tree 9 -> 2 barriers x {bar * 1000:.0f} ns, same pairings"
    term_risk = "re-reads the row cnt-1 the WG just loaded: L2 hits would leave only issue time (0 .. this bound)"
    rank_us = 12 * bp + m["lds_probe_us"]
    L = [
        Lever("in_norm", "norm_tree_all", norm, "PHASE", norm_f, "AGPR +40 for norm_tree_all, 0 scratch"),
        Lever("qkv", "gemv_no_terminal_all", t(term["qkv"][0]), "BYTES", term["qkv"][1], term_risk),
        Lever("prep", "attn_prologue_live", 2 * m["gamma_rt_us"] + tree, "PHASE",
              f"2 of its 3 removed dependent RTs x {m['gamma_rt_us']} us (one of kv_post's may hide behind the q_absorb "
              f"wait) + 10 -> 3 tree barriers x {bar*1000:.0f} ns (OPS:65-72)", "AGPR +14, 0 scratch"),
        Lever("attn", "none passing (attn_cache_double_buffer REJECTED)", 0.0, "-",
              "double-buffered cache batches: +160 B/lane scratch in 3 implementations (records/attn_cache_double_buffer)",
              "no register-neutral lever found yet"),
        Lever("reduce", "gemv_no_terminal_all", t(term["merge"][0]), "BYTES", term["merge"][1], term_risk),
        Lever("o_proj", "gemv_no_terminal_all", t(term["o_proj"][0]), "BYTES", term["o_proj"][1], term_risk),
        Lever("router", "norm_tree_all + gemv_no_terminal_all", norm + t(term["router"][0]), "PHASE+BYTES",
              norm_f + "; " + term["router"][1], term_risk),
        Lever("topk", "topk_rank", m["routing_us"] - rank_us, "PHASE",
              f"{m['routing_bperms']} serial bpermutes ({m['routing_us']} us stamped) -> softmax max+sum 12 x "
              f"{bp*1000:.0f} ns + 64 independent readlanes (priced at the {m['lds_probe_us']*1000:.1f} ns LDS probe)",
              "host proof: fleet's ids and weights bit for bit on 6,000 NaN-free inputs"),
        Lever("moe_up", "gemv_no_terminal_all", t(term["gate_up"][0]), "BYTES", term["gate_up"][1], term_risk),
        Lever("moe_down", "gemv_no_terminal_all", t(term["down"][0]), "BYTES", term["down"][1], term_risk),
        Lever("lm_head", "gemv_no_terminal_all + norm_tree_all + argmax_tree", t(term["lm_head"][0]) + norm + tree,
              "BYTES+PHASE", term["lm_head"][1] + "; final norm: " + norm_f + "; candidate pick: " + tree_f, term_risk),
        Lever("argmax", "argmax_tree", tree, "PHASE", tree_f + " (OPS:107)", "host proof incl. ties / NaN / inf"),
        # layer 0 only (counted in the prologue row): the dense post-norm and the dense gate_up's terminal pair
        Lever("_dense_post_norm", "norm_tree_all", norm, "PHASE", norm_f, ""),
        Lever("_dense_gate_up", "gemv_no_terminal_all", t(term["gate_up"][0]), "BYTES", term["gate_up"][1], term_risk),
    ]
    return L


def status(lever: str) -> str:
    """PASS / REJECTED per patch named in a lever string, from results/levers/<patch>.json on this branch."""
    out = []
    for name in re.findall(r"[a-z_]+", lever.split("(")[0]):
        rec = paths.ROOT / f"examples/dsv2lite/results/levers/{name}.json"
        if name in ("none", "passing"):
            continue
        out.append(f"{name}: {json.loads(rec.read_text()).get('verdict', 'PASS')}" if rec.exists() else f"{name}: not on this branch")
    return "; ".join(out) or "REJECTED (see lever)"


def stage_of(layer: int, phase: str) -> str | None:
    if phase in ("embed",) or layer == 0:
        return "prologue"
    if phase in ("lm_head", "argmax"):
        return phase
    if phase == "qabs":
        return None
    if layer == 26 and phase != "qkv":
        return "tail"
    return {"qkv": "qkv", "attn": "attn", "merge": "reduce", "oproj": "o_proj", "router": "router",
            "gate_up": "moe_up", "down": "moe_down"}[phase]


SPLIT = {"qkv": "in_norm", "attn": "prep", "gate_up": "topk"}   # staging + prologue of these phases is its own stage
ORDER = ["prologue", "in_norm", "qkv", "prep", "attn", "reduce", "o_proj", "router", "topk", "moe_up", "moe_down",
         "lm_head", "argmax", "tail"]


def donor_ratios() -> dict:
    """(staging + prologue) / (staging + prologue + stream) per phase, from the stamped 09-27b trace [E]."""
    old = data_sets() / "results_2026-09-27b"
    donor, _ = ld.ledger(ld.load_etx(old / "etx_plan.json", old / "etx_trace.bin", old / "etx_pro.bin"), 0)
    ratio = {}
    for ph, v in aggregate(donor).items():
        rest = v["staging"] + v["prologue"] + v["stream"]
        ratio[ph] = (v["staging"] + v["prologue"]) / rest
    return ratio


def measured(plan: Path, trace: Path, stamps: Path | None = None) -> tuple[dict, dict, float]:
    """us per stage of one traced token, instances per stage, and the span. With stamps (legacy prologue or STAMPS2),
    in_norm / prep / topk are the stamped staging + prologue [M]; without, the donor ratios split the stream [E]."""
    rows, span = ld.ledger(ld.load_etx(plan, trace, stamps), 0)
    ratio = None if stamps else donor_ratios()
    us, n = defaultdict(float), defaultdict(set)
    for layer, ph, c in rows:
        st = stage_of(layer, ph)
        total = sum(c.values())
        if st in ("qkv", "attn", "moe_up") and ph in SPLIT:
            pre = c["staging"] + c["prologue"] if stamps else c["stream"] * ratio[ph]
            us[SPLIT[ph]] += pre
            n[SPLIT[ph]].add(layer)
            total -= pre
        us[st] += total
        n[st].add(layer)
    return dict(us), {k: len(v) for k, v in n.items()}, span


def current() -> tuple[dict, dict, float]:
    """The stride-32 trace (x1s), unstamped: the account every expected value starts from."""
    pinned_parser()
    new = data_sets() / "results_2026-09-27d/sess2"
    return measured(new / "x1s_plan.json", new / "x1s_trace.bin")


def _write_atomic(path: Path, text: str) -> None:
    """Write a temporary file in the same directory, then os.replace it: a reader sees the old or the new file, never a part."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.chmod(tmp, 0o644)              # mkstemp makes 0600; keep the usual file mode
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_table(out: Path | None = None) -> int:
    """Write stage_table.{json,md} to out (default OUT, the committed files)."""
    m = measured_inputs()
    cur, count, span = current()
    assert abs(sum(cur.values()) - span) < 1e-6, (sum(cur.values()), span)
    L = {lv.stage: lv for lv in levers(m)}
    # Layer 0 and layer 26's tail carry the same per-instance levers as their core stages.
    l0 = ["in_norm", "qkv", "prep", "attn", "reduce", "o_proj"]
    L["prologue"] = Lever("prologue", "layer-0 instances of " + ", ".join(l0) + " + dense post-norm + dense gate_up",
                          sum(L[s].per_instance_us for s in l0) + L["_dense_post_norm"].per_instance_us
                          + L["_dense_gate_up"].per_instance_us, "as rows",
                          "one instance of each layer-0 stage lever + norm_tree_all on the dense post-norm + "
                          "gemv_no_terminal_all on the dense gate_up", "as the rows it repeats")
    tail = ["prep", "attn", "reduce", "o_proj", "router", "topk", "moe_up", "moe_down"]
    L["tail"] = Lever("tail", "layer-26 instances of " + ", ".join(tail), sum(L[s].per_instance_us for s in tail),
                      "as rows", "one instance of each post-qkv stage lever", "as the rows it repeats")
    inst = {s: (1 if s in ("prologue", "tail", "lm_head", "argmax") else count[s]) for s in ORDER}
    rows = []
    for s in ORDER:
        save = inst[s] * L[s].per_instance_us
        rows.append(dict(stage=s, instances=inst[s], current_us=cur[s], expected_us=cur[s] - save, delta_us=-save,
                         lever=L[s].name, basis=L[s].basis, formula=L[s].formula, risk=L[s].risk,
                         per_instance_us=L[s].per_instance_us,
                         status=status(L[s].name) if s not in ("prologue", "tail") else "as the rows it repeats"))
    saving = -sum(r["delta_us"] for r in rows)
    # SIM: shrink each task body by its stage's per-instance saving (in_norm+qkv on qkv tasks, prep+attn on attn tasks,
    # topk+moe_up on gate_up tasks, the prologue/tail levers on their layer's tasks), replay, compare spans.
    rp = default_replay()
    per_task = {}
    for tid, task in rp.tasks.items():
        name = task.name
        mt = re.match(r"([a-z_]+?)(?:_(\d+))?$", name)
        ph, layer = mt[1], int(mt[2]) if mt[2] else -1
        cut = 0.0
        if ph == "qkv":
            cut = L["in_norm"].per_instance_us + L["qkv"].per_instance_us
        elif ph == "attn":
            cut = L["prep"].per_instance_us + L["attn"].per_instance_us
        elif ph == "merge":
            cut = L["reduce"].per_instance_us
        elif ph == "oproj":
            cut = L["o_proj"].per_instance_us
        elif ph in ("router", "norm"):
            cut = L["router"].per_instance_us if ph == "router" else L["in_norm"].per_instance_us
        elif ph == "gate_up":
            cut = L["topk"].per_instance_us + L["moe_up"].per_instance_us
        elif ph == "down":
            cut = L["moe_down"].per_instance_us
        elif ph in ("lm_head", "argmax"):
            cut = L[ph].per_instance_us
        per_task[tid] = cut
    base_sim = rp.run()
    what_if = rp.run(lambda name, tid: per_task[tid])
    sim_saving = base_sim - what_if
    summary = dict(
        traced_token_us=span, untraced_device_us=UNTRACED_DEVICE_US,
        arithmetic=dict(saving_us=saving, step_us=UNTRACED_DEVICE_US - saving,
                        tok_s=1e6 / (UNTRACED_DEVICE_US - saving)),
        simulated=dict(replay_base_us=base_sim, replay_what_if_us=what_if, saving_us=sim_saving,
                       step_us=UNTRACED_DEVICE_US - sim_saving, tok_s=1e6 / (UNTRACED_DEVICE_US - sim_saving)),
        limit=dict(step_us=LIMIT_US, tok_s=1e6 / LIMIT_US),
        checkpoint_300=dict(step_us=1e6 / 300))
    out = Path(out) if out is not None else OUT
    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(out / "stage_table.json", json.dumps(dict(inputs=m, rows=rows, summary=summary), indent=2) + "\n")
    md = ["# ETX DeepSeek: per-stage account, expected values and the step they add to", "",
          "Generated by `python -m examples.dsv2lite.stage_table` (do not edit). Current = traced stride-32 token "
          f"(x1s, {span:.2f} us; [M] except the staging/prologue split [E]). Device step now {UNTRACED_DEVICE_US} us "
          f"= {1e6/UNTRACED_DEVICE_US:.1f} tok/s; byte limit {LIMIT_US:.1f} us = {1e6/LIMIT_US:.0f} tok/s.", "",
          "| stage | n | current us | expected us | delta us | lever | local verification | basis | per instance | formula | risk |",
          "|---|---:|---:|---:|---:|---|---|---|---:|---|---|"]
    for r in rows:
        md.append(f"| {r['stage']} | {r['instances']} | {r['current_us']:.2f} | {r['expected_us']:.2f} | "
                  f"{r['delta_us']:.2f} | {r['lever']} | {r['status']} | {r['basis']} | {r['per_instance_us']:.3f} | {r['formula']} | {r['risk']} |")
    a, s = summary["arithmetic"], summary["simulated"]
    md += [f"| **sum** | | **{span:.2f}** | **{span - saving:.2f}** | **{-saving:.2f}** | | | | | | |", "",
           f"**Summary, arithmetic (rows add):** {UNTRACED_DEVICE_US} - {saving:.1f} = **{a['step_us']:.1f} us/token = "
           f"{a['tok_s']:.1f} tok/s**.",
           f"**Summary, simulated plan (replay what-if):** replay {s['replay_base_us']:.1f} -> {s['replay_what_if_us']:.1f} us, "
           f"saving {s['saving_us']:.1f}; {UNTRACED_DEVICE_US} - {s['saving_us']:.1f} = **{s['step_us']:.1f} us/token = "
           f"{s['tok_s']:.1f} tok/s**.",
           f"**Against the limit:** {a['step_us']:.1f} us is {a['step_us'] - LIMIT_US:.1f} us above the byte limit "
           f"({LIMIT_US:.1f} us); these levers remove {saving:.1f} of the {UNTRACED_DEVICE_US - LIMIT_US:.1f} us "
           f"non-byte pool ({100 * saving / (UNTRACED_DEVICE_US - LIMIT_US):.1f}%).", "",
           "Inputs (read from the files named):", ""]
    md += [f"- `{k}` = {v} ({m['sources'][k]})" for k, v in m.items() if k != "sources"]
    md += ["", "BYTES rows are upper bounds: the removed loads re-read a row the workgroup has just loaded, so if they "
           "hit in L2 only their issue time is saved. Every lever is default-off; none is measured on a GPU yet."]
    _write_atomic(out / "stage_table.md", "\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, help="write stage_table.{json,md} here instead of the committed files")
    return write_table(out=parser.parse_args(argv).out)


if __name__ == "__main__":
    raise SystemExit(main())
