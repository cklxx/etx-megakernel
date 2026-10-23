#!/usr/bin/env python3
"""The phase cost model of a compiled plan (docs/qwen/MODEL-vllm-kernels.md section 19), replacing the per-task /
per-event model of section 10.

The old model charged a tax per task and counted events, so folding operations and cutting events predicted large
gains; the hardware paid 0.7% of them (build 2 5.934 ms -> build 3 5.895). What the measurements say instead is
that cost follows how *wide* a phase is, not how many phases there are: build 3 runs 8 phases a layer and the qknr
build 13, and they measure 5.895 and 5.875 ms.

A task of an n-task phase costs

    (floor_us + etx_us + fused_us) * min(1, n / 296)
                        floor_us is hardware: 296 workgroups reading 896 B each still take 3.2 us (hw_bench
                        stream, no events, no ETX). etx_us and fused_us are ETX's own share, fitted to the eight
                        measured builds -- fused_us only in the megakernel, and it is positive, which is the
                        opposite of what fusing was supposed to buy
  + chain_us            the kernel's own dependent work where it is known to be large (attention partitions)
  + bytes / rate        rate = min(what one workgroup pulls alone, its share of the bandwidth the device sustains
                        for transfers of that size -- 2.1 TB/s at 28 KB per workgroup, 3.5 TB/s at 1 MB)

The bytes of a phase come from the model's shapes and are divided by the number of tasks the plan actually gives
that phase, so a plan that spreads 0.5 MB over 304 tasks is charged for the width it asked for.

  python3 -m examples.vllm_llm.pmodel PLAN.json [--mapping contig|rows]
  python3 -m examples.vllm_llm.validate                      the eight builds this is fitted to
"""
from __future__ import annotations
from . import paths

import argparse
import json
import re
from pathlib import Path

from etx.sim import phase
from etx.sim.xcd import Cost, Hw, simulate

PROFILES = ("2026-09-30", "2026-10-08")
STREAM_ROLES = {"qkv", "o_proj", "router", "moe_up", "moe_down", "lm_head"}
# Round11 critical bodies, from another build (docs/qwen/FLOORS-2026-10-08.md).
NARROW_BODY_US = {"prep": 6.04, "reduce": 5.58, "topk": 8.35, "prologue": 11.3,
                  "in_norm": 5.6, "argmax": 10.1}
# Pre-session host excess: results_tiles/ab{1,2,3}_all_x.log; VLLM-LADDER-2026-10-01.md.
HOST_US = {"fused": 140.0, "unfused": 36.0}

ETX_US = 7.5            # fitted: ETX's per-phase-width cost on top of the hardware floor, paid in both forms
FUSED_US = 1.8          # fitted: what the megakernel costs on top of that, over the same tile launched as its own
                        # kernel. It is positive: build 3 measures 5.895 ms fused and 5.660 unfused, and the qknr
                        # build 5.875 / 5.623, although an unfused launch also pays a 1.61 us kernel boundary. One
                        # kernel with 388 task types, 128 VGPRs and 10 spills is not free to dispatch into.

# ETX_VL_COH (docs/qwen/MODEL-vllm-kernels.md section 20): DSL tiles read activations coherently and their acquire no
# longer invalidates the XCD's L2. What that removes from the fitted per-phase constants cannot be derived from
# first principles -- the fit bundles acquire, argument path, release and wave start -- so the model carries it as
# a RANGE, not a point: the measured pieces bound the inv + write-back share of (etx_us + fused_us) at roughly
# 0.5 us (idle inv 0.19 + fence 0.18, little contention) to 1.5 us (loaded inv 0.63 + the wbl2 arrive on a loaded
# device, both ways). The second, separately switchable effect is the touch hook finally surviving into the body:
# without COH the acquire's buffer_inv sc1 threw the touched lines away (why section 18 delivered ~0 of its
# predicted 0.34 ms); with COH the dense GEMV phases' touched first trips overlap their wait.
COH_SAVE_LO, COH_SAVE_HI = 0.5, 1.5
# ETX_VL_ARGPF + ETX_VL_BAR2 on top of COH: the argument path is bounded by the measured 0.24 ms spread between
# arg-path implementations (2026-09-28b, all-or-nothing per step), the dropped loop barriers by one to two
# s_swbar (0.106-0.282 us) per phase. Applied as a per-step delta range, clamped at 4.6 ms (section 20).
ALL_SAVE_LO_MS, ALL_SAVE_HI_MS = 0.10, 0.35
ALL_FLOOR_MS = 4.6
COH_ROLES = {"qkv", "o_proj", "router", "moe_up", "moe_down", "prep", "lm_head",
             "in_norm", "post_norm", "topk", "moe_sum"}          # DSL tiles; attn / reduce / chains stay imported
TOUCH_ROLES = {"qkv", "o_proj", "router", "lm_head"}             # dense dsl_gemv{,_fold}: the only forms with a touch
TOUCH_BYTES = 60 * 1024                                          # per workgroup, one 64-row first trip (section 18)


def role_bytes(cfg: dict, ctx: int = 1024) -> dict:
    """Bytes one layer's phase reads, from the model's shapes (bf16). lm_head and embed are per step."""
    h, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or h // nh
    mi, topk = cfg["moe_intermediate_size"], cfg["num_experts_per_tok"]
    return {
        "qkv": (nh + 2 * nkv) * hd * h * 2,
        "o_proj": h * nh * hd * 2,
        "router": cfg["num_experts"] * h * 2,
        "moe_up": topk * 2 * mi * h * 2,
        "moe_down": topk * h * mi * 2,
        "attn": 2 * nkv * hd * ctx * 2,          # K and V of the context, read by the partitions
        "reduce": 0, "prep": 0, "in_norm": 0, "post_norm": 0, "topk": 0, "moe_act": 0, "moe_sum": 0,
        "lm_head": cfg["vocab_size"] * h * 2,
        "qknr": 0, "rope": 0, "cache": 0, "q_norm": 0, "k_norm": 0,
    }


def _role(name: str) -> str:
    """qkv_12 -> qkv, moe_down_3_x -> moe_down, r1_o_proj_7 -> o_proj, chain_qknr_cache_4 -> qknr_cache."""
    s = re.sub(r"^r\d+_", "", name)
    s = re.sub(r"_x$", "", s)
    s = re.sub(r"_\d+$", "", s)
    return s


def cost_fn(plan: dict, cfg: dict, mapping: str = "contig", etx_us: float = ETX_US, ctx: int = 1024,
            variant: dict | None = None, fused: bool = True, fused_us: float = FUSED_US,
            coh_save_us: float = 0.0, coh_touch: bool = False, profile: str = "2026-09-30"):
    """Cost of every task of `plan`: the phase floor, ETX's per-task cost, and the bytes this task's phase moves
    divided by the tasks the plan gives it."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    names = {t["id"]: n for n, t in plan["types"].items()}
    ntask: dict[str, int] = {}
    for d in plan["descs"]:
        n = names[d["type"]]
        ntask[n] = ntask.get(n, 0) + 1
    rb = role_bytes(cfg, ctx)
    variant = variant or {}
    cache: dict[str, Cost] = {}

    def cost(name: str, desc: dict) -> Cost:
        if name in cache:
            return cache[name]
        r = _role(name)
        if r.startswith("chain_"):           # q/k norm + rope + cache run back to back in one task
            rest = r[len("chain_"):]
            total = sum(v for k, v in rb.items() if k in rest.split("_") or f"_{k}_" in f"_{rest}_")
        else:
            total = rb.get(r, 0.0)
        nt = max(ntask.get(name, 1), 1)
        per = total / nt
        # Both fixed costs scale with how wide the phase is, not with how many phases there are. The floor was
        # measured with 296 workgroups starting and hitting memory together, and the builds say the same of ETX's
        # own share: build 3 runs 8 phases a layer and the qknr build 13, and they measure 5.895 and 5.875 ms --
        # the five extra narrow phases cost nothing, while a 304-task phase costs about 10 us however few bytes it
        # moves. So a task of an n-task phase pays (floor + etx) * n / 296.
        w = min(1.0, nt / phase.N_REF)
        e = etx_us + (fused_us if fused else 0.0)
        if coh_save_us > 0.0 and fused and r in COH_ROLES:      # ETX_VL_COH: the acquire inv and (for all-coherent
            e = max(e - coh_save_us, 0.0)                        # producers) the wbl2 arrive are gone
        if profile == "2026-10-08":
            e = 0.0
        c = Cost(floor_us=phase.FLOOR_US * w, etx_us=e * w, stream_bytes=per,
                 cu_rate=phase.cu_rate(mapping), agg_bw=phase.agg_bw(per, mapping) if per > 0 else 0.0)
        if profile == "2026-10-08":
            if r in STREAM_ROLES and per > 0:
                c = Cost(**{**c.__dict__, "agg_bw": phase.BW_HBM, "cu_rate": 1e9})
            if r in NARROW_BODY_US:
                c = Cost(**{**c.__dict__, "chain_us": NARROW_BODY_US[r]})
        if profile == "2026-09-30" and coh_touch and fused and r in TOUCH_ROLES and per > 0:  # the touch survives only without the inv
            c = Cost(**{**c.__dict__, "touch_bytes": min(TOUCH_BYTES, per)})
        if r == "attn":                       # the attention partitions carry their own dependent chain
            c = Cost(**{**c.__dict__, "chain_us": 1.9})
        for key, f in variant.items():
            if isinstance(f, (int, float)):
                c = Cost(**{**c.__dict__, key: f})
        cache[name] = c
        return c
    return cost


def hw(handoff_domain: float = 0.4, handoff_device: float = 1.8) -> Hw:
    return Hw(bw_max=phase.BW_HBM, n_half=0.0, handoff_us={"DOMAIN": handoff_domain, "DEVICE": handoff_device},
              lat_us=0.2)


def step_us(plan_path: str | Path, cfg: dict, mapping: str = "contig", etx_us: float = ETX_US,
            variant: dict | None = None, fused: bool = True, coh_save_us: float = 0.0, coh_touch: bool = False,
            profile: str = "2026-09-30", **kw) -> float:
    plan = json.load(open(plan_path))
    rows, st = simulate(plan, cost_fn(plan, cfg, mapping, etx_us, variant=variant, fused=fused,
                                      coh_save_us=coh_save_us, coh_touch=coh_touch, profile=profile), hw(**kw))
    return st["makespan_us"] / 1e3


def wall_ms(plan: str | Path, cfg: dict, fused: bool, profile: str) -> float:
    """Step plus pre-session host excess; step_us returns milliseconds despite its legacy name."""
    handoffs = {} if fused else {"handoff_domain": 1.61, "handoff_device": 1.61}
    return step_us(plan, cfg, fused=fused, profile=profile, **handoffs) + HOST_US[
        "fused" if fused else "unfused"] / 1e3


def predict_coh(plan_path: str | Path, cfg: dict, mapping: str = "contig") -> dict:
    """The build-4 (ETX_VL_COH) prediction as a range (docs/qwen/MODEL-vllm-kernels.md section 20): the overhead share
    alone at its low and high estimates, then with the touch landing."""
    out = {"base": step_us(plan_path, cfg, mapping)}
    out["coh_lo"] = step_us(plan_path, cfg, mapping, coh_save_us=COH_SAVE_LO)
    out["coh_hi"] = step_us(plan_path, cfg, mapping, coh_save_us=COH_SAVE_HI)
    out["coh_touch_lo"] = step_us(plan_path, cfg, mapping, coh_save_us=COH_SAVE_LO, coh_touch=True)
    out["coh_touch_hi"] = step_us(plan_path, cfg, mapping, coh_save_us=COH_SAVE_HI, coh_touch=True)
    return out


def predict_all(plan_path: str | Path, cfg: dict, mapping: str = "contig") -> dict:
    """Build 5 (COH + ARGPF + BAR2): the COH range shifted by the bounded argument-path and barrier savings;
    never promised below ALL_FLOOR_MS."""
    r = predict_coh(plan_path, cfg, mapping)
    for k in ("coh_lo", "coh_hi", "coh_touch_lo", "coh_touch_hi"):
        r["all_" + k[4:]] = max(ALL_FLOOR_MS, r[k] - (ALL_SAVE_HI_MS if k.endswith("lo") else ALL_SAVE_LO_MS))
    return r


def width_sweep(cfg: dict, mapping: str = "contig", etx_us: float = ETX_US, fused_us: float = FUSED_US,
                layers: int | None = None) -> list:
    """What each phase would cost at every width, under the fitted model. The plan fixes these widths today; the
    model says they are the remaining lever, so this is what a search should be over."""
    n_layers = layers if layers is not None else 48
    rb = role_bytes(cfg)
    cur = {"qkv": 304, "o_proj": 304, "router": 304, "moe_up": 192, "moe_down": 256}
    out = []
    for role, now in cur.items():
        total = rb[role]

        def at(n):
            per = total / n
            return (phase.FLOOR_US + etx_us + fused_us) * min(1.0, n / phase.N_REF) + \
                per / min(phase.cu_rate(mapping), phase.agg_bw(per, mapping) / n)
        best = min(range(1, 305), key=at)
        out.append((role, total, now, at(now), best, at(best), (at(now) - at(best)) * n_layers / 1e3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--config", default=str(paths.ROOT / "examples/llm/configs/qwen3-30b-a3b.json"))
    ap.add_argument("--mapping", default="contig", choices=["contig", "rows"])
    ap.add_argument("--profile", choices=PROFILES, default=PROFILES[0])
    ap.add_argument("--unfused", action="store_true", help="separate launches with 1.61 us handoffs")
    ap.add_argument("--etx-us", type=float, default=ETX_US)
    ap.add_argument("--coh", action="store_true", help="the ETX_VL_COH range (section 20)")
    ap.add_argument("--all", dest="all_", action="store_true", help="the COH + ARGPF + BAR2 range (build 5)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.all_:
        r = predict_all(a.plan, cfg, a.mapping)
        print(f"base {r['base']:.3f} ms; all overhead only {r['all_hi']:.3f}-{r['all_lo']:.3f}; "
              f"with touch {r['all_touch_hi']:.3f}-{r['all_touch_lo']:.3f}   (coh save {COH_SAVE_LO}-{COH_SAVE_HI} us, "
              f"argpf+bar2 {ALL_SAVE_LO_MS}-{ALL_SAVE_HI_MS} ms, floor {ALL_FLOOR_MS})")
        return
    if a.coh:
        r = predict_coh(a.plan, cfg, a.mapping)
        print(f"base {r['base']:.3f} ms; coh overhead only {r['coh_hi']:.3f}-{r['coh_lo']:.3f}; "
              f"with touch {r['coh_touch_hi']:.3f}-{r['coh_touch_lo']:.3f}   (save {COH_SAVE_LO}-{COH_SAVE_HI} us)")
        return
    handoffs = {} if not a.unfused else {"handoff_domain": 1.61, "handoff_device": 1.61}
    ms = step_us(a.plan, cfg, a.mapping, a.etx_us, fused=not a.unfused, profile=a.profile, **handoffs)
    print(f"{ms:.3f} ms   (mapping {a.mapping}, etx {a.etx_us} us, "
          f"floor {phase.FLOOR_US} us, HBM {phase.BW_HBM / 1e6:.2f} TB/s)")

    if a.profile == "2026-10-08":
        print(f"wall {ms + HOST_US['unfused' if a.unfused else 'fused'] / 1e3:.3f} ms")


if __name__ == "__main__":
    main()
