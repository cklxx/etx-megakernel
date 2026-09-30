#!/usr/bin/env python3
"""The phase cost model of a compiled plan (docs/MODEL-vllm-kernels.md section 19), replacing the per-task /
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

import argparse
import json
import re
from pathlib import Path

from etx.sim import phase
from etx.sim.xcd import Cost, Hw, simulate

ETX_US = 7.5            # fitted: ETX's per-phase-width cost on top of the hardware floor, paid in both forms
FUSED_US = 1.8          # fitted: what the megakernel costs on top of that, over the same tile launched as its own
                        # kernel. It is positive: build 3 measures 5.895 ms fused and 5.660 unfused, and the qknr
                        # build 5.875 / 5.623, although an unfused launch also pays a 1.61 us kernel boundary. One
                        # kernel with 388 task types, 128 VGPRs and 10 spills is not free to dispatch into.
ROOT = Path(__file__).resolve().parents[2]


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
            variant: dict | None = None, fused: bool = True, fused_us: float = FUSED_US):
    """Cost of every task of `plan`: the phase floor, ETX's per-task cost, and the bytes this task's phase moves
    divided by the tasks the plan gives it."""
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
        c = Cost(floor_us=phase.FLOOR_US * w, etx_us=(etx_us + (fused_us if fused else 0.0)) * w, stream_bytes=per,
                 cu_rate=phase.cu_rate(mapping), agg_bw=phase.agg_bw(per, mapping) if per > 0 else 0.0)
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
            variant: dict | None = None, fused: bool = True, **kw) -> float:
    plan = json.load(open(plan_path))
    rows, st = simulate(plan, cost_fn(plan, cfg, mapping, etx_us, variant=variant, fused=fused), hw(**kw))
    return st["makespan_us"] / 1e3


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
    ap.add_argument("--config", default=str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"))
    ap.add_argument("--mapping", default="contig", choices=["contig", "rows"])
    ap.add_argument("--etx-us", type=float, default=ETX_US)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    print(f"{step_us(a.plan, cfg, a.mapping, a.etx_us):.3f} ms   (mapping {a.mapping}, etx {a.etx_us} us, "
          f"floor {phase.FLOOR_US} us, HBM {phase.BW_HBM / 1e6:.2f} TB/s)")



if __name__ == "__main__":
    main()
