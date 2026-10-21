#!/usr/bin/env python3
"""Batching against layer pipelining across XCDs on the build-3 structure, analytic (docs/qwen/MODEL-vllm-kernels.md
section 17): what a batch of B tokens costs on the whole GPU (dense weights once, the union of B tokens' experts,
the per-launch tax as in the simulator) against a pipeline of S stages of 8/S XCDs each, micro-batches of m tokens,
S x m requests in flight, each stage streaming its 48/S layers at the stage's own bandwidth.

  python3 -m examples.vllm_llm.batch_pp [--bw 3.56] [--tax 1.99] [--handoff-premium 0.4] [--f 1.0,0.8,0.6]

Measured inputs: the model sizes (examples/llm/configs/qwen3-30b-a3b.json, bf16), the streaming bandwidth (hw_bench lf:
3.56 TB/s at 192 x 64 KB in flight, 3.93 max). Simulated inputs: the per-step tax of build 3 (sim 3.54 ms - the 1.55 ms
bandwidth floor = 1.99 ms; x1.11 for the realistic figure), the device-wide hand-off premium (sensitivity: hand-offs
0 saves 0.57 ms; the DEVICE -> DOMAIN part taken as 0.4 ms). ASSUMED until hw_bench lfx runs: a stage of n XCDs
pulls f x n/8 of the GPU's bandwidth (f = 1.0, 0.8, 0.6); layers split evenly (lm_head's 0.62 GB ignored in the split).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sizes(cfg: dict) -> dict:
    H, NH, NKV, HD = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    E, K, MI, L, V = cfg["num_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    dense = ((NH + 2 * NKV) * HD * H + NH * HD * H + E * H) * 2          # qkv, o_proj, router (bytes per layer)
    expert = 3 * MI * H * 2                                                # gate, up, down
    return dict(dense=dense, expert=expert, E=E, K=K, L=L, lm_head=V * H * 2)


def step_bytes(s: dict, B: int) -> float:
    """dense once per layer, the union of B tokens' top-k experts (independent uniform routing), lm_head once"""
    experts = s["E"] * (1 - (1 - s["K"] / s["E"]) ** B)
    return s["L"] * (s["dense"] + experts * s["expert"]) + s["lm_head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, default=3.56, help="GPU streaming bandwidth, TB/s (measured, hw_bench lf 192 x 64 KB)")
    ap.add_argument("--tax", type=float, default=1.99, help="build 3's simulated step minus its bandwidth floor, ms")
    ap.add_argument("--handoff-premium", type=float, default=0.4, help="ms of the tax that device-wide hand-offs cost over XCD-local ones")
    ap.add_argument("--f", default="1.0,0.8,0.6", help="assumed per-XCD bandwidth fractions of BW/8")
    ap.add_argument("--realistic", type=float, default=1.11)
    a = ap.parse_args()
    s = sizes(json.load(open(ROOT / "examples/llm/configs/qwen3-30b-a3b.json")))
    bw = a.bw * 1e12 / 1e3                                                 # bytes per ms
    print(f"bytes per token: {step_bytes(s, 1) / 1e9:.2f} GB (dense {s['L'] * s['dense'] / 1e9:.2f}, experts {s['L'] * s['K'] * s['expert'] / 1e9:.2f}, "
          f"lm_head {s['lm_head'] / 1e9:.2f}); floor at {a.bw} TB/s: {step_bytes(s, 1) / bw:.2f} ms; tax {a.tax} ms (x{a.realistic} realistic)")
    fs = [float(x) for x in a.f.split(",")]
    print("\n(a) full-GPU batch of B tokens: step = tax + bytes(B) / BW              (b)-(c) pipeline: S stages of 8/S XCDs, micro-batch m = B / S,")
    print("    tok/s = B / step; latency = step                                    stage step = (tax - premium) / S + bytes(m) / S / (f x BW x (8/S) / 8);")
    print("                                                                        tok/s = m / stage step; latency = S x stage step\n")
    hdr = f"{'B':>3s} {'batch ms':>9s} {'tok/s':>6s}"
    for S in (8, 4, 2):
        for f in fs:
            hdr += f" | PP S={S} f={f:.1f}: {'ms':>6s} {'tok/s':>5s}"
    print(hdr)
    for B in (1, 2, 4, 8, 16):
        step = (a.tax + step_bytes(s, B) / bw) * a.realistic
        line = f"{B:3d} {step:9.2f} {B / step * 1e3:6.0f}"
        for S in (8, 4, 2):
            for f in fs:
                if B % S:
                    line += f" | {'-':>24s}"; continue
                m = B // S
                stage = ((a.tax - a.handoff_premium) / S + step_bytes(s, m) / S / (f * bw / S)) * a.realistic
                line += f" | {'':17s}{S * stage:6.2f} {m / stage * 1e3:5.0f}"
        print(line)
    print("\nPP columns: latency per token (S x stage step) and throughput; '-' = fewer requests than stages (bubbles).")
    print("Under every f, batching with the same B requests in flight gives more tokens per second than the pipeline: the pipeline")
    print("reads the dense weights once per micro-batch instead of once per batch, and only saves the device-wide hand-off premium.")


if __name__ == "__main__":
    main()
