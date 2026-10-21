#!/usr/bin/env python3
"""Qwen3-30B-A3B decode with vLLM's kernels fused by ETX: the per-token time predicted from vLLM's own kernel trace and
the two ETX costs that decide it (docs/qwen/MODEL-moe.md).

  python3 examples/vllm_llm/moe_predict.py results_2026-09-27/vlnext/moe/layer.json [--h 1.2 --p 0.5]

Per layer, vLLM runs 17 kernels back to back; each costs F (the kernel boundary of a HIP graph, 1.6 us measured) plus
its body b = duration - F. In the ETX graph the same kernels run as tasks: a task costs its body, plus p (what the
imported form adds per task: descriptor and argument-block round trips, the wrapper), plus h (the event hand-off) unless
it is chained behind the previous task in the same workgroup. Two launches do no model work and leave the critical path:
the fill of num_tokens_post_padded (a constant at batch 1) and the copy of the MoE output (aliased instead).

The trace inflates the durations; every duration is scaled so the layer matches the untraced 4.898 ms/token.
"""
from __future__ import annotations

import argparse
import json

F = 1.6            # us, kernel boundary inside a HIP graph (bench/sync/sync_bench.hip, 2026-09-27)
VLLM_MS = 4.898    # vLLM batch 1, AITER off, 1024-token context (2026-09-27)
LAYERS = 48
DROP = ("FillFunctor", "copyBuffer")
# positions (in vLLM's layer order) that run behind the previous task in the same workgroup (ETX_VL_CHAIN): moe_sum ->
# fused_add_rms_norm, and the q/k norm -> rope -> cache run of the attention block
CHAINED = ("triton_red_fused_fused_add_rms_norm_1", "triton_poi_fused_3", "reshape_and_cache")


def predict(kernels, scale, h, p, tail_ms):
    kept = [k for k in kernels if not any(d in k["name"] for d in DROP)]
    bodies = sum(k["mean_us"] * scale - F for k in kept)
    handoffs = sum(0 if any(c in k["name"] for c in CHAINED) else 1 for k in kept)
    layer = bodies + handoffs * h + len(kept) * p
    return layer, layer * LAYERS / 1e3 + tail_ms, len(kept), handoffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer_json")
    ap.add_argument("--h", type=float, nargs="*", default=[1.0, 1.2, 1.6, 2.0, 2.5, 3.0])
    ap.add_argument("--p", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.35])
    a = ap.parse_args()
    lj = json.load(open(a.layer_json))
    ks = lj["kernels"]
    traced_layer = sum(k["mean_us"] for k in ks)
    # the rest of the token (embedding, final norm, lm_head 622 MB, sampling) from the same trace is not in layer.json;
    # take it as what the untraced token leaves after 48 layers scaled alike
    scale = 0.92          # untraced / traced busy time per token, 4.898 / 5.313 (vllm_profile.py analyze)
    tail_ms = VLLM_MS - LAYERS * traced_layer * scale / 1e3
    print(f"vLLM layer {traced_layer:.1f} us traced, {traced_layer * scale:.1f} us untraced; rest of the token {tail_ms:.3f} ms")
    _, _, n, hn = predict(ks, scale, 0, 0, tail_ms)
    print(f"ETX: {n} tasks per layer on the critical path, {hn} hand-offs (vLLM: {len(ks)} launches)")
    print("ms/token (vLLM %.3f); rows h = hand-off us, columns p = per-task import cost us" % VLLM_MS)
    print("   h \\ p " + "".join(f"{p:9.2f}" for p in a.p))
    for h in a.h:
        print(f"{h:8.2f} " + "".join(f"{predict(ks, scale, h, p, tail_ms)[1]:9.3f}" for p in a.p))
    # break-even: hn * h + n * p = vLLM layer - kept bodies
    kept_bodies = sum(k["mean_us"] * scale - F for k in ks if not any(d in k["name"] for d in DROP))
    budget = traced_layer * scale - kept_bodies
    print(f"break-even: {hn} h + {n} p = {budget:.1f} us per layer (e.g. p = 0.5 -> h < {(budget - n * 0.5) / hn:.2f} us; "
          f"h = 1.2 -> p < {(budget - hn * 1.2) / n:.2f} us)")


if __name__ == "__main__":
    main()
