"""Byte floors of one DeepSeek-V2-Lite decode step at batch B (docs/deepseek/BATCH-2026-10-14.md section 2).

Two ways to serve B requests in one step:
  replicated  each request reads every weight itself (ETX today: ETX_DSV2_REQUESTS=B): B x the batch-1 bytes.
  batched     every weight read once for the B columns (idealized column-batched model, not measured vLLM traffic): the dense
              weights, the shared experts and lm_head once, the routed experts once per DISTINCT expert the B requests
              select in a layer, the latent KV cache once per request.
The distinct-expert count E(B) is the uniform-routing expectation 64 * (1 - (1 - 6/64)^B) unless a measured mean is
given (examples/dsv2lite/routing_overlap.py, recorded on the box by the batch session). Floors divide by the
achievable stream rate (CLAUDE.md: about 4.3 TB/s on MI300X). Estimates [E], never measurements.

  python -m examples.dsv2lite.batch_floor [--context 1024] [--union B=E ...]
"""
from __future__ import annotations

import argparse
import json

from examples.dsv2lite.bytes import HIDDEN, MOE_INTER, SHARED, TOPK, B as BF16, per_layer

LAYERS, FIRST_DENSE, ROUTED = 27, 1, 64
RATE_TBS = 4.3
EXPERT = 3 * MOE_INTER * HIDDEN * BF16              # gate + up + down of one expert, bytes


def uniform_union(b: int) -> float:
    """Expected distinct routed experts per layer when b requests each pick TOPK of ROUTED uniformly."""
    return ROUTED * (1 - (1 - TOPK / ROUTED) ** b)


def step_bytes(b: int, context: int = 1024, union: float | None = None) -> dict[str, float]:
    """Bytes of one step serving b requests: replicated and batched (weights once)."""
    p = per_layer(context)
    attn_w = p["qkv"] + p["qabs"] + p["merge"] + p["oproj"]   # every layer
    kv = p["attn"]                                                                 # per layer, per request
    dense_mlp = p["dense_gu"] + p["dense_dn"]
    moe_layers = LAYERS - FIRST_DENSE
    # The first dense layer normalizes but returns before the router GEMV.
    routers = moe_layers * p["router"]
    one = (LAYERS * (attn_w + kv) + routers + FIRST_DENSE * dense_mlp + moe_layers * (TOPK + SHARED) * EXPERT
           + p["lm_head"] + p["embed"])
    e = uniform_union(b) if union is None else union
    batched = (LAYERS * (attn_w + b * kv) + routers + FIRST_DENSE * dense_mlp + moe_layers * (e + SHARED) * EXPERT
               + p["lm_head"] + b * p["embed"])
    return dict(replicated=b * one, batched=batched, union=e, one=one)


def floors(context: int = 1024, unions: dict[int, float] | None = None) -> dict[int, dict]:
    out = {}
    for b in (1, 2, 4, 8):
        s = step_bytes(b, context, (unions or {}).get(b))
        out[b] = dict(union=round(s["union"], 2), replicated_gb=round(s["replicated"] / 1e9, 3),
                      batched_gb=round(s["batched"] / 1e9, 3),
                      replicated_floor_ms=round(s["replicated"] / (RATE_TBS * 1e9), 3),
                      batched_floor_ms=round(s["batched"] / (RATE_TBS * 1e9), 3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--union", action="append", default=[], help="B=E: a measured mean of distinct experts per layer")
    a = ap.parse_args()
    unions = {int(k): float(v) for k, v in (u.split("=") for u in a.union)}
    print(json.dumps(floors(a.context, unions), indent=1))


if __name__ == "__main__":
    main()
