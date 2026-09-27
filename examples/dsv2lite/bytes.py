"""Bytes each DeepSeek-V2-Lite decode task reads from memory, for `etx.tools.timeline --bytes`.

Unique bytes per token (weights once, the latent KV cache once per layer), divided evenly over the tasks of the grid
that reads them; re-reads from L2 / Infinity Cache (every head reading the shared latent cache, every worker reading
the folded x) are not counted, so the numbers add up to the byte floor (about 4.9 GB per token).

  python -m examples.dsv2lite.bytes [--context 1024] > build/dsv2lite/bytes.json
"""
from __future__ import annotations

import argparse
import json

# DeepSeek-Coder-V2-Lite (config.json): hidden 2048, 16 heads, q = nope 128 + rope 64, latent KV 512 + rope 64,
# v 128, 64 routed experts top-6 + 2 shared, expert inter 1408, dense inter 10944 (layer 0), vocab 102400, bf16
HIDDEN, HEADS, NOPE, ROPE, KV_LORA, V = 2048, 16, 128, 64, 512, 128
MOE_INTER, DENSE_INTER, TOPK, SHARED, VOCAB = 1408, 10944, 6, 2, 102400
XCDS, W = 8, 37
B = 2  # bf16


def per_layer(context: int) -> dict[str, float]:
    """grid prefix -> bytes per layer"""
    units = TOPK + SHARED
    return {
        "qkv": (HIDDEN * HEADS * (NOPE + ROPE) + HIDDEN * (KV_LORA + ROPE)) * B,
        "qabs": KV_LORA * HEADS * NOPE * B,                 # W_UK
        "attn": context * (KV_LORA + ROPE) * B,             # latent cache, once
        "merge": KV_LORA * HEADS * V * B,                   # W_UV
        "oproj": HEADS * V * HIDDEN * B,
        "router": HIDDEN * 64 * B,
        "gate_up": units * 2 * MOE_INTER * HIDDEN * B,
        "down": units * MOE_INTER * HIDDEN * B,
        "dense_gu": 2 * DENSE_INTER * HIDDEN * B,
        "dense_dn": DENSE_INTER * HIDDEN * B,
        "lm_head": VOCAB * HIDDEN * B,
        "embed": HIDDEN * B,
    }


TASKS = {"qkv": XCDS * W, "qabs": HEADS * 2, "attn": HEADS * 16, "merge": HEADS * 16, "oproj": XCDS * W, "router": XCDS * W,
         "gate_up": XCDS * W, "down": XCDS * W, "dense_gu": XCDS * W, "dense_dn": XCDS * W, "lm_head": XCDS * W, "embed": 1}


def spec(context: int = 1024, workers_per_xcd: int = W) -> dict[str, float]:
    tasks = dict(TASKS)
    for k in ("qkv", "oproj", "router", "gate_up", "down", "dense_gu", "dense_dn", "lm_head"):
        tasks[k] = XCDS * workers_per_xcd
    return {rf"^(r\d+_)?{k}(_-?\d+)?$": v / tasks[k] for k, v in per_layer(context).items()}   # r<i>_: the interleaved graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=W)
    a = ap.parse_args()
    print(json.dumps(spec(a.context, a.workers), indent=1))


if __name__ == "__main__":
    main()
