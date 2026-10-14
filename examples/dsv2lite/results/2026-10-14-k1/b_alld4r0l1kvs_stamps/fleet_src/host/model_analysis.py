#!/usr/bin/env python3
"""Byte accounting for DeepSeek-Coder-V2-Lite-Base batch-1 decode on MI300X.

Reproduces the per-token HBM traffic table in docs/design.md §1 straight from
config.json, so the design's numbers are checked rather than asserted.

    python3 src/host/model_analysis.py --config reference/dsv2lite_config.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

BF16 = 2  # bytes per element
# Decimal units throughout: bandwidth is quoted in TB/s (10^12 B/s), so byte
# counts must be decimal for the roofline to be self-consistent. This is what
# docs/design.md §1 uses; binary MiB would inflate every figure by 4.9%.
MB = 1_000_000
GB = 1_000_000_000

# MI300X, from the design doc §9. Peak is the spec number; achievable is AMD's
# own stream microbenchmark (SPX/NPS1, ROCm blog Feb 2025).
HBM_PEAK_TBS = 5.3
HBM_ACHIEVABLE_TBS = 4.0
BF16_PEAK_PFLOPS = 1.3


@dataclass(frozen=True)
class Cfg:
    hidden: int
    layers: int
    heads: int
    kv_lora_rank: int
    qk_nope: int
    qk_rope: int
    v_head: int
    dense_inter: int
    moe_inter: int
    n_routed: int
    n_shared: int
    top_k: int
    vocab: int
    first_k_dense: int

    @classmethod
    def load(cls, path: Path) -> "Cfg":
        c = json.loads(path.read_text())
        return cls(
            hidden=c["hidden_size"],
            layers=c["num_hidden_layers"],
            heads=c["num_attention_heads"],
            kv_lora_rank=c["kv_lora_rank"],
            qk_nope=c["qk_nope_head_dim"],
            qk_rope=c["qk_rope_head_dim"],
            v_head=c["v_head_dim"],
            dense_inter=c["intermediate_size"],
            moe_inter=c["moe_intermediate_size"],
            n_routed=c["n_routed_experts"],
            n_shared=c["n_shared_experts"],
            top_k=c["num_experts_per_tok"],
            vocab=c["vocab_size"],
            first_k_dense=c["first_k_dense_replace"],
        )

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope + self.qk_rope

    @property
    def kv_a_out(self) -> int:
        """kv_a_proj_with_mqa output: compressed KV plus the shared rope key."""
        return self.kv_lora_rank + self.qk_rope

    @property
    def cache_row(self) -> int:
        """One absorbed-MLA cache row: c ‖ k_pe, shared by every head."""
        return self.kv_lora_rank + self.qk_rope


def attention_bytes(c: Cfg) -> dict[str, int]:
    """Weights read once per token by the attention block of one layer."""
    return {
        "q_proj": c.hidden * c.heads * c.q_head_dim * BF16,
        "kv_a_proj_with_mqa": c.hidden * c.kv_a_out * BF16,
        "kv_b_proj": c.kv_lora_rank * c.heads * (c.qk_nope + c.v_head) * BF16,
        "o_proj": c.heads * c.v_head * c.hidden * BF16,
    }


def moe_bytes(c: Cfg) -> dict[str, int]:
    """Weights read per token by one MoE layer: router, top-k routed, shared."""
    one_expert = 3 * c.hidden * c.moe_inter * BF16  # gate, up, down
    return {
        "router_gate": c.hidden * c.n_routed * BF16,
        "routed_experts": c.top_k * one_expert,
        "shared_experts": 3 * c.hidden * (c.moe_inter * c.n_shared) * BF16,
    }


def dense_mlp_bytes(c: Cfg) -> int:
    return 3 * c.hidden * c.dense_inter * BF16


def report(c: Cfg, ctx: int, decode: int) -> None:
    attn = attention_bytes(c)
    moe = moe_bytes(c)
    attn_total = sum(attn.values())
    moe_total = sum(moe.values())
    dense = dense_mlp_bytes(c)
    lm_head = c.vocab * c.hidden * BF16
    embed_row = c.hidden * BF16

    n_moe_layers = c.layers - c.first_k_dense
    layer0 = attn_total + dense
    moe_layer = attn_total + moe_total
    weights = layer0 + n_moe_layers * moe_layer + lm_head + embed_row

    # Absorbed MLA: one cache row per layer per position, shared by all heads.
    seq = ctx + decode
    kv_bytes = c.layers * seq * c.cache_row * BF16
    total = weights + kv_bytes

    print(f"model: hidden={c.hidden} layers={c.layers} (layer<{c.first_k_dense} dense, rest MoE)")
    print(f"       heads={c.heads} q_head_dim={c.q_head_dim} kv_lora={c.kv_lora_rank} "
          f"cache_row={c.cache_row}")
    print(f"       experts {c.n_routed} routed (top-{c.top_k}) + {c.n_shared} shared, "
          f"moe_inter={c.moe_inter}")

    print("\nper-layer attention weights (bf16):")
    for k, v in attn.items():
        print(f"  {k:<22} {v/MB:8.2f} MB")
    print(f"  {'subtotal':<22} {attn_total/MB:8.2f} MB")

    print("\nper-layer MoE weights (bf16), read per token:")
    for k, v in moe.items():
        print(f"  {k:<22} {v/MB:8.2f} MB")
    print(f"  {'subtotal':<22} {moe_total/MB:8.2f} MB")
    print(f"  (one routed expert       {3*c.hidden*c.moe_inter*BF16/MB:8.2f} MB, "
          f"{c.top_k} of {c.n_routed} read)")

    print("\nper-token totals:")
    print(f"  layer 0 (dense MLP)    {layer0/MB:8.2f} MB")
    print(f"  each MoE layer         {moe_layer/MB:8.2f} MB  x{n_moe_layers}")
    print(f"  lm_head                {lm_head/MB:8.2f} MB")
    print(f"  embed row              {embed_row/1024:8.2f} KB")
    print(f"  weights total          {weights/MB:8.2f} MB")
    print(f"  KV cache @ {seq} pos   {kv_bytes/MB:8.2f} MB  "
          f"({c.layers} layers x {seq} x {c.cache_row} x {BF16}B)")
    print(f"  TOTAL                  {total/GB:8.3f} GB")

    floor_peak = total / (HBM_PEAK_TBS * 1e12) * 1e3
    floor_real = total / (HBM_ACHIEVABLE_TBS * 1e12) * 1e3
    print("\nroofline:")
    print(f"  floor @ {HBM_PEAK_TBS} TB/s (peak)        {floor_peak:6.2f} ms/token "
          f"({1e3/floor_peak:6.1f} tok/s)")
    print(f"  floor @ {HBM_ACHIEVABLE_TBS} TB/s (achievable)  {floor_real:6.2f} ms/token "
          f"({1e3/floor_real:6.1f} tok/s)")

    # Active params: everything streamed except the embedding row.
    active_params = (weights - embed_row) / BF16
    gflop = 2 * active_params / 1e9
    compute_us = gflop / (BF16_PEAK_PFLOPS * 1e6) * 1e6
    print(f"  active params           {active_params/1e9:6.2f} G -> {gflop:.2f} GFLOP/token")
    print(f"  compute @ {BF16_PEAK_PFLOPS} PFLOPS      {compute_us:6.1f} us "
          f"-> memory-bound by {floor_real*1e3/compute_us:.0f}x")

    full_model = weights - lm_head - embed_row  # rough: all layers resident
    full_model = (layer0 + n_moe_layers * (attn_total + c.n_routed * 3 * c.hidden
                  * c.moe_inter * BF16 + moe["router_gate"] + moe["shared_experts"])
                  + 2 * c.vocab * c.hidden * BF16)
    print(f"\n  whole model resident    {full_model/GB:6.2f} GB (fits 192 GB HBM)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[2] / "reference" / "dsv2lite_config.json")
    ap.add_argument("--context", type=int, default=1024, help="prompt tokens")
    ap.add_argument("--decode", type=int, default=32, help="tokens generated")
    a = ap.parse_args()
    report(Cfg.load(a.config), a.context, a.decode)


if __name__ == "__main__":
    main()
