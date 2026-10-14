#!/usr/bin/env python3
"""Prefill → Fleet decode cache conversion (docs/design.md §5).

The task spec requires the interface between reference prefill and Fleet decode
to be defined explicitly, including the KV-cache layout. HF's DeepseekV2 caches
per-head K `[b,16,S,192]` and V `[b,16,S,128]`, which the absorbed decode path
cannot consume: absorbed MLA reads one compressed row per position, shared by
every head.

Conversion, done once after prefill and excluded from decode latency:

    hook kv_a_proj_with_mqa on each layer  ->  raw [S, 576]
    cols   0:512  kv_a_layernorm           ->  c
    cols 512:576  RoPE(pos)                ->  k_pe
    write [layer][pos][576]

RoPE is applied by calling HF's own `apply_rotary_pos_emb` with cos/sin from the
model's rotary module, so YaRN is never re-implemented here — a second
implementation is a second thing that can silently disagree.

Validation (`--verify`) reconstructs K and V from the converted cache via
kv_b_proj and compares against HF's own cache: max abs err <= 1e-2.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # for `reference.`


@dataclass
class CacheLayout:
    """[layer][pos][kv_lora_rank + qk_rope_head_dim], bf16 on device."""
    layers: int
    max_pos: int
    kv_lora_rank: int
    qk_rope: int

    @property
    def row(self) -> int:
        return self.kv_lora_rank + self.qk_rope

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.layers, self.max_pos, self.row)

    def nbytes(self, dtype_size: int = 2) -> int:
        return self.layers * self.max_pos * self.row * dtype_size

    def describe(self) -> str:
        return (f"[{self.layers}][{self.max_pos}][{self.row}] "
                f"cols 0:{self.kv_lora_rank} = c (post kv_a_layernorm), "
                f"{self.kv_lora_rank}:{self.row} = k_pe (post RoPE)  "
                f"-> {self.nbytes()/1e6:.1f} MB bf16")


def capture_compressed_kv(model, input_ids, layout: CacheLayout):
    """Run HF prefill with hooks on kv_a_proj_with_mqa; return raw [L, S, 576].

    The hook records the projection *before* kv_a_layernorm and RoPE, which is
    the only place the compressed representation exists in HF's graph.
    """
    import torch

    raw: dict[int, "torch.Tensor"] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_mod, _inp, out):
            raw[idx] = out.detach()[0].float().cpu()  # [S, 576]
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(
            make_hook(i)))
    try:
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    stacked = torch.stack([raw[i] for i in range(layout.layers)])  # [L, S, 576]
    return stacked, out


def build_fleet_cache(model, raw, layout: CacheLayout):
    """Apply kv_a_layernorm to c and RoPE to k_pe, per layer, per position."""
    import torch
    from reference.modeling_deepseek import apply_rotary_pos_emb

    L, S, _ = raw.shape
    cache = torch.zeros(L, S, layout.row, dtype=torch.float32)

    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            dev = attn.kv_a_layernorm.weight.device     # wherever the model lives
            c_part = raw[i, :, : layout.kv_lora_rank].to(dev)
            k_pe = raw[i, :, layout.kv_lora_rank:].to(dev)
            pos_ids = torch.arange(S, device=dev)[None, :]

            dtype = attn.kv_a_layernorm.weight.dtype          # bf16 for the real model
            c_norm = attn.kv_a_layernorm(c_part.to(dtype))

            # HF's own rotary module supplies cos/sin, so YaRN lives in one
            # place — and the rotation runs in the model's dtype, as it does
            # inside HF's forward, so the rows are bit-identical to HF's.
            k_pe_4d = k_pe.to(dtype)[None, None]              # [1,1,S,64]
            cos, sin = attn.rotary_emb(k_pe_4d, seq_len=S)
            _, k_rot = apply_rotary_pos_emb(k_pe_4d, k_pe_4d, cos, sin, pos_ids)

            cache[i, :, : layout.kv_lora_rank] = c_norm.float().cpu()
            cache[i, :, layout.kv_lora_rank:] = k_rot[0, 0].float().cpu()
    return cache


def write_fleet_cache_bin(cache, layout: CacheLayout, path: Path) -> None:
    """Write the cache as raw bf16 [layers][max_pos][row], the exact bytes the
    launcher uploads; decode positions past the prefill are zero and the
    kernel appends them one per step.
    """
    import torch

    L, S, R = cache.shape
    assert (L, R) == (layout.layers, layout.row), (cache.shape, layout.shape)
    assert S <= layout.max_pos, f"prefill {S} exceeds max_pos {layout.max_pos}"
    full = torch.zeros(layout.shape, dtype=torch.float32)
    full[:, :S] = cache
    raw = full.to(torch.bfloat16).view(torch.int16).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw.tobytes(order="C"))
    assert path.stat().st_size == layout.nbytes()
    print(f"wrote {path}  {layout.describe()}")


def verify(model, cache, hf_past, layout: CacheLayout, tol: float = 1e-2) -> bool:
    """Reconstruct the full K (nope ‖ rope) and V from the cache *as the
    launcher will load it* (rounded to bf16) and compare with HF's own cache.
    """
    import torch

    ok = True
    cache = cache.to(torch.bfloat16).float()             # what fleet_cache.bin holds
    L = layout.kv_lora_rank
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        Dk, Dv = attn.qk_nope_head_dim, attn.v_head_dim
        H = attn.num_heads
        dev = attn.kv_b_proj.weight.device

        c = cache[i, :, :L].to(dev)
        k_pe = cache[i, :, L:].to(dev)                   # [S, 64], post-RoPE
        kv = attn.kv_b_proj(c.to(attn.kv_b_proj.weight.dtype)).float()
        kv = kv.view(-1, H, Dk + Dv)
        k_nope, v = kv[..., :Dk], kv[..., Dk:]

        k_hf, v_hf = hf_past[i]
        k_hf = k_hf[0].transpose(0, 1).float().to(dev)   # [S, H, 192] = nope ‖ rope
        v_hf = v_hf[0].transpose(0, 1).float().to(dev)   # [S, H, 128]

        e_k = (k_nope - k_hf[..., :Dk]).abs().max().item()
        e_r = (k_pe[:, None, :] - k_hf[..., Dk:]).abs().max().item()   # same for every head
        e_v = (v - v_hf).abs().max().item()
        good = e_k <= tol and e_r <= tol and e_v <= tol
        ok &= good
        print(f"  layer {i:2d}: K_nope err {e_k:.2e}  K_rope err {e_r:.2e}  "
              f"V err {e_v:.2e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, help="local model dir")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=Path("build/fleet_cache_check.npy"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--describe", action="store_true",
                    help="print the layout without loading the model")
    a = ap.parse_args()

    layout = CacheLayout(layers=27, max_pos=a.tokens + 32,
                         kv_lora_rank=512, qk_rope=64)
    print("Fleet cache layout:", layout.describe())
    if a.describe or not a.model:
        return

    import torch
    from transformers import AutoModelForCausalLM

    # This standalone mode exists for the reconstruction check on a random
    # prompt. The cache the launcher decodes from is written by
    # reference_run.py, from the same prefill that produced the golden tokens.
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    ids = torch.randint(0, 100, (1, a.tokens))

    raw, out = capture_compressed_kv(model, ids, layout)
    cache = build_fleet_cache(model, raw, layout)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, cache.numpy().astype(np.float32))
    print(f"wrote {a.out}  {cache.shape}")

    if a.verify:
        past = out.past_key_values
        print("reconstruction check (K,V from the bf16 compressed cache vs HF):")
        ok = verify(model, cache, past, layout)
        print("PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
