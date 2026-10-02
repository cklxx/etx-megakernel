#!/usr/bin/env python3
"""Pack a HF Qwen checkpoint in the layout vLLM's kernels read (vl.bin + vl.manifest).

vLLM's layout: qkv_proj rows = [q; k; v] (QKVParallelLinear), gate_up_proj rows = [gate; up]
(MergedColumnParallelLinear, silu_and_mul takes x[:d] and x[d:]), every weight row-major [out][in] bf16.
MoE layers (FusedMoE): the router `gate` [E][H]; w13 [E][2 * I][H] with each expert's gate rows then its up rows
(FusedMoE.weight_loader: w1 -> shard 0, w3 -> shard 1 of w13_weight); w2 [E][H][I].
The rotary cache is vLLM's RotaryEmbedding.cos_sin_cache: float32 cos | sin of pos * base^(-2i/rot),
cast to the model dtype (bf16), computed with torch like vLLM does (on the GPU when there is one, as
vLLM builds it under the model's device context).

  python examples/vllm_llm/prep.py --src ~/llm/qwen3-8b/hf --out ~/llm/qwen3-8b --maxlen 1072
Writes <out>/vl.bin and <out>/vl.manifest; the golden file comes from examples/llm/prep.py (--context 1024).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.llm import model as M  # noqa: E402


def longrope_cos_sin_cache(theta: float, rot: int, maxlen: int, cfg: dict) -> torch.Tensor:
    """Phi's longrope (vLLM Phi3LongRoPEScaledRotaryEmbedding): inv_freq = 1 / (factor_i * theta^(2i/rot)) with the
    short factors below original_max_position_embeddings and the long ones above; cos/sin scaled by the config's
    explicit short_mscale / long_mscale (Phi-3.5-MoE carries both; no derived formula is used when they are given).
    The cache layout matches cos_sin_cache: maxlen rows of cos (rot/2) | sin (rot/2)."""
    rs = cfg["rope_scaling"]
    orig = int(rs.get("original_max_position_embeddings") or cfg.get("original_max_position_embeddings"))
    dev = "cpu"
    t = torch.arange(maxlen, dtype=torch.float, device=dev)
    out = torch.empty(maxlen, rot, dtype=torch.float)
    for which, lo, hi in (("short", 0, min(maxlen, orig)), ("long", min(maxlen, orig), maxlen)):
        if lo >= hi:
            continue
        fac = torch.tensor(rs[f"{which}_factor"], dtype=torch.float, device=dev)
        msc = float(rs.get(f"{which}_mscale") or 1.0)
        inv_freq = 1.0 / (fac * theta ** (torch.arange(0, rot, 2, dtype=torch.float, device=dev) / rot))
        fr = torch.einsum("i,j->ij", t[lo:hi], inv_freq)
        out[lo:hi] = torch.cat((fr.cos() * msc, fr.sin() * msc), dim=-1)
    return out


def cos_sin_cache(theta: float, rot: int, maxlen: int) -> torch.Tensor:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inv_freq = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float, device=dev) / rot))
    t = torch.arange(maxlen, dtype=torch.float, device=dev)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxlen", type=int, default=1072)
    a = ap.parse_args()
    src, out = Path(a.src).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    d = M.load_dims(str(src / "config.json"))
    from safetensors import safe_open
    index = {}
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                index[k] = f
    handles = {}

    def get(name: str) -> torch.Tensor:
        f = index[name]
        if f not in handles:
            handles[f] = safe_open(f, framework="pt")
        return handles[f].get_tensor(name).to(torch.bfloat16).contiguous()

    lines, off = [], 0
    wf = open(out / "vl.bin", "wb")

    def put(name: str, t: torch.Tensor):
        nonlocal off
        pad = (-off) % 256
        if pad:
            wf.write(b"\0" * pad); off += pad
        b = t.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()
        wf.write(b)
        lines.append(f"t {name} {off} {t.numel()}")
        off += len(b)

    pre = "model."
    put("embed", get(pre + "embed_tokens.weight"))
    put("final_norm", get(pre + "norm.weight"))
    if not d.tie:
        put("lm_head", get("lm_head.weight"))
    cfg_json = json.load(open(src / "config.json"))
    if (cfg_json.get("rope_scaling") or {}).get("type") == "longrope":
        put("cos_sin", longrope_cos_sin_cache(d.theta, d.HD, a.maxlen, cfg_json))
    else:
        put("cos_sin", cos_sin_cache(d.theta, d.HD, a.maxlen))
    if not d.tie and "lm_head.bias" in index:
        put("lm_head_b", get("lm_head.bias"))
    if (pre + "norm.bias") in index:
        put("final_norm_b", get(pre + "norm.bias"))
    for L in range(d.L):
        p = f"{pre}layers.{L}."
        put(f"L{L}.ln1", get(p + "input_layernorm.weight"))
        put(f"L{L}.ln2", get(p + "post_attention_layernorm.weight"))
        if (p + "input_layernorm.bias") in index:                    # PhiMoE: nn.LayerNorm with a bias
            put(f"L{L}.ln1b", get(p + "input_layernorm.bias"))
            put(f"L{L}.ln2b", get(p + "post_attention_layernorm.bias"))
        s = p + "self_attn."
        put(f"L{L}.wqkv", torch.cat([get(s + "q_proj.weight"), get(s + "k_proj.weight"), get(s + "v_proj.weight")], 0))
        if (s + "q_proj.bias") in index:
            put(f"L{L}.bqkv", torch.cat([get(s + "q_proj.bias"), get(s + "k_proj.bias"), get(s + "v_proj.bias")], 0))
        if (s + "q_norm.weight") in index:
            put(f"L{L}.qn", get(s + "q_norm.weight"))
            put(f"L{L}.kn", get(s + "k_norm.weight"))
        put(f"L{L}.wo", get(s + "o_proj.weight"))
        if (s + "o_proj.bias") in index:
            put(f"L{L}.bo", get(s + "o_proj.bias"))
        # Qwen spells the MoE block mlp.{gate,experts.N.{gate_proj,up_proj,down_proj}}; Mixtral spells it
        # block_sparse_moe.{gate,experts.N.{w1,w3,w2}} with w1=gate, w3=up, w2=down (vLLM's FusedMoE maps
        # w1->shard 0 and w3->shard 1 of w13 for both). Detect by what the checkpoint actually contains.
        m = p + ("block_sparse_moe." if (p + "block_sparse_moe.gate.weight") in index else "mlp.")
        if d.moe:
            gk, uk, dk = (("w1", "w3", "w2") if "block_sparse_moe" in m else ("gate_proj", "up_proj", "down_proj"))
            put(f"L{L}.wr", get(m + "gate.weight"))
            put(f"L{L}.w13", torch.stack([torch.cat([get(f"{m}experts.{e}.{gk}.weight"), get(f"{m}experts.{e}.{uk}.weight")], 0)
                                          for e in range(d.E)]))
            put(f"L{L}.w2", torch.stack([get(f"{m}experts.{e}.{dk}.weight") for e in range(d.E)]))
        else:
            put(f"L{L}.wgu", torch.cat([get(m + "gate_proj.weight"), get(m + "up_proj.weight")], 0))
            put(f"L{L}.wd", get(m + "down_proj.weight"))
        print(f"  packed layer {L + 1}/{d.L}", end="\r", flush=True)
    wf.close()
    print()
    cfg = {"H": d.H, "NH": d.NH, "NKV": d.NKV, "HD": d.HD, "INTER": d.INTER, "L": d.L, "V": d.V, "tie": int(d.tie),
           "bias": int(d.bias), "qk_norm": int(d.qk_norm), "maxlen": a.maxlen, "bytes": off,
           "moe": int(d.moe), "E": d.E, "TOPK": d.TOPK, "MI": d.MI, "norm_topk": int(d.norm_topk),
           "ln_norm": int(getattr(d, "ln_norm", False)), "sparsemixer": int(getattr(d, "sparsemixer", False)),
           "bias_o": int(getattr(d, "bias_o", False)), "bias_lm": int(getattr(d, "bias_lm", False))}
    with open(out / "vl.manifest", "w") as f:
        for k, v in cfg.items():
            f.write(f"cfg {k} {v}\n")
        f.write(f"cfgf eps {d.eps!r}\n")
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}/vl.bin ({off / 1e9:.2f} GB) and vl.manifest: {json.dumps(cfg)}")


if __name__ == "__main__":
    main()
