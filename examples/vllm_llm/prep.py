#!/usr/bin/env python3
"""Pack a HF Qwen checkpoint in the layout vLLM's kernels read (vl.bin + vl.manifest).

vLLM's layout: qkv_proj rows = [q; k; v] (QKVParallelLinear), gate_up_proj rows = [gate; up]
(MergedColumnParallelLinear, silu_and_mul takes x[:d] and x[d:]), every weight row-major [out][in] bf16.
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
    if d.moe:
        raise SystemExit("MoE layers are not in the vLLM import set yet (fused_moe is a Triton kernel)")
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
    put("cos_sin", cos_sin_cache(d.theta, d.HD, a.maxlen))
    for L in range(d.L):
        p = f"{pre}layers.{L}."
        put(f"L{L}.ln1", get(p + "input_layernorm.weight"))
        put(f"L{L}.ln2", get(p + "post_attention_layernorm.weight"))
        s = p + "self_attn."
        put(f"L{L}.wqkv", torch.cat([get(s + "q_proj.weight"), get(s + "k_proj.weight"), get(s + "v_proj.weight")], 0))
        if (s + "q_proj.bias") in index:
            put(f"L{L}.bqkv", torch.cat([get(s + "q_proj.bias"), get(s + "k_proj.bias"), get(s + "v_proj.bias")], 0))
        if (s + "q_norm.weight") in index:
            put(f"L{L}.qn", get(s + "q_norm.weight"))
            put(f"L{L}.kn", get(s + "k_norm.weight"))
        put(f"L{L}.wo", get(s + "o_proj.weight"))
        m = p + "mlp."
        put(f"L{L}.wgu", torch.cat([get(m + "gate_proj.weight"), get(m + "up_proj.weight")], 0))
        put(f"L{L}.wd", get(m + "down_proj.weight"))
        print(f"  packed layer {L + 1}/{d.L}", end="\r", flush=True)
    wf.close()
    print()
    cfg = {"H": d.H, "NH": d.NH, "NKV": d.NKV, "HD": d.HD, "INTER": d.INTER, "L": d.L, "V": d.V, "tie": int(d.tie),
           "bias": int(d.bias), "qk_norm": int(d.qk_norm), "maxlen": a.maxlen, "bytes": off}
    with open(out / "vl.manifest", "w") as f:
        for k, v in cfg.items():
            f.write(f"cfg {k} {v}\n")
        f.write(f"cfgf eps {d.eps!r}\n")
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}/vl.bin ({off / 1e9:.2f} GB) and vl.manifest: {json.dumps(cfg)}")


if __name__ == "__main__":
    main()
