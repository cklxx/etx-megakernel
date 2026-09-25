#!/usr/bin/env python3
"""CPU reference of the examples/llm tile pipeline: the same operations, in the same order, with bf16
rounding at the same points as tiles.hip, in numpy. It is the specification the GPU tiles are checked
against, and it is checked itself against Hugging Face.

  python examples/llm/ref_numpy.py --model ~/models/qwen2.5-0.5b --gen 8 [--context 0] [--hf] [--dump ref.npz]

--hf     also runs the HF model (bf16, CPU) on the same prompt and compares tokens and per-layer residuals
--dump   writes the per-layer residuals of the LAST prompt step and the generated ids, for compare_dump.py
Runs the prompt token by token like the kernel does, so prefill and decode share one code path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.llm import model as M  # noqa: E402


def bfr(x):
    """round-to-nearest-even to bf16, returned as float32 (exactly what __float2bfloat16 does)."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    u = x.view(np.uint32)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) & 0xFFFF0000).astype(np.uint32)
    return r.view(np.float32)


def bf16_to_f32(a: np.ndarray) -> np.ndarray:            # uint16 bf16 bits -> float32
    return (a.astype(np.uint32) << 16).view(np.float32)


class Weights:
    """Lazy bf16 -> fp32 access to the safetensors of a HF checkpoint."""

    def __init__(self, src: Path):
        from safetensors import safe_open
        self.index, self.handles = {}, {}
        for f in sorted(src.glob("*.safetensors")):
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    self.index[k] = f
        self.cache: dict[str, np.ndarray] = {}

    def __contains__(self, k):
        return k in self.index

    def get(self, k: str) -> np.ndarray:
        if k in self.cache:
            return self.cache[k]
        from safetensors import safe_open
        f = self.index[k]
        if f not in self.handles:
            self.handles[f] = safe_open(f, framework="pt")
        import torch
        t = self.handles[f].get_tensor(k)
        if t.dtype == torch.bfloat16:
            v = bf16_to_f32(t.contiguous().view(torch.int16).numpy().view(np.uint16))
        else:
            v = t.float().numpy()
        self.cache[k] = v
        return v


def rmsnorm(x, w, eps):                                  # HF Qwen2RMSNorm in bf16
    r = np.float32(1.0) / np.sqrt(np.float32(np.mean(x.astype(np.float32) ** 2)) + np.float32(eps))
    return bfr(w * bfr(x * r))


def rope_tables(theta, hd, ctx):
    half = hd // 2
    inv = (1.0 / (np.float32(theta) ** (np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd)))).astype(np.float32)
    pos = np.arange(ctx, dtype=np.float32)[:, None]
    f = pos * inv[None, :]
    return bfr(np.cos(f)), bfr(np.sin(f))


def apply_rope(v, cos, sin):                             # rotate_half, each product rounded, sum rounded
    half = v.shape[-1] // 2
    rot = np.concatenate([-v[half:], v[:half]])
    c = np.concatenate([cos, cos]); s = np.concatenate([sin, sin])
    return bfr(bfr(v * c) + bfr(rot * s))


def silu(g):
    return g / (np.float32(1.0) + np.exp(-g))


class Ref:
    def __init__(self, src: Path, ctx: int):
        self.d = M.load_dims(str(src / "config.json"))
        self.W = Weights(src)
        self.ctx = ctx
        d = self.d
        self.cos, self.sin = rope_tables(d.theta, d.HD, ctx)
        self.kc = np.zeros((d.L, ctx, d.NKV, d.HD), np.float32)
        self.vc = np.zeros((d.L, ctx, d.NKV, d.HD), np.float32)

    def step(self, tok: int, pos: int, dump=None) -> int:
        d, W = self.d, self.W
        x = W.get("model.embed_tokens.weight")[tok].copy()
        for L in range(d.L):
            p = f"model.layers.{L}."
            xs = rmsnorm(x, W.get(p + "input_layernorm.weight"), d.eps)
            a = p + "self_attn."
            q = bfr(W.get(a + "q_proj.weight") @ xs + (W.get(a + "q_proj.bias") if (a + "q_proj.bias") in W else 0))
            k = bfr(W.get(a + "k_proj.weight") @ xs + (W.get(a + "k_proj.bias") if (a + "k_proj.bias") in W else 0))
            v = bfr(W.get(a + "v_proj.weight") @ xs + (W.get(a + "v_proj.bias") if (a + "v_proj.bias") in W else 0))
            q = q.reshape(d.NH, d.HD); k = k.reshape(d.NKV, d.HD); v = v.reshape(d.NKV, d.HD)
            if d.qk_norm:
                qn, kn = W.get(a + "q_norm.weight"), W.get(a + "k_norm.weight")
                q = np.stack([rmsnorm(h, qn, d.eps) for h in q]); k = np.stack([rmsnorm(h, kn, d.eps) for h in k])
            q = np.stack([apply_rope(h, self.cos[pos], self.sin[pos]) for h in q])
            k = np.stack([apply_rope(h, self.cos[pos], self.sin[pos]) for h in k])
            self.kc[L, pos] = k; self.vc[L, pos] = v
            scale = np.float32(1.0 / np.sqrt(d.HD))
            o = np.zeros((d.NH, d.HD), np.float32)
            for h in range(d.NH):
                kvh = h // d.G
                K = self.kc[L, : pos + 1, kvh]; V = self.vc[L, : pos + 1, kvh]
                s = (K @ q[h]) * scale
                s = s - s.max(); pr = np.exp(s); pr = pr / pr.sum()
                o[h] = bfr(pr @ V)
            x = bfr(x + bfr(W.get(a + "o_proj.weight") @ o.reshape(-1)))
            xs = rmsnorm(x, W.get(p + "post_attention_layernorm.weight"), d.eps)
            m = p + "mlp."
            if d.moe:
                logits = bfr(W.get(m + "gate.weight") @ xs)
                pr = np.exp(logits - logits.max()); pr = pr / pr.sum()
                order = np.lexsort((np.arange(d.E), -pr))[: d.TOPK]        # top-k, ties -> lower id
                wts = pr[order]
                if d.norm_topk:
                    wts = wts / wts.sum()
                wts = bfr(wts)
                acc = np.float32(0.0)
                for e, we in sorted(zip(order.tolist(), wts.tolist())):     # ascending expert id, like index_add_
                    g = bfr(W.get(f"{m}experts.{e}.gate_proj.weight") @ xs)
                    u = bfr(W.get(f"{m}experts.{e}.up_proj.weight") @ xs)
                    hh = bfr(bfr(silu(g)) * u)
                    y = bfr(bfr(W.get(f"{m}experts.{e}.down_proj.weight") @ hh) * np.float32(we))
                    acc = bfr(acc + y)
                x = bfr(x + acc)
            else:
                g = bfr(W.get(m + "gate_proj.weight") @ xs); u = bfr(W.get(m + "up_proj.weight") @ xs)
                hh = bfr(bfr(silu(g)) * u)
                x = bfr(x + bfr(W.get(m + "down_proj.weight") @ hh))
            if dump is not None:
                dump.append(x.copy())
        xs = rmsnorm(x, W.get("model.norm.weight"), d.eps)
        lm = W.get("lm_head.weight") if "lm_head.weight" in W else W.get("model.embed_tokens.weight")
        logits = bfr(lm @ xs)
        return int(np.argmax(logits))                     # first max wins, like torch.argmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--prompt", default="The history of the transistor begins in 1947 at Bell Labs, where")
    ap.add_argument("--context", type=int, default=0, help="fleet's benchmark prompt of N tokens instead of --prompt")
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    src = Path(a.model).expanduser()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(src)
    if a.context:
        from examples.llm.prep import context_prompt_ids
        ids = context_prompt_ids(tok, a.context)
    else:
        ids = tok(a.prompt).input_ids
    ref = Ref(src, len(ids) + a.gen + 8)
    seq = list(ids); out = []; dumps = []
    t0 = time.time()
    for t in range(len(ids) + a.gen - 1):
        nxt = ref.step(seq[t], t, dumps if t == len(ids) - 1 else None)
        if t >= len(ids) - 1:
            out.append(nxt)
            if len(seq) <= t + 1:
                seq.append(nxt)
    print(f"numpy reference: {len(ids)} prompt tokens, {a.gen} generated in {time.time() - t0:.1f}s")
    print("generated:", out, repr(tok.decode(out)))
    if a.dump:
        np.savez(a.dump, prompt=np.array(ids), gen=np.array(out), layers=np.stack(dumps))
        print("wrote", a.dump)
    if a.hf:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
        model.eval()
        with torch.no_grad():
            x = torch.tensor([ids])
            g = model.generate(x, max_new_tokens=a.gen, do_sample=False, eos_token_id=None, pad_token_id=tok.eos_token_id)
            hf = g[0, len(ids):].tolist()
            hs = model(x, output_hidden_states=True).hidden_states
        match = next((i for i in range(a.gen) if hf[i] != out[i]), a.gen)
        print(f"HF greedy: {hf}; tokens equal up to {match}/{a.gen}")
        for L, h in enumerate(hs[1:]):
            ours = dumps[L]; theirs = h[0, -1].float().numpy()
            # HF's hidden_states[L+1] is the residual after layer L, except the last one which has the final norm applied
            if L == len(hs) - 2:
                continue
            cos = float(ours @ theirs / (np.linalg.norm(ours) * np.linalg.norm(theirs) + 1e-30))
            rel = float(np.max(np.abs(ours - theirs)) / max(np.max(np.abs(theirs)), 1e-12))
            flag = "" if rel < 2e-2 and cos > 0.999 else "  <-- differs"
            print(f"  layer {L:2d}: cosine {cos:.6f}  max rel {rel:.2e}{flag}")


if __name__ == "__main__":
    main()
