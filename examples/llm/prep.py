"""Prepare one model for examples/llm: download, Hugging Face greedy reference, packed weights.

  python examples/llm/prep.py --model Qwen/Qwen3-8B --out ~/llm/qwen3-8b [--gen 32] [--prompt "..."]

Writes <out>/config.json, weights.bin (bf16, 256-byte aligned tensors in the layout the tiles read),
llm.manifest (cfg and tensor lines for the host) and golden.txt (prompt ids, HF greedy ids, HF top-1
margins at each generated position). Needs torch (ROCm), transformers, safetensors, huggingface_hub.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.llm import model as M   # noqa: E402

PROMPT = ("The history of the transistor begins in 1947 at Bell Labs, where John Bardeen, Walter Brattain "
          "and William Shockley")


def reference(src: Path, prompt: str, gen: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=gen, min_new_tokens=gen, do_sample=False, num_beams=1,
                             pad_token_id=tok.eos_token_id)
        full = out[:, : ids.shape[1] + gen]
        logits = model(full).logits[0].float()                 # teacher-forced logits for the margins
    gen_ids = out[0, ids.shape[1]:].tolist()
    margins = []
    for i in range(gen):
        top2 = torch.topk(logits[ids.shape[1] - 1 + i], 2).values
        margins.append(float(top2[0] - top2[1]))
    text = tok.decode(gen_ids)
    del model
    torch.cuda.empty_cache()
    return ids[0].tolist(), gen_ids, margins, text


def pack(src: Path, out: Path, d: M.Dims):
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
    wf = open(out / "weights.bin", "wb")

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
    for L in range(d.L):
        p = f"{pre}layers.{L}."
        put(f"L{L}.ln1", get(p + "input_layernorm.weight"))
        put(f"L{L}.ln2", get(p + "post_attention_layernorm.weight"))
        a = p + "self_attn."
        put(f"L{L}.wqkv", torch.cat([get(a + "q_proj.weight"), get(a + "k_proj.weight"), get(a + "v_proj.weight")], 0))
        if (a + "q_proj.bias") in index:
            put(f"L{L}.bqkv", torch.cat([get(a + "q_proj.bias"), get(a + "k_proj.bias"), get(a + "v_proj.bias")], 0))
        if (a + "q_norm.weight") in index:
            put(f"L{L}.qn", get(a + "q_norm.weight"))
            put(f"L{L}.kn", get(a + "k_norm.weight"))
        put(f"L{L}.wo", get(a + "o_proj.weight"))
        m = p + "mlp."
        if d.moe:
            put(f"L{L}.wr", get(m + "gate.weight"))
            for kind, key in (("eg", "gate_proj"), ("eu", "up_proj"), ("ed", "down_proj")):
                put(f"L{L}.{kind}", torch.stack([get(f"{m}experts.{e}.{key}.weight") for e in range(d.E)], 0))
        else:
            put(f"L{L}.wg", get(m + "gate_proj.weight"))
            put(f"L{L}.wu", get(m + "up_proj.weight"))
            put(f"L{L}.wd", get(m + "down_proj.weight"))
        print(f"  packed layer {L + 1}/{d.L}", end="\r", flush=True)
    wf.close()
    print()
    return lines, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--src", default=None, help="local snapshot dir (default: download to <out>/hf)")
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out)); out.mkdir(parents=True, exist_ok=True)
    src = Path(a.src) if a.src else out / "hf"
    if not (src / "config.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(a.model, local_dir=src, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"], max_workers=16)
    cfg = json.loads((src / "config.json").read_text())
    (out / "config.json").write_text(json.dumps(cfg))
    d = M.load_dims(str(out / "config.json"))
    part = M.partition(d)

    if not (out / "golden.txt").exists():
        pids, gids, margins, text = reference(src, a.prompt, a.gen)
        (out / "golden.txt").write_text("prompt: " + " ".join(map(str, pids)) + "\n" + "gen: " + " ".join(map(str, gids)) + "\n"
                                        + "margin: " + " ".join(f"{m:.4f}" for m in margins) + "\n")
        print("HF greedy:", repr(text))
    if not (out / "llm.manifest").exists():
        tlines, nbytes = pack(src, out, d)
        # bytes of weights one decode step reads (embedding row excluded; MoE: only the top-k experts)
        attn = (d.NQKV * d.H + d.H * d.NH * d.HD) * 2
        mlp = (d.E * d.H + d.TOPK * 3 * d.MI * d.H) * 2 if d.moe else 3 * d.INTER * d.H * 2
        active = d.L * (attn + mlp) + d.V * d.H * 2
        cfg_lines = [f"cfg {k} {v}" for k, v in dict(
            H=d.H, NH=d.NH, NKV=d.NKV, HD=d.HD, INTER=d.INTER if not d.moe else 0, L=d.L, V=d.V, E=d.E, TOPK=d.TOPK, MI=d.MI,
            CTX=M.CTX, CHUNK=M.CHUNK, qk_norm=int(d.qk_norm), bias=int(d.bias), moe=int(d.moe), norm_topk=int(d.norm_topk),
            tie=int(d.tie), eps=d.eps, theta=d.theta, active_bytes=active,
            **{f"rpt_{k}": v for k, v in part.items()}).items()]
        (out / "llm.manifest").write_text("\n".join(cfg_lines + tlines) + "\n")
        print(f"packed {nbytes / 1e9:.2f} GB; active per token {active / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
