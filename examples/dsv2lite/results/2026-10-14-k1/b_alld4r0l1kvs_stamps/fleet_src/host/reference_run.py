#!/usr/bin/env python3
"""HuggingFace reference run: the golden states every Fleet boundary is judged against.

Runs the stock DeepseekV2 implementation on the 1,024-token prompt and decodes
32 greedy tokens, saving:

  * per-layer hidden states of the first decode step (the forward that takes
    the prefill's output token as input)               -> layer-boundary checks
  * the compressed KV rows captured at kv_a_proj_with_mqa -> Fleet cache
  * 33 greedy token ids (prefill output + 32 decode outputs) and their top-2
    logit margins                                        -> end-to-end check

and, next to the .npz, the two files the C++ launcher consumes directly:
`fleet_cache.bin` (the converted cache, bf16, see kv_convert.py) and
`golden_tokens.txt` (the greedy token ids, one per line; the first is the
prefill's output and is the first decode input).

Prefill is explicitly out of scope for optimisation (task spec), so this is a
stock eager run; its latency is not a baseline and is not reported as one.

    python3 src/host/reference_run.py --model ~/models/dsv2-lite-base \
        --out build/golden.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kv_convert import CacheLayout, build_fleet_cache, write_fleet_cache_bin  # noqa: E402


def build_prompt(tokenizer, n_tokens: int) -> "torch.Tensor":
    """A fixed, reproducible prompt of exactly n_tokens."""
    import torch

    text = ("def solve(n):\n    # compute the answer\n" * 400)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.numel() < n_tokens:
        reps = n_tokens // ids.numel() + 1
        ids = ids.repeat(reps)
    return ids[:n_tokens][None, :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("build/golden.npz"))
    ap.add_argument("--device", default="cuda")  # ROCm reports as cuda
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    print(f"loading {a.model} (bf16)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager").to(a.device).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    ids = build_prompt(tok, a.context).to(a.device)
    print(f"prompt: {ids.shape[1]} tokens")

    # ---- capture per-layer hidden states and the compressed KV rows
    hidden: dict[int, np.ndarray] = {}
    compressed: dict[int, np.ndarray] = {}
    handles = []

    def layer_hook(i):
        def h(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hidden[i] = t.detach()[0, -1].float().cpu().numpy()
        return h

    def kv_hook(i):
        def h(_m, _inp, out):
            compressed[i] = out.detach()[0].float().cpu().numpy()
        return h

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(layer_hook(i)))
        handles.append(layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(kv_hook(i)))

    # Greedy pick = torch.argmax on the bf16 logits: on a tie it returns the
    # lowest index, which is the rule the kernel's argmax task implements.
    # topk/argsort are only used for the margin and do not order ties.
    def pick(logits_bf16):
        lg = logits_bf16.float()
        tok = int(torch.argmax(lg))
        top2 = torch.topk(lg, 2).values
        return tok, float(top2[0] - top2[1])

    # HF's eager DeepseekV2 attention asserts on a missing attention_mask, so
    # an explicit all-ones mask is passed for the prefill and for every decode
    # step (length = tokens so far + 1).
    def mask(n):
        return torch.ones(1, n, dtype=torch.long, device=a.device)

    print("prefill")
    with torch.no_grad():
        out = model(ids, attention_mask=mask(ids.shape[1]), use_cache=True)
    # The kv hooks have captured the prefill rows; the layer hooks captured
    # the last *prompt* position, which is not what the decode step sees.
    for h in handles:
        h.remove()
    handles = [layer.register_forward_hook(layer_hook(i))
               for i, layer in enumerate(model.model.layers)]
    hidden.clear()

    first_logits = out.logits[0, -1]
    tok0, m0 = pick(first_logits)
    print(f"  first token {tok0}, top-2 logit margin {m0:.4f}")

    # ---- greedy decode, no sampling, so the token sequence is a hard check.
    # gen_ids[0] is the prefill's output and the first decode *input*; each
    # decode step i then has gen_ids[i+1] as its expected output, so a.decode
    # launches need a.decode + 1 entries.
    print(f"decoding {a.decode} tokens greedily")
    gen_ids, margins = [tok0], [m0]
    past = out.past_key_values
    next_id = torch.tensor([[tok0]], device=a.device)

    with torch.no_grad():
        for step in range(a.decode):
            o = model(next_id, attention_mask=mask(ids.shape[1] + step + 1),
                      past_key_values=past, use_cache=True)
            if step == 0:
                # Per-layer hidden states of the first decode step: the
                # golden for layer-boundary checks of Fleet's step 0.
                for h in handles:
                    h.remove()
            past = o.past_key_values
            t_id, m = pick(o.logits[0, -1])
            next_id = torch.tensor([[t_id]], device=a.device)
            gen_ids.append(t_id)
            margins.append(m)

    print("  " + repr(tok.decode(gen_ids))[:160])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        hidden=np.stack([hidden[i] for i in range(n_layers)]),
        compressed_kv=np.stack([compressed[i] for i in range(n_layers)]),
        prompt_ids=ids[0].cpu().numpy(),
        token_ids=np.array(gen_ids, dtype=np.int64),
        margins=np.array(margins, dtype=np.float32),
        first_logits=first_logits.float().cpu().numpy(),
    )
    meta = {
        "model": str(a.model), "context": a.context, "decode": a.decode,
        "layers": n_layers, "dtype": "bfloat16", "attn": "eager",
        "tokens": gen_ids, "min_margin": float(min(margins)),
    }
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {a.out} and {a.out.with_suffix('.json')}")

    # Launcher inputs: the converted cache from *this* prefill, and the tokens.
    layout = CacheLayout(layers=n_layers, max_pos=a.context + a.decode,
                         kv_lora_rank=cfg.kv_lora_rank, qk_rope=cfg.qk_rope_head_dim)
    raw = torch.from_numpy(np.stack([compressed[i] for i in range(n_layers)]))
    cache = build_fleet_cache(model, raw, layout)
    write_fleet_cache_bin(cache, layout, a.out.parent / "fleet_cache.bin")
    (a.out.parent / "golden_tokens.txt").write_text(
        "".join(f"{t}\n" for t in gen_ids))
    print(f"wrote {a.out.parent / 'golden_tokens.txt'} ({len(gen_ids)} tokens)")
    # Per-layer outputs of the first decode step, fp32 [layers][hidden]: the
    # launcher compares its own layer outputs against these (design.md §6).
    hid = np.stack([hidden[i] for i in range(n_layers)]).astype(np.float32)
    (a.out.parent / "golden_hidden.bin").write_bytes(hid.tobytes(order="C"))
    print(f"wrote {a.out.parent / 'golden_hidden.bin'} {hid.shape}")
    print(f"  smallest top-2 margin over the run: {min(margins):.4f} "
          f"(a small margin here is where bf16 drift would flip a token)")


if __name__ == "__main__":
    main()
