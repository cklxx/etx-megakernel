"""Is DeepSeek-V2-Lite's routing structured enough for Zyphra-style locality tricks?

Zyphra (arXiv 2610.09372 / zyphra.com/our-work/expert-coupling-in-moe-pretraining) claims two
correlations that let them cut MoE all-to-all traffic on MI300X: experts selected by the same
token are correlated within a layer, and a token's experts at layer l predict its experts at
layer l+1.  Their model is top-2 of 128; their headline is "0.8% of expert pairs carry 42% of
tokens".

This script measures the same two correlations on DeepSeek-V2-Lite (top-6 of 64), plus a
third one that matters for a batch-1 decode megakernel: how much the selected experts overlap
between consecutive decode steps (cross-step weight reuse).

Every measured number is reported next to a control: routing drawn uniformly at random with
the same shape (same steps, layers, top-k, experts).  The control is what "no structure"
looks like at this sample size, and at top-6/64 one pair's uniform share (k(k-1)/E(E-1)) is
60x the one at top-2/128, so the raw Zyphra metric is not comparable across configs without it.

Usage:
  capture (needs a GPU and ~/models/dsv2-lite-base; the 31 GB download does not fit on the laptop):
    ~/fleet-mi300x/.venv/bin/python examples/dsv2lite/routing_stats.py \
        --model ~/models/dsv2-lite-base --steps 32 --batch 4 --dump ids.npz
  analyze locally from a dump:
    .venv/bin/python examples/dsv2lite/routing_stats.py --ids ids.npz
  self-test of the analysis code on synthetic routing (no model, no GPU):
    .venv/bin/python examples/dsv2lite/routing_stats.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- analysis

def pairs_of(row: np.ndarray) -> list[tuple[int, int]]:
    """The C(k,2) unordered expert pairs a single token selected (row is one token's ids)."""
    return [(int(a), int(b)) for a, b in combinations(sorted(int(x) for x in row), 2)]


def pair_concentration(ids: np.ndarray, k: int, E: int, frac: float = 0.008) -> dict:
    """Share of tokens that select one of the hottest 0.8% of expert pairs, per layer.

    Zyphra's form: "in layer 8 of our top-2 model, just 64 of the 8128 possible pairs account
    for 42% of tokens".  A token counts once, however many of the bucket pairs it selected.
    """
    n_tok, n_lay, _ = ids.shape
    total_pairs = E * (E - 1) // 2
    base = k * (k - 1) / (E * (E - 1))                       # P(a given pair is co-selected)
    m = max(1, round(frac * total_pairs))                    # Zyphra's 0.8% bucket
    shares, seen, hottest = [], [], []
    for lay in range(n_lay):
        counts: dict[tuple[int, int], int] = {}
        row_pairs = [pairs_of(ids[tok, lay]) for tok in range(n_tok)]
        for ps in row_pairs:
            for p in ps:
                counts[p] = counts.get(p, 0) + 1
        seen.append(len(counts))
        ranked = sorted(counts, key=lambda p: (-counts[p], p))
        top = set(ranked[:m])
        shares.append(sum(1 for ps in row_pairs if any(p in top for p in ps)) / n_tok)
        hottest.append(counts[ranked[0]] / n_tok if ranked else 0.0)
    shares = np.array(shares)
    uniform = min(1.0, m * base)
    return {
        "tokens": n_tok, "layers": n_lay, "top_k": k, "experts": E,
        "pairs_total": total_pairs, "pairs_seen_mean": float(np.mean(seen)),
        "bucket_pairs": m, "bucket_share": float(shares.mean()),
        "bucket_share_min": float(shares.min()), "bucket_share_max": float(shares.max()),
        "uniform_per_pair": base, "uniform_share": uniform,
        "ratio_vs_uniform": float(shares.mean()) / uniform,
        "hottest_pair_share": float(np.mean(hottest)),
        "hottest_pair_ratio": float(np.mean(hottest)) / base,
    }


def cross_layer(ids: np.ndarray, k: int, E: int, min_support: int = 8) -> dict:
    """How well does a token's expert at layer l predict its experts at layer l+1?

    Zyphra's form: 112 of 128 experts send >=30% of their tokens to one particular expert next
    layer, median 48%.  We report the same two numbers, over experts with enough support.
    """
    n_tok, n_lay, _ = ids.shape
    p_max, support, supported = [], [], 0
    for lay in range(n_lay - 1):
        for e in range(E):
            sel = np.where((ids[:, lay] == e).any(axis=1))[0]
            if len(sel) < min_support:
                continue
            supported += 1
            support.append(len(sel))
            nxt = ids[sel, lay + 1].ravel()
            p_max.append(np.bincount(nxt, minlength=E).max() / len(sel))
    if not p_max:
        return {"experts_with_support": 0}
    p_max = np.array(p_max)
    return {
        "uniform_p_max_expectation": k / E,
        "experts_with_support": supported,
        "median_support_tokens": float(np.median(support)),
        "median_p_max": float(np.median(p_max)),
        "min_p_max": float(p_max.min()),
        "experts_over_30pct": int((p_max >= 0.30).sum()),
    }


def step_overlap(ids: np.ndarray, k: int, E: int) -> dict:
    """Experts shared by the same sequence on consecutive decode steps (cross-step reuse)."""
    n_tok, n_lay, _ = ids.shape
    ov = [len(set(ids[t, lay]) & set(ids[t + 1, lay]))
          for lay in range(n_lay) for t in range(n_tok - 1)]
    return {
        "mean_shared_experts": float(np.mean(ov)),
        "uniform_expectation": k * k / E,
    }


def analyze(ids: np.ndarray, k: int, E: int) -> dict:
    return {
        "pair_concentration": pair_concentration(ids, k, E),
        "cross_layer": cross_layer(ids, k, E),
        "step_overlap": step_overlap(ids, k, E),
    }


def uniform_control(steps: int, layers: int, k: int, E: int, seed: int = 0) -> np.ndarray:
    """Same shape, no structure: every token picks k experts uniformly without replacement."""
    rng = np.random.default_rng(seed)
    scores = rng.random((steps, layers, E))
    idx = np.argpartition(-scores, k - 1, axis=2)[:, :, :k]
    return np.sort(idx).astype(np.int16)


def report(name: str, res: dict, k: int) -> None:
    pc, cl, so = res["pair_concentration"], res["cross_layer"], res["step_overlap"]
    print(f"== {name} ==")
    print(f"  pair concentration: the hottest {pc['bucket_pairs']} of {pc['pairs_total']} pairs "
          f"({pc['pairs_seen_mean']:.0f} seen per layer) carry {100*pc['bucket_share']:.2f}% of "
          f"tokens (per layer {100*pc['bucket_share_min']:.1f}-{100*pc['bucket_share_max']:.1f}%); "
          f"uniform {100*pc['uniform_share']:.2f}% -> ratio {pc['ratio_vs_uniform']:.2f}x")
    print(f"    the single hottest pair carries {100*pc['hottest_pair_share']:.2f}% of tokens "
          f"({pc['hottest_pair_ratio']:.1f}x one pair's uniform share "
          f"{100*pc['uniform_per_pair']:.4f}%)")
    if cl["experts_with_support"]:
        print(f"  cross-layer: {cl['experts_over_30pct']} of {cl['experts_with_support']} experts "
              f"send >=30% to one next-layer expert; median max-share {100*cl['median_p_max']:.1f}% "
              f"(uniform {100*cl['uniform_p_max_expectation']:.1f}%)")
    print(f"  step overlap: {so['mean_shared_experts']:.2f} of {k} experts shared with the next "
          f"step; uniform {so['uniform_expectation']:.2f}")


# ---------------------------------------------------------------- self-test

def selftest() -> int:
    """Two things must hold: planted structure is detected, and sample size is respected.

    The metric is sample-size sensitive, because the "hottest 0.8% of pairs" are picked by
    observed frequency from the sample itself.  Measured controls (uniform routing, top-6 of
    64): 32 tokens/layer give bucket 5.0x and hottest-pair 13.7x -- Zyphra's numbers are
    reproducible from pure noise there -- while 50,000 tokens/layer give 1.06x and 1.18x.
    The self-test pins both ends, so a later measurement cannot be read without its control.
    """
    k, E = 6, 64
    fails = []

    small = analyze(uniform_control(32, 26, k, E, seed=0), k, E)
    report("control A: uniform, 32 tokens/layer (a 32-step decode)", small, k)
    if small["pair_concentration"]["hottest_pair_ratio"] <= 5.0:
        fails.append("a 32-token-per-layer sample must inflate the hottest pair; control broken")
    if small["cross_layer"]["experts_over_30pct"] == 0:
        fails.append("32 tokens/layer must still produce spurious >=30% experts; control broken")

    big = analyze(uniform_control(50000, 2, k, E, seed=0), k, E)
    report("control B: uniform, 50000 tokens/layer (Zyphra's regime)", big, k)
    pc, cl = big["pair_concentration"], big["cross_layer"]
    if not 0.90 <= pc["ratio_vs_uniform"] <= 1.25:
        fails.append(f"uniform routing must score about 1x at 50k tokens/layer, got {pc['ratio_vs_uniform']:.2f}")
    if pc["hottest_pair_ratio"] > 1.5:
        fails.append(f"uniform routing's hottest pair must be about 1x, got {pc['hottest_pair_ratio']:.2f}")
    if cl["experts_over_30pct"] != 0:
        fails.append("uniform routing must have no expert send >=30% to one next-layer expert")
    if big["step_overlap"]["mean_shared_experts"] > 1.0:
        fails.append("consecutive uniform decode steps must share about k^2/E experts")

    planted = uniform_control(50000, 2, k, E, seed=0).reshape(-1, k).copy()
    planted[::10, 0], planted[::10, 1] = 0, 1                 # 10% of tokens on the pair (0,1)
    rp = analyze(planted.reshape(50000, 2, k), k, E)
    report("planted: 10% of tokens on one pair, 50000 tokens/layer", rp, k)
    if rp["pair_concentration"]["hottest_pair_ratio"] <= 8.0:
        fails.append("a pair that 10% of tokens select must stand out")

    if len(pairs_of(np.array([0, 1, 2, 3, 4, 5]))) != k * (k - 1) // 2:
        fails.append("a top-6 token must give 15 pairs")
    if abs(k * (k - 1) / (E * (E - 1)) - 30 / (64 * 63)) > 1e-12:
        fails.append("baseline formula is wrong")

    for f in fails:
        print("FAIL: " + f)
    print("selftest: " + ("FAILED" if fails else "PASS"))
    return 1 if fails else 0


# ---------------------------------------------------------------- capture

def capture_dsv2(args) -> np.ndarray:
    """Record every MoE gate's top-k ids during a real greedy decode (needs the model on a GPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(args.fleet) / "src" / "host"))
    from reference_run import build_prompt            # fleet's golden prompt construction

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True,
                                                 attn_implementation="eager").to("cuda").eval()
    base = build_prompt(tok, args.context)[0]
    n = base.numel()
    ids = torch.stack([torch.cat([base[(97 * i) % n:], base[:(97 * i) % n]]) for i in range(args.batch)]).to("cuda")

    picks: dict[int, list] = {}
    decoding = {"on": False}

    def hook(layer: int):
        def h(_m, _inp, out):
            if decoding["on"]:
                picks.setdefault(layer, []).append(out[0].view(args.batch, -1).cpu().to(torch.int16).numpy())
        return h

    handles = [lay.mlp.gate.register_forward_hook(hook(i))
               for i, lay in enumerate(model.model.layers) if hasattr(lay.mlp, "gate")]
    with torch.no_grad():
        mask = torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=mask, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        decoding["on"] = True
        for step in range(args.steps):
            mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
            out = model(input_ids=nxt, attention_mask=mask, past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            if step % 8 == 0:
                print(f"step {step}", flush=True)
    for h in handles:
        h.remove()

    # [steps, layers, k] per sequence, stacked over sequences: [batch*steps, layers, k]
    stack = np.stack([np.stack(picks[L]) for L in sorted(picks)], axis=1)   # [steps, layers, batch, k]
    return stack.transpose(2, 0, 1, 3).reshape(-1, stack.shape[1], stack.shape[3])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="path to DeepSeek-V2-Lite (capture mode)")
    ap.add_argument("--fleet", default=str(Path.home() / "code" / "fleet-mi300x"))
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dump", type=Path, help="write the captured ids to this .npz")
    ap.add_argument("--ids", type=Path, help="analyze a previously dumped .npz")
    ap.add_argument("--control-seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.ids:
        z = np.load(a.ids)
        ids, k, E = z["ids"], int(z["top_k"]), int(z["n_routed"])
        print(f"loaded {ids.shape[0]} tokens x {ids.shape[1]} layers, top-{k} of {E} "
              f"from {a.ids}")
    elif a.model:
        ids = capture_dsv2(a)
        cfg = json.loads((Path(a.model) / "config.json").read_text())
        k, E = int(cfg["num_experts_per_tok"]), int(cfg["n_routed_experts"])
        if a.dump:
            np.savez_compressed(a.dump, ids=ids, top_k=k, n_routed=E)
            print(f"wrote {a.dump}")
    else:
        ap.error("give --model (capture), --ids (analyze) or --selftest")

    report("DeepSeek-V2-Lite, measured", analyze(ids, k, E), k)
    ctrl = uniform_control(ids.shape[0], ids.shape[1], k, E, seed=a.control_seed)
    report("same shape, uniform routing (control)", analyze(ctrl, k, E), k)
    print("\nZyphra's config for comparison: top-2 of 128, so one pair's uniform share is "
          f"{2*1/(128*127)*100:.4f}%; at top-{k} of {E} it is "
          f"{k*(k-1)/(E*(E-1))*100:.4f}%, "
          f"{(k*(k-1)/(E*(E-1)))/(2*1/(128*127)):.0f}x higher.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())