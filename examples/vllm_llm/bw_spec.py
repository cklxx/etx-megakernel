#!/usr/bin/env python3
"""Make a timeline byte spec from the plan and the decode-step weight ledger.

Example:
  python -m examples.vllm_llm.bw_spec PLAN --config examples/llm/configs/qwen3-30b-a3b.json --context 1055 --argmax-write-next 1 --output bytes.json

The values retain the weight and KV ledger. Extra rules count logical payload bytes
for topk, reduce, and argmax. They do not count cache transactions or duplicate loads.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from etx.tools.timeline import load_plan
from examples.vllm_llm import byte_ledger as bl


def payload_bytes(plan: dict, cfg: dict, context: int, write_next: bool | None) -> dict[str, int]:
    """Count logical payload bytes for the supported batch-one host layouts.

    The plan lacks tensor shapes and runtime arguments. The config supplies shapes.
    Kernel symbols select the host layout. Unknown layouts fail closed.
    """
    counts = Counter(d["type"] for d in plan["descs"])
    result = {}
    for name, ty in plan["types"].items():
        n = counts[ty["id"]]
        if not n:
            continue
        if re.fullmatch(r"topk_\d+_x", name):
            if ty["symbol"] != "etx_tile_rep_dsl_topk":
                raise bl.Refusal(f"{name}: missing input: payload layout for {ty['symbol']}")
            E, K = bl.need(cfg, "E", "TOPK")
            # Each replica reads bf16 logits and writes float weights plus two int arrays.
            result[name] = n * (2 * E + 12 * K)
        elif re.fullmatch(r"reduce_\d+", name):
            NH, NKV, HD = bl.need(cfg, "NH", "NKV", "HD")
            attn = plan["types"].get(name.replace("reduce_", "attn_"), {})
            if (ty["symbol"] != "etx_tile_attn_reduce" or n != NKV
                    or attn.get("symbol") != f"etx_tile_attn_mfma16_g{NH // NKV}"):
                raise bl.Refusal(f"{name}: missing input: attention partition and head-slot layout")
            slots = counts[attn["id"]]
            parts = (context + 255) // 256  # This imported kernel has a fixed 256-token partition.
            if slots % NKV or slots // NKV < parts:
                raise bl.Refusal(f"{name}: attention grid cannot cover context {context}")
            # bf16 partials, two float statistics, bf16 output; count each unique element once.
            result[name] = NH * (parts * (2 * HD + 8) + 2 * HD)
        elif name == "argmax":
            if ty["symbol"] != "vl_argmax" or n != 1:
                raise bl.Refusal("argmax: missing input: native payload layout")
            if write_next is None:
                raise bl.Refusal("argmax: missing input: write_next runtime flag; use --argmax-write-next 0 or 1")
            V, = bl.need(cfg, "V")
            # bf16 logits, token_out, optional token_in, and the end timestamp.
            result[name] = 2 * V + 4 + 4 * int(write_next) + 8
    return result


def spec(plan: dict, config: str | Path, context: int, maxlen: int = 1072,
         *, write_next: bool | None = None) -> dict[str, float]:
    cfg, tensors = bl.read_manifest(bl.manifest_from_config(config, maxlen))
    if int(plan["devices"]) != 1:
        raise bl.Refusal("this byte spec requires a single-device plan")
    names = {t["id"]: n for n, t in plan["types"].items()}
    counts = Counter(names[d["type"]] for d in plan["descs"])
    r = bl.ledger(cfg, tensors, batch=1, context=context)
    known: dict[str, int] = {"prologue": r["globals"]["embed rows"],
                             "lm_head": r["globals"]["lm_head"] + r["globals"]["final_norm"]}
    for layer, row in enumerate(r["layers"]):
        p = f"L{layer}."
        # The phase names and their folds come from this plan, not the current build flags.
        folded_qkv = layer > 0 and f"in_norm_{layer}_x" not in counts
        folded_router = f"post_norm_{layer}_x" not in counts
        norm1 = (tensors[p + "ln1"] + tensors.get(p + "ln1b", 0)) * bl.BF16
        norm2 = (tensors[p + "ln2"] + tensors.get(p + "ln2b", 0)) * bl.BF16
        if layer == 0 or not folded_qkv:
            known[f"in_norm_{layer}_x"] = norm1
        if not folded_router:
            known[f"post_norm_{layer}_x"] = norm2
        known[f"qkv_{layer}"] = ((tensors[p + "wqkv"] + tensors.get(p + "bqkv", 0)) * bl.BF16
                                    + (norm1 if folded_qkv else 0))
        known[f"prep_{layer}"] = ((tensors.get(p + "qn", 0) + tensors.get(p + "kn", 0)) * bl.BF16
                                   + (r["globals"]["cos_sin rows"] if layer == 0 else 0) + row["kv_write"])
        known[f"attn_{layer}"] = row["kv_read"]
        known[f"o_proj_{layer}"] = (tensors[p + "wo"] + tensors.get(p + "bo", 0)) * bl.BF16
        known[f"router_{layer}"] = row["router"] + (norm2 if folded_router else 0)
        known[f"moe_up_{layer}"] = tensors[p + "w13"] // cfg["E"] * r["active"] * bl.BF16
        known[f"moe_down_{layer}"] = tensors[p + "w2"] // cfg["E"] * r["active"] * bl.BF16
    # Do not assign bytes to a name that has no task. No missing weight phase is silent.
    absent = [n for n in known if n not in counts]
    if absent:
        raise bl.Refusal(f"plan lacks byte-bearing phase {absent[0]}")
    rules = {f"^{re.escape(n)}$": v / counts[n] for n, v in known.items()}
    if sum(known.values()) != r["total"]:
        raise bl.Refusal(f"plan phase bytes {sum(known.values())} differ from ledger {r['total']}")
    extra = payload_bytes(plan, cfg, context, write_next)
    rules.update({f"^{re.escape(n)}$": v / counts[n] for n, v in extra.items()})
    return rules


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--config", required=True, help="model config.json")
    ap.add_argument("--context", type=int, required=True, help="KV positions read in the traced step")
    ap.add_argument("--maxlen", type=int, default=1072, help="packed cache length (default 1072)")
    ap.add_argument("--argmax-write-next", type=int, choices=(0, 1),
                    help="runtime flag from the host; required for native argmax")
    ap.add_argument("--output", help="JSON output path; default: standard output")
    a = ap.parse_args(argv)
    try:
        rules = spec(load_plan(a.plan), a.config, a.context, a.maxlen, write_next=a.argmax_write_next)
    except bl.Refusal as e:
        ap.error(str(e))
    text = json.dumps(rules, indent=2) + "\n"
    if a.output:
        Path(a.output).write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
