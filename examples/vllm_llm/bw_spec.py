#!/usr/bin/env python3
"""Make a timeline byte spec from the plan and the decode-step weight ledger.

Example:
  python -m examples.vllm_llm.bw_spec PLAN --config examples/llm/configs/qwen3-30b-a3b.json --context 1055 --output bytes.json

The values cover model weights and KV traffic. They exclude activation and metadata
traffic. An absent rule means that the ledger has no byte count for that phase.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from etx.tools.timeline import load_plan
from examples.vllm_llm import byte_ledger as bl


def spec(plan: dict, config: str | Path, context: int, maxlen: int = 1072) -> dict[str, float]:
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
    return rules


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--config", required=True, help="model config.json")
    ap.add_argument("--context", type=int, required=True, help="KV positions read in the traced step")
    ap.add_argument("--maxlen", type=int, default=1072, help="packed cache length (default 1072)")
    ap.add_argument("--output", help="JSON output path; default: standard output")
    a = ap.parse_args(argv)
    try:
        rules = spec(load_plan(a.plan), a.config, a.context, a.maxlen)
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
