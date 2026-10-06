#!/usr/bin/env python3
"""Bytes one decode step reads, layer by layer, from a vl.manifest, and the bandwidth floor they imply.

  python -m examples.vllm_llm.byte_ledger --manifest ~/llm/phi-3.5-moe/vl.manifest --batch 1 --context 1024
  python -m examples.vllm_llm.byte_ledger --config examples/llm/configs/phi-3.5-moe.json --batch 1 --context 1024

--manifest reads the file prep.py wrote next to vl.bin (cfg lines + one `t <name> <offset> <numel>` line per tensor).
--config derives the same tensor list from the HF config with prep.py's layout rules, for a machine without the
packed weights; the output says which source it used. Tests check the derivation against the byte totals of three
packed manifests (prep logs in results_2026-09-27 and results_2026-09-28b).

Counted per step: every weight a decode step reads once (norms, biases, qkv, o, router, lm_head in full; of the
expert tensors only the active experts' share; one embedding row and one rotary row per token), plus the KV cache:
`context` positions read and one position written per sequence and layer, bf16 like host.hip's caches. Activations
(KB per layer) are not counted. The floor divides by 4.3 TB/s, the achievable MI300X rate AGENTS.md section 2 assumes:
it is an estimate, not a measurement.

Refuses, naming the input, when a cfg key or a tensor the step reads is missing, when a tensor has no byte rule, and
when the active expert count is not determined (batch > 1 of an MoE model needs a measured --active-experts).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

BW_TBPS = 4.3
BF16 = 2
ALIGN = 256          # prep.py pads every tensor to a 256-byte offset

# per-layer tensor -> ledger column; experts are scaled by the active share, the rest read in full
LAYER_COLUMNS = {"ln1": "norms", "ln2": "norms", "ln1b": "norms", "ln2b": "norms", "qn": "norms", "kn": "norms",
                 "wqkv": "attn_w", "bqkv": "attn_w", "wo": "attn_w", "bo": "attn_w",
                 "wr": "router", "w13": "experts", "w2": "experts", "wgu": "mlp", "wd": "mlp"}
GLOBALS = {"embed", "final_norm", "final_norm_b", "lm_head", "lm_head_b", "cos_sin"}
COLUMNS = ("norms", "attn_w", "router", "experts", "mlp", "kv_read", "kv_write")


class Refusal(Exception):
    """An input the floor depends on is missing or not covered by a rule."""


def read_manifest(text: str) -> tuple[dict, dict]:
    cfg, tensors = {}, {}
    for n, line in enumerate(text.splitlines(), 1):
        f = line.split()
        if not f:
            continue
        if f[0] == "cfg" and len(f) == 3:
            cfg[f[1]] = int(f[2])
        elif f[0] == "cfgf" and len(f) == 3:
            cfg[f[1]] = float(f[2])
        elif f[0] == "t" and len(f) == 4:
            tensors[f[1]] = int(f[3])
        else:
            raise Refusal(f"manifest line {n} is not a cfg, cfgf or t line: {line!r}")
    return cfg, tensors


def manifest_from_config(config: str | Path, maxlen: int = 1072) -> str:
    """The vl.manifest prep.py would write for this config, without the weights (presence from the config flags)."""
    if os.environ.get("ETX_LLM_LAYERS"):
        raise Refusal("ETX_LLM_LAYERS is set: load_dims would truncate the model")
    from examples.llm.model import load_dims
    d = load_dims(str(config))
    lines, off = [], 0

    def put(name, numel):
        nonlocal off
        off += (-off) % ALIGN
        lines.append(f"t {name} {off} {numel}")
        off += numel * BF16

    put("embed", d.V * d.H)
    put("final_norm", d.H)
    if not d.tie:
        put("lm_head", d.V * d.H)
    put("cos_sin", maxlen * d.HD)
    if not d.tie and d.bias_lm:
        put("lm_head_b", d.V)
    if d.ln_norm:
        put("final_norm_b", d.H)
    for L in range(d.L):
        p = f"L{L}."
        put(p + "ln1", d.H); put(p + "ln2", d.H)
        if d.ln_norm:
            put(p + "ln1b", d.H); put(p + "ln2b", d.H)
        put(p + "wqkv", d.NQKV * d.H)
        if d.bias:
            put(p + "bqkv", d.NQKV)
        if d.qk_norm:
            put(p + "qn", d.HD); put(p + "kn", d.HD)
        put(p + "wo", d.H * d.NH * d.HD)
        if d.bias_o:
            put(p + "bo", d.H)
        if d.moe:
            put(p + "wr", d.E * d.H); put(p + "w13", d.E * 2 * d.MI * d.H); put(p + "w2", d.E * d.H * d.MI)
        else:
            put(p + "wgu", 2 * d.INTER * d.H); put(p + "wd", d.H * d.INTER)
    cfg = {"H": d.H, "NH": d.NH, "NKV": d.NKV, "HD": d.HD, "INTER": d.INTER, "L": d.L, "V": d.V, "tie": int(d.tie),
           "bias": int(d.bias), "qk_norm": int(d.qk_norm), "maxlen": maxlen, "bytes": off, "moe": int(d.moe), "E": d.E,
           "TOPK": d.TOPK, "MI": d.MI}
    return "\n".join([f"cfg {k} {v}" for k, v in cfg.items()] + lines) + "\n"


def need(cfg: dict, *keys: str) -> list[int]:
    for k in keys:
        if k not in cfg:
            raise Refusal(f"missing input: cfg {k}")
    return [cfg[k] for k in keys]


def ledger(cfg: dict, tensors: dict, batch: int, context: int, active_experts: int | None = None) -> dict:
    """Bytes of one decode step: {'layers': [{column: bytes}], 'globals': {name: bytes}, totals}."""
    L, H, NKV, HD, V, maxlen, moe, tie = need(cfg, "L", "H", "NKV", "HD", "V", "maxlen", "moe", "tie")
    if batch < 1:
        raise Refusal(f"batch {batch} must be at least 1")
    if not 1 <= context <= maxlen:
        raise Refusal(f"context {context} is outside 1..maxlen {maxlen}: the KV cache holds maxlen positions")
    active = None
    if moe:
        E, TOPK = need(cfg, "E", "TOPK")
        if active_experts is not None:
            if not TOPK <= active_experts <= min(E, batch * TOPK):
                raise Refusal(f"active experts {active_experts} outside {TOPK}..{min(E, batch * TOPK)}")
            active = active_experts
        elif batch == 1:
            active = TOPK              # one token selects TOPK distinct experts
        else:
            raise Refusal(f"missing input: active experts per layer at batch {batch} (the measured expert union)")
    for name in tensors:
        m = re.fullmatch(r"L(\d+)\.(\w+)", name)
        if m:
            if int(m.group(1)) >= L or m.group(2) not in LAYER_COLUMNS:
                raise Refusal(f"tensor {name} has no byte rule")
        elif name not in GLOBALS:
            raise Refusal(f"tensor {name} has no byte rule")
    required = ["embed", "final_norm", "cos_sin"] + ([] if tie else ["lm_head"])
    layer_required = ["ln1", "ln2", "wqkv", "wo"] + (["wr", "w13", "w2"] if moe else ["wgu", "wd"])
    for name in required + [f"L{l}.{t}" for l in range(L) for t in layer_required]:
        if name not in tensors:
            raise Refusal(f"missing input: tensor {name}")
    if tensors["embed"] != V * H:
        raise Refusal(f"tensor embed has {tensors['embed']} elements, not V * H = {V * H}")
    if tensors["cos_sin"] % maxlen:
        raise Refusal(f"tensor cos_sin has {tensors['cos_sin']} elements, not a multiple of maxlen {maxlen}")

    glob = {"embed rows": batch * H * BF16, "cos_sin rows": batch * (tensors["cos_sin"] // maxlen) * BF16,
            "final_norm": tensors["final_norm"] * BF16 + tensors.get("final_norm_b", 0) * BF16,
            "lm_head": (tensors["embed"] if tie else tensors["lm_head"] + tensors.get("lm_head_b", 0)) * BF16}
    kv_pos = 2 * NKV * HD * BF16
    layers = []
    for l in range(L):
        row = dict.fromkeys(COLUMNS, 0)
        for t, col in LAYER_COLUMNS.items():
            n = tensors.get(f"L{l}.{t}")
            if n is None:
                continue
            if col == "experts":
                if n % cfg["E"]:
                    raise Refusal(f"tensor L{l}.{t} has {n} elements, not a multiple of E = {cfg['E']}")
                n = n // cfg["E"] * active
            row[col] += n * BF16
        row["kv_read"] = batch * context * kv_pos
        row["kv_write"] = batch * kv_pos
        layers.append(row)
    weights = sum(glob.values()) + sum(r[c] for r in layers for c in COLUMNS if not c.startswith("kv"))
    kv = sum(r["kv_read"] + r["kv_write"] for r in layers)
    total = weights + kv
    return {"layers": layers, "globals": glob, "weights": weights, "kv": kv, "total": total, "active": active,
            "batch": batch, "context": context, "floor_ms": total / (BW_TBPS * 1e12) * 1e3}


def report(r: dict, source: str, cfg: dict, out=None) -> None:
    out = out or sys.stdout
    mb = 1e6
    act = f", active experts {r['active']} of {cfg['E']} per layer" if r["active"] is not None else ""
    print(f"byte ledger: {source}", file=out)
    print(f"conditions: batch {r['batch']}, context {r['context']} KV positions read per sequence and layer, "
          f"bf16 weights and KV{act}", file=out)
    cols = [c for c in COLUMNS if any(row[c] for row in r["layers"])]
    print("layer " + " ".join(f"{c:>10s}" for c in cols) + f" {'total':>10s}   (MB)", file=out)
    for l, row in enumerate(r["layers"]):
        print(f"{l:5d} " + " ".join(f"{row[c] / mb:10.3f}" for c in cols) + f" {sum(row.values()) / mb:10.3f}", file=out)
    print("globals: " + ", ".join(f"{k} {v / mb:.3f} MB" for k, v in r["globals"].items()), file=out)
    print(f"TOTAL {r['total'] / 1e9:.4f} GB per decode step = weights {r['weights'] / 1e9:.4f} GB + KV {r['kv'] / 1e9:.4f} GB",
          file=out)
    per_token = f" = {r['floor_ms'] / r['batch']:.3f} ms per generated token" if r["batch"] > 1 else ""
    print(f"FLOOR {r['floor_ms']:.3f} ms/token (one decode step at {BW_TBPS} TB/s{per_token}) "
          f"[estimate: assumed achievable bandwidth, not measured]", file=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m examples.vllm_llm.byte_ledger", description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="vl.manifest written by examples/vllm_llm/prep.py")
    src.add_argument("--config", help="HF config.json: derive the manifest with prep.py's layout rules")
    ap.add_argument("--maxlen", type=int, default=1072, help="with --config: prep.py --maxlen (default 1072)")
    ap.add_argument("--batch", type=int, required=True, help="sequences in one decode step")
    ap.add_argument("--context", type=int, required=True, help="KV positions attention reads per sequence")
    ap.add_argument("--active-experts", type=int, help="distinct experts per layer per step (measured; batch > 1)")
    a = ap.parse_args(argv)
    try:
        if a.manifest:
            path = Path(a.manifest).expanduser()
            if not path.is_file():
                raise Refusal(f"missing input: manifest {path}")
            text, source = path.read_text(), f"{path} (packed manifest)"
        else:
            path = Path(a.config).expanduser()
            if not path.is_file():
                raise Refusal(f"missing input: config {path}")
            text = manifest_from_config(path, a.maxlen)
            source = f"{path} (derived with prep.py's layout rules, maxlen {a.maxlen}; not a packed manifest)"
        cfg, tensors = read_manifest(text)
        r = ledger(cfg, tensors, a.batch, a.context, a.active_experts)
    except Refusal as e:
        print(f"REFUSED: {e}. No floor printed.")
        return 2
    report(r, source, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
