"""Prepare one model for examples/llm: download, Hugging Face greedy reference, packed weights.

  python examples/llm/prep.py --model Qwen/Qwen3-8B --out ~/llm/qwen3-8b [--gen 32] [--prompt "..."]

Writes <out>/config.json, weights.bin (bf16, 256-byte aligned tensors in the layout the tiles read),
llm.manifest (cfg and tensor lines for the host) and golden.txt (prompt ids, HF greedy ids, HF top-1
margins at each generated position). Needs torch (ROCm), transformers, safetensors, huggingface_hub.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from etx.external import AssetPreparationRequest, prepare_model_assets   # noqa: E402

# Implementation: etx/models/llm/prep.py behind etx.external.prepare_model_assets.
PROMPT = ("The history of the transistor begins in 1947 at Bell Labs, where John Bardeen, Walter Brattain "
          "and William Shockley")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--src", default=None, help="local snapshot dir (default: download to <out>/hf)")
    ap.add_argument("--context", type=int, default=0, help="use fleet's benchmark prompt of this many tokens; writes golden_c<N>.txt")
    ap.add_argument("--sliced", action="store_true", help="also pack the per-XCD slices (sliced.bin / sliced.manifest)")
    ap.add_argument("--rotate", type=int, default=0, help="rotate the --context prompt by N tokens (a second request); golden_c<N>_r<rot>.txt")
    ap.add_argument("--golden-only", action="store_true", help="write the golden file only (no packing)")
    ap.add_argument("--prefill-only", action="store_true",
                    help="write the coverage reference (prefill_*.txt) for an EXISTING golden file, then exit")
    a = ap.parse_args()
    if a.rotate and not a.context:
        raise SystemExit("--rotate rotates the --context prompt: give --context too")
    prepare_model_assets(AssetPreparationRequest("llm", str(ROOT), vars(a)))


if __name__ == "__main__":
    main()
