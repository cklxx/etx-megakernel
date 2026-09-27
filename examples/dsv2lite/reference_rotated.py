"""fleet's HF reference run for another request: the golden prompt rotated by 97 * r tokens (the construction
examples/llm/vllm_bench.py --batch uses for sequence r), so the interleaved build decodes two different requests.
Writes golden.npz, fleet_cache.bin, golden_tokens.txt and golden_hidden.bin into --out's directory, exactly as fleet's
src/host/reference_run.py does for request 0. The fleet repository is not modified: its build_prompt is wrapped.

  ~/fleet-mi300x/.venv/bin/python examples/dsv2lite/reference_rotated.py --fleet ~/fleet-mi300x --rotate 97 \
      -- --model ~/models/dsv2-lite-base --out ~/fleet-mi300x/build_r1/golden.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", default=str(Path.home() / "fleet-mi300x"))
    ap.add_argument("--rotate", type=int, default=97)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then reference_run.py's own arguments")
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    sys.path.insert(0, str(Path(a.fleet) / "src" / "host"))
    import reference_run as rr

    orig = rr.build_prompt

    def rotated(tokenizer, n_tokens):
        ids = orig(tokenizer, n_tokens)
        k = a.rotate % ids.shape[1]
        import torch
        out = torch.cat([ids[:, k:], ids[:, :k]], dim=1)
        print(f"prompt rotated by {k} tokens (first ids {out[0, :4].tolist()})")
        return out

    rr.build_prompt = rotated
    out = Path(rest[rest.index("--out") + 1]) if "--out" in rest else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = ["reference_run.py"] + rest
    rr.main()


if __name__ == "__main__":
    main()
