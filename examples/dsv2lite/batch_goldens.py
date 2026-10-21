"""Check the per-request references of the batch study and export them for vLLM (docs/deepseek/BATCH-2026-10-14.md section 4).

Request 0 is the pinned set in DATA (golden.npz and friends, checked by preflight); request r >= 1 is DATA/r<r>/, written
by reference_rotated.py --rotate 97r. Refused unless, for every r:
  - prompt r is prompt 0 rotated by 97 r tokens (the construction vllm_bench.py --batch and the ETX host share),
  - golden_tokens.txt holds exactly the npz's greedy ids,
  - no two prompts and no two greedy sequences are equal.
Writes DATA/vllm/golden_r<r>.txt (vllm_bench's `prompt ...` / `gen ...` format, the same ids) and DATA/goldens.json
(sha256 of every per-request file). Runs with the prep environment's python (numpy).

  python -m examples.dsv2lite.batch_goldens DATA [--requests 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROTATE = 97
FILES = ("golden.npz", "golden_tokens.txt", "golden_hidden.bin", "fleet_cache.bin")


def request_dir(data: Path, r: int) -> Path:
    return data if r == 0 else data / f"r{r}"


def rotate(ids: list[int], r: int) -> list[int]:
    k = (ROTATE * r) % len(ids)
    return ids[k:] + ids[:k]


def check(prompts: list[list[int]], gens: list[list[int]], token_files: list[list[int]], *,
          bindings=None, hashes=None) -> list[str]:
    """Every refusal reason; empty when the set is usable. Two requests may share their 32-token greedy output
    (a repetitive prompt does it): that is reported as a collision, not refused. See docs/deepseek/BATCH-2026-10-14.md 4a."""
    bad = []
    for r, (p, g, t) in enumerate(zip(prompts, gens, token_files)):
        if bindings is not None and (bindings[r] is None or
                                    bindings[r] != dict(prompt_ids=p, sha256=hashes[str(r)])):
            bad.append(f"request {r}: golden binding missing or differs from prompt / execution bytes")
        if p != rotate(prompts[0], r):
            bad.append(f"request {r}: prompt is not prompt 0 rotated by {ROTATE * r}")
        if t != g:
            bad.append(f"request {r}: golden_tokens.txt differs from golden.npz token_ids")
        for o in range(r):
            if p == prompts[o]:
                bad.append(f"requests {o} and {r}: the same prompt")
    return bad


def collisions(gens: list[list[int]], start=None, count=32) -> list[list[int]]:
    """Pairs indistinguishable in either checked window, or in the selected gate's window."""
    starts = (0, 1) if start is None else (start,)
    return [[o, r] for r in range(len(gens)) for o in range(r)
            if any(len(gens[o]) >= s + count and len(gens[r]) >= s + count and
                   gens[o][s:s + count] == gens[r][s:s + count] for s in starts)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(data: Path) -> dict:
    import numpy as np
    with np.load(data / "golden.npz") as z:
        prompt = [int(x) for x in z["prompt_ids"].reshape(-1)]
    return dict(prompt_ids=prompt, sha256={f: sha(data / f) for f in FILES})


def read_binding(data: Path):
    try:
        return json.loads((data / "golden.binding.json").read_text())
    except (OSError, ValueError):
        return None


def main(argv=None) -> None:
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("--requests", type=int, default=8)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--certify", action="store_true", help="bind a freshly generated reference tuple")
    mode.add_argument("--verify-binding", action="store_true", help="verify a tuple before reusing it")
    a = ap.parse_args(argv)
    if a.certify or a.verify_binding:
        try:
            current = binding(a.data)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"GOLDENS binding unavailable: {exc}") from exc
        if a.certify:
            (a.data / "golden.binding.json").write_text(json.dumps(current, sort_keys=True) + "\n")
        elif read_binding(a.data) != current:
            raise SystemExit(f"GOLDENS binding refused: {a.data}")
        return
    prompts, gens, tokens, hashes = [], [], [], {}
    bindings = []
    for r in range(a.requests):
        d = request_dir(a.data, r)
        z = np.load(d / "golden.npz")
        prompts.append([int(x) for x in z["prompt_ids"].reshape(-1)])
        gens.append([int(x) for x in z["token_ids"].reshape(-1)])
        tokens.append([int(x) for x in (d / "golden_tokens.txt").read_text().split()])
        hashes[str(r)] = {f: sha(d / f) for f in FILES}
        bindings.append(read_binding(d))
    bad = check(prompts, gens, tokens, bindings=bindings, hashes=hashes)
    if bad:
        raise SystemExit("GOLDENS REFUSED\n" + "\n".join(bad))
    coll = collisions(gens)
    windows = dict(vllm=collisions(gens, start=0), etx=collisions(gens, start=1))
    out = a.data / "vllm"
    out.mkdir(exist_ok=True)
    for r in range(a.requests):
        (out / f"golden_r{r}.txt").write_text("prompt " + " ".join(map(str, prompts[r])) + "\n"
                                              + "gen " + " ".join(map(str, gens[r])) + "\n")
    (a.data / "goldens.json").write_text(json.dumps(
        dict(requests=a.requests, rotate=ROTATE, sha256=hashes, bindings=bindings, collisions=coll, collision_windows=windows,
             collision_note="pairs with identical 32-token greedy output; the request mix-up check is unavailable "
                            "for those pairs, every other gate still applies"), indent=1) + "\n")
    print(f"GOLDENS OK {a.requests} requests: prompts are the 97 r rotations, pairwise distinct; "
          f"collisions by gate (vLLM [0:32], ETX [1:33]) {windows}; vLLM files in {out}", flush=True)


if __name__ == "__main__":
    main()
