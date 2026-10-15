"""How many distinct experts do concurrent requests touch? The input of the batch2 / batch4 estimate in whatif.py
(which assumes uniform routing). DeepSeek-V2-Lite, HF reference model on one GPU, B sequences decoded together:
sequence i is fleet's 1024-token prompt rotated by 97*i tokens (as examples/llm/vllm_bench.py --batch does), greedy.
Every MoE gate's top-6 per sequence per decode step is recorded; the report gives, per layer and overall, the mean
number of distinct routed experts for 1, 2 and 4 sequences of the same step.

Run with fleet's environment (transformers < 5, the flash_attn stub):
  ~/fleet-mi300x/.venv/bin/python examples/dsv2lite/routing_overlap.py --model ~/models/dsv2-lite-base --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path


def validate_prefix(generated: list[list[int]], expected: list[list[int]], steps: int) -> None:
    if len(generated) != len(expected) or any(g[:steps+1] != e[:steps+1] or len(g) != steps+1
                                             for g, e in zip(generated, expected)):
        raise ValueError("routing greedy trajectory differs from the bound timed prefix")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fleet", default=str(Path.home() / "fleet-mi300x"))
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--json", default=None)
    ap.add_argument("--reference-data", type=Path, help="use and verify producer-bound executed prompts/token prefix")
    a = ap.parse_args()
    if a.reference_data and (a.batch not in (1,2,4) or a.steps != 32 or a.context != 1024):
        ap.error("bound attribution requires B1/B2/B4, context 1024, 32 steps")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(a.fleet) / "src" / "host"))
    from reference_run import build_prompt          # fleet's prompt construction (the golden prompt)

    if a.reference_data:
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="eager").to("cuda").eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    bound = None
    if a.reference_data:
        from examples.dsv2lite.reference_prepare import verify, recipe
        from examples.dsv2lite.pinned import fleet_export
        from examples.dsv2lite.batch_goldens import read_binding, sha
        from examples.llm.vllm_bench import read_golden
        bound = []
        expected_recipe = recipe(fleet_export(Path(a.fleet)), Path(a.model), 0)
        for r in range(a.batch):
            d = a.reference_data if r == 0 else a.reference_data / f'r{r}'
            receipt = json.loads((d / 'golden.producer.json').read_text())
            verify(d, dict(expected_recipe, rotation=97*r))
            prompt, tokens = read_golden(a.reference_data / 'vllm' / f'golden_r{r}.txt')
            if prompt != read_binding(d)['prompt_ids'] or tokens != [int(x) for x in (d/'golden_tokens.txt').read_text().split()]:
                raise ValueError('routing exported prompt/tokens differ from producer-bound tuple')
            if len(prompt) != a.context or len(tokens) < a.steps+1:
                raise ValueError('routing reference has incomplete timed prefix')
            bound.append((prompt, tokens))
    base = build_prompt(tok, a.context)[0]
    n = base.numel()
    ids = torch.stack([torch.cat([base[(97 * i) % n:], base[:(97 * i) % n]]) for i in range(a.batch)]).to("cuda")

    if bound:
        ids = torch.tensor([p for p, _ in bound], dtype=torch.long, device='cuda')
    generated = []

    picks: dict[int, list] = {}                      # layer -> list over decode steps of [B, k] expert ids
    decoding = {"on": False}

    def hook(layer):
        def h(_m, _inp, out):
            if decoding["on"]:
                picks.setdefault(layer, []).append(out[0].view(a.batch, -1).cpu().tolist())
        return h

    handles = [lay.mlp.gate.register_forward_hook(hook(i)) for i, lay in enumerate(model.model.layers) if hasattr(lay.mlp, "gate")]
    with torch.no_grad():
        mask = torch.ones_like(ids)                  # the remote model code asserts a mask; no padding (equal lengths)
        out = model(input_ids=ids, attention_mask=mask, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(nxt.flatten().cpu().tolist())
        decoding["on"] = True
        for step in range(a.steps):
            mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
            out = model(input_ids=nxt, attention_mask=mask, past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(nxt.flatten().cpu().tolist())
            if step % 8 == 0:
                print(f"step {step}", flush=True)
    for h in handles:
        h.remove()

    if bound:
        validate_prefix([list(x) for x in zip(*generated)], [g for _, g in bound], a.steps)
    if len(picks) != 26 or any(len(v) != a.steps for v in picks.values()):
        raise ValueError('routing needs all 26 MoE layers on every decode step')

    def mean_union(group_size: int, layer_steps: list, *, subsets=False) -> float:
        vals = []
        for step in layer_steps:
            for g in (combinations(range(a.batch), group_size) if subsets else [range(group_size)]):
                vals.append(len(set().union(*(set(step[i]) for i in g))))
        return sum(vals) / len(vals)

    per_layer = {L: {s: mean_union(s, st, subsets=True) for s in (1, 2, 4, 8) if s <= a.batch} for L, st in sorted(picks.items())}
    overall = {s: sum(v[s] for v in per_layer.values()) / len(per_layer) for s in (1, 2, 4, 8) if s <= a.batch}
    prefix_layers = {L: {s: mean_union(s, st) for s in overall} for L, st in sorted(picks.items())}
    executed_prefix = {s: sum(v[s] for v in prefix_layers.values()) / len(prefix_layers) for s in overall}
    k = len(next(iter(picks.values()))[0][0])
    n_routed = model.config.n_routed_experts
    uniform = {s: n_routed * (1 - (1 - k / n_routed) ** s) for s in overall}
    print(f"MoE layers {len(per_layer)}, top-{k} of {n_routed}, {a.steps} decode steps x {a.batch} sequences")
    for s in overall:
        print(f"ROUTING distinct routed experts for {s} sequence(s): measured {overall[s]:.2f}, uniform-routing estimate {uniform[s]:.2f}")
    for L, v in per_layer.items():
        print(f"  layer {L:2d}: " + "  ".join(f"{s} seq {x:5.2f}" for s, x in v.items()))
    if a.json:
        Path(a.json).write_text(json.dumps({"overall": overall, "uniform": uniform, "per_layer": per_layer, "top_k": k,
                                            "n_routed": n_routed, "steps": a.steps, "batch": a.batch,
                                            "executed_prefix": executed_prefix, "prefix_per_layer": prefix_layers,
                                            "picks": picks, "bound_prefix": bool(bound),
                                            "prefix_window": "ETX input golden[0:32], output golden[1:33]; vLLM output golden[0:32]",
                                            "reference_sha256": ({str(r): sha((a.reference_data if r == 0 else a.reference_data/f'r{r}')/'golden.producer.json')
                                                                   for r in range(a.batch)} if bound else {}),
                                            "limitation": "HF routing on bound tokens; ETX routing equality is not directly measured"}, indent=1))


if __name__ == "__main__":
    main()
