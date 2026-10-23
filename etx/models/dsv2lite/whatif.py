"""What-if estimates for re-laying out the DeepSeek-V2-Lite MoE layer (the ETX schedule work, docs/qwen/PLAN-v3.md 4.4).

Starts from fleet's measured layer-5 timeline (129.3 us: five latency-bound phases, 74 us, then the expert stream,
59 us) and asks what each layout would give, with every unmeasured input either read from the overlap / sync
microbenchmark logs of the next GPU session or swept in a sensitivity table. It is arithmetic on measured phase
times, not a simulator; its job is to say which measurement decides which layout, and at what value it pays.

  python -m examples.dsv2lite.whatif [--overlap ~/vlnext/overlap.log] [--event-us 1.4]

Layouts (per MoE layer; 26 of the 27 layers are MoE):
  base       fleet's layout, reproduced from the phase table
  shared     the two shared experts (34.6 MB) stream during the router window (their input, the normed h, is ready
             after the router's prologue), from a second workgroup per CU; the six routed units are spread over all
             eight XCDs, which puts one cross-XCD event between gate_up and down
  interleave two requests offset by half a layer: A's latency phases run while B streams its experts and vice versa,
             one workgroup per request per CU (needs the megakernel at <= 256 registers per lane, today 286)
  batch2     two tokens per weight read (tiles decode two sequences at once): latency phases unchanged, the expert
             stream reads the union of both tokens' experts
Unmeasured inputs: f = how much slower a latency-bound fold gets while the same CUs stream (overlap_bench fold
place=same), the cross-XCD event cost under load, and the Infinity Cache read rate (overlap_bench stream).
Assumption: every latency-bound phase slows by the fold's f (they are all short dependent loads under load); the
routing is uniform for the batch2 union (real routing is skewed, which shrinks the union and favours batch2).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from etx.models.declarations import declaration as _declaration
from . import paths

MOE_INTER = _declaration(paths.ROOT, "dsv2lite.bytes").MOE_INTER
HIDDEN = _declaration(paths.ROOT, "dsv2lite.bytes").HIDDEN
TOPK = _declaration(paths.ROOT, "dsv2lite.bytes").TOPK
SHARED = _declaration(paths.ROOT, "dsv2lite.bytes").SHARED
per_layer = _declaration(paths.ROOT, "dsv2lite.bytes").per_layer

# fleet v0.16, layer 5 (docs/STATUS.md; results/timeline_v16_default_L5.txt): start, end, prologue (us)
PHASES = [("q/kv_a", 0.0, 19.3, 7.6), ("attention", 20.2, 40.8, 10.5), ("merge+W_UV", 39.8, 51.1, 0.0),
          ("o_proj", 49.7, 59.7, 0.7), ("norm+router", 60.4, 73.2, 7.6), ("gate_up", 72.4, 108.9, 5.7),
          ("down", 106.6, 129.3, 1.6)]
LAYER_US = 129.3
TOKEN_MS = 3.60             # fleet's shipped figure; 26 MoE layers + the dense layer + lm_head + small
MOE_LAYERS = 26
BW = 4.3                    # TB/s achievable
UNIT_MB = (2 * MOE_INTER * HIDDEN + MOE_INTER * HIDDEN) * 2 / 1e6      # one expert, gate_up + down: 17.3 MB


def latency_us() -> float:
    return PHASES[4][2] - PHASES[0][1]            # q/kv_a start .. router end: 73.2


def expert_us() -> float:
    return PHASES[6][2] - PHASES[4][2]            # router end .. down end: 56.1 (the phases overlap a little)


def routed_union_units(tokens: int, n_routed: int = 64, k: int = TOPK) -> float:
    """Expected distinct routed experts for `tokens` independent top-k draws (uniform routing: an estimate)."""
    return n_routed * (1.0 - (1.0 - k / n_routed) ** tokens)


# ------------------------------------------------------------------ layouts
def base() -> float:
    return LAYER_US


def shared_layout(f: float, event_us: float, bg_tbs: float) -> float:
    """Shared experts in the router window. The window left after the router's prologue is 12.8 - 7.6 = 5.2 us; the
    second workgroup streams shared bytes then at bg_tbs, the rest of the shared bytes stay in the expert phase. The
    router prologue slows by f (it runs while the same CUs stream). The expert phase then carries fewer bytes spread
    over all eight XCDs, plus one cross-XCD event."""
    lat = latency_us()
    window = PHASES[4][2] - PHASES[4][1] - PHASES[4][3]
    shared_mb = SHARED * UNIT_MB
    moved = min(shared_mb, window * bg_tbs)                    # MB (TB/s x us = MB)
    expert_mb = (TOPK + SHARED) * UNIT_MB
    exp = expert_us() * (expert_mb - moved) / expert_mb        # the stream scales with its bytes once spread evenly
    return lat + (f - 1.0) * PHASES[4][3] + exp + event_us


def interleave_period(f: float, g: float = 1.0) -> float:
    """Two requests; one period advances both by one layer. Each half-period runs one request's latency phases
    (slowed by f: the other request streams on the same CUs) beside the other's expert stream (slowed by g)."""
    return 2.0 * max(latency_us() * f, expert_us() * g)


ROUTED_UNION: dict[int, float] = {}      # measured distinct routed experts per step for N sequences (--routing)


def batch2_step() -> float:
    """Two tokens per weight read: latency phases as today, the expert stream over the union of both tokens' units."""
    units1 = TOPK + SHARED
    units2 = ROUTED_UNION.get(2, routed_union_units(2)) + SHARED
    return latency_us() + expert_us() * units2 / units1


def token_ms(layer_us: float) -> float:
    """Per-token time with MOE_LAYERS layers at layer_us and the rest of fleet's token unchanged."""
    return TOKEN_MS + MOE_LAYERS * (layer_us - LAYER_US) / 1000.0


# ------------------------------------------------------------------ microbenchmark logs
def parse_overlap(text: str) -> dict:
    out: dict = {"fold": [], "stream": {}}
    for line in text.splitlines():
        m = re.search(r"RESULT fold place=(\w+)\s+(?:fg (\d+) CUs/XCD)?\s*bg depth (\d+): fold median\s+([\d.]+) us.*bg ([\d.]+) TB/s", line)
        if m:
            out["fold"].append({"place": m.group(1), "fg": int(m.group(2) or 0), "depth": int(m.group(3)),
                                "us": float(m.group(4)), "bg_tbs": float(m.group(5))})
        m = re.search(r"RESULT stream\s+([\d.]+) MB\s+cold ([\d.]+) TB/s\s+warm ([\d.]+) TB/s\s+mall ([\d.]+) TB/s", line)
        if m:
            out["stream"][float(m.group(1))] = {"cold": float(m.group(2)), "warm": float(m.group(3)), "mall": float(m.group(4))}
    return out


def f_curve(parsed: dict) -> list[tuple[float, float]]:
    """(background TB/s, fold slowdown) for one foreground + one background workgroup per CU."""
    same = [r for r in parsed["fold"] if r["place"] == "same"]
    base0 = next((r["us"] for r in same if r["depth"] == 0), None)
    if not base0:
        return []
    return sorted((r["bg_tbs"], r["us"] / base0) for r in same if r["depth"] > 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overlap", type=Path, help="overlap_bench output (RESULT lines)")
    ap.add_argument("--routing", type=Path, help="routing_overlap.py JSON: measured expert union instead of uniform routing")
    ap.add_argument("--event-us", type=float, default=None, help="cross-XCD event cost under load, us (default: sweep 1.4 / 6.2)")
    a = ap.parse_args(argv)

    lat, exp = latency_us(), expert_us()
    print(f"fleet layer 5: {LAYER_US} us = latency-bound phases {lat:.1f} us + expert stream {exp:.1f} us; "
          f"token {TOKEN_MS} ms, {MOE_LAYERS} MoE layers")
    layer_mb = sum(v for k, v in per_layer(1024).items() if k not in ("dense_gu", "dense_dn", "lm_head", "embed")) / 1e6
    print(f"byte floor {layer_mb:.0f} MB / {BW} TB/s = {layer_mb / BW:.0f} us per layer\n")

    fs = [1.0, 1.15, 1.3, 1.5, 2.0]
    measured = None
    if a.overlap and a.overlap.exists():
        parsed = parse_overlap(a.overlap.read_text())
        curve = f_curve(parsed)
        if curve:
            measured = curve
            print("measured fold slowdown with a streaming workgroup on the same CU (overlap_bench fold place=same):")
            for tbs, f in curve:
                print(f"  background {tbs:.2f} TB/s -> fold x{f:.2f}")
            solo = next((r["us"] for r in parsed["fold"] if r["place"] == "solo"), None)
            same0 = next((r["us"] for r in parsed["fold"] if r["place"] == "same" and r["depth"] == 0), None)
            if solo and same0:
                print(f"  fold alone (8 workgroups) {solo:.2f} us; all CUs folding, nothing streaming {same0:.2f} us")
        if parsed["stream"]:
            print("stream rates (overlap_bench stream): MB  cold / warm / Infinity Cache, TB/s")
            for mb, r in sorted(parsed["stream"].items()):
                print(f"  {mb:6.1f}  {r['cold']:.2f} / {r['warm']:.2f} / {r['mall']:.2f}")
        print()
    events = [a.event_us] if a.event_us is not None else [1.4, 6.2]
    if a.routing and a.routing.exists():
        import json
        r = json.loads(a.routing.read_text())
        ROUTED_UNION.update({int(k): float(v) for k, v in r["overall"].items()})
        print(f"measured routed-expert union (routing_overlap.py): " + ", ".join(f"{k} seq {v:.2f}" for k, v in sorted(ROUTED_UNION.items())) + "\n")

    print("layout      condition                         layer us   token ms   faster by  per-request latency   throughput")
    print(f"base        fleet today                        {base():8.1f}   {token_ms(base()):8.2f}   {'':8}   {'':19}   1.00x")
    for ev in events:
        for f in (fs if measured is None else [measured[-1][1]]):
            for bg in (2.0, 4.0):
                lu = shared_layout(f, ev, bg)
                print(f"shared      f={f:.2f} event {ev:.1f} us bg {bg:.0f} TB/s    {lu:8.1f}   {token_ms(lu):8.2f}   "
                      f"{(1 - token_ms(lu) / TOKEN_MS) * 100:+6.1f}%")
    for f in (fs if measured is None else sorted({round(x[1], 2) for x in measured})):
        p = interleave_period(f)
        lat_ms = token_ms(p)
        print(f"interleave  f={f:.2f} (2 requests)               {p:8.1f}   {lat_ms:8.2f}   {'':8}   "
              f"{(lat_ms / TOKEN_MS - 1) * 100:+5.0f}% per token        {2 * TOKEN_MS / lat_ms:.2f}x")
    b2 = batch2_step()
    b2_ms = token_ms(b2)
    src = "measured" if 2 in ROUTED_UNION else "uniform routing"
    print(f"batch2      union {ROUTED_UNION.get(2, routed_union_units(2)):.1f}+{SHARED} units ({src})  {b2:8.1f}   {b2_ms:8.2f}   {'':8}   "
          f"{(b2_ms / TOKEN_MS - 1) * 100:+5.0f}% per token        {2 * TOKEN_MS / b2_ms:.2f}x")
    print("\nReading: interleave beats batch2 only while f stays below "
          f"{batch2_step() / 2 / latency_us():.2f}; "
          "compare both against vLLM at batch 2 (vllm_bench --batch 2) before building either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
