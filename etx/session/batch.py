"""Batch studies: total throughput and per-request latency, each decided on its own (docs/BATCH-2026-10-14.md).

Model-agnostic. A study has, per batch size B, one ETX arm and the strongest legal vLLM configuration on the same box.
Round 0 selects the vLLM baseline per B; rounds 1-5 give one (ETX, vLLM) pair per B each. Every timing is decimal
text from a log, turned into integer nanoseconds (experiment.ns); every gain is an exact rational. Nothing here
launches anything.

  ETX invocation   `BATCHREQ r=.. tokens=32/32 layers=L warmup=32/32 interval_median_ms=.. device_median_ms=..` per
                   request and one `BATCH requests=B tokens=32 step_median_ms=..` (examples/dsv2lite/host.hip).
  vLLM process     `VLLMSEQ label=.. batch=B phase=warmup|sample<k> seq=r first_divergence=n/32 tokens=t` per sequence
                   and phase, `VLLMB label=.. batch=B per_token_ms=v1,..,v5` per B (examples/llm/vllm_bench.py --goldens).
"""
from __future__ import annotations

from fractions import Fraction
import re

from etx.session.experiment import ns

MIN_GAIN = Fraction(2, 100)
SPREAD_FACTOR = 2
ROUNDS = 5
SAMPLES = 5
LABELS = ("WIN", "LOSS", "TIE")


class Incomplete(ValueError):
    """A gate failed: the invocation (or configuration) gives no number."""


def kv(line: str) -> dict:
    return dict(re.findall(r"(\w+)=(\S+)", line))


def etx_result(log: str, b: int, tokens: int = 32, layers: int | None = 27) -> dict:
    """Every request of one B-request ETX invocation passes its own gates, or Incomplete. Returns the step time
    (throughput input) and the worst request's token interval (latency input), both exact decimal text."""
    if re.search(r"MISMATCH|\[FAIL\]|no reference token", log):
        raise Incomplete("a correctness failure in the log")
    head = [kv(x) for x in re.findall(r"^BATCH (requests=.*)$", log, re.M)]
    if len(head) != 1 or head[0].get("requests") != str(b) or head[0].get("tokens") != str(tokens):
        raise Incomplete(f"expected one BATCH line for {b} requests x {tokens} tokens, got {head}")
    reqs = [kv(x) for x in re.findall(r"^BATCHREQ (r=.*)$", log, re.M)]
    if sorted(int(q["r"]) for q in reqs) != list(range(b)):
        raise Incomplete(f"expected requests 0..{b - 1} once each, got {[q.get('r') for q in reqs]}")
    full = f"{tokens}/{tokens}"
    for q in reqs:
        if q["tokens"] != full or q["warmup"] != full:
            raise Incomplete(f"request {q['r']}: tokens {q['tokens']}, warm-up {q['warmup']} (need {full})")
        if layers is not None and q["layers"] != str(layers):
            raise Incomplete(f"request {q['r']}: {q['layers']} layers within the gate (need {layers})")
    step = head[0]["step_median_ms"]
    worst = max(reqs, key=lambda q: ns(q["interval_median_ms"]))
    if ns(step) <= 0 or ns(worst["interval_median_ms"]) <= 0:
        raise Incomplete("a zero timing")
    return dict(step_ms=step, latency_ms=worst["interval_median_ms"], worst_request=int(worst["r"]),
                requests={int(q["r"]): dict(interval_ms=q["interval_median_ms"], device_ms=q["device_median_ms"])
                          for q in reqs})


def median_text(values: list[str]) -> str:
    """The median of an odd number of decimal texts, as one of them (no arithmetic on the path)."""
    if len(values) % 2 != 1:
        raise ValueError("an odd number of samples is required")
    return sorted(values, key=ns)[len(values) // 2]


def vllm_results(log: str, label: str, tokens: int = 32, samples: int = SAMPLES) -> dict[int, dict]:
    """Per B: the step time (median per-token decode cost of the batch, decimal text) and its legality: every sequence
    `tokens` tokens equal to its golden on the warm-up and on every sample."""
    seqs = [kv(x) for x in re.findall(r"^VLLMSEQ (label=.*)$", log, re.M)]
    out = {}
    for line in re.findall(r"^VLLMB (label=.*)$", log, re.M):
        d = kv(line)
        if d["label"] != label:
            continue
        b = int(d["batch"])
        per = d["per_token_ms"].split(",")
        mine = [s for s in seqs if s["label"] == label and s["batch"] == str(b)]
        phases = ["warmup", *(f"sample{k}" for k in range(samples))]
        seen = {(s["phase"], int(s["seq"])): s for s in mine}
        missing = [(p, r) for p in phases for r in range(b) if (p, r) not in seen]
        bad = [(p, r) for (p, r), s in seen.items()
               if s["first_divergence"] != f"{tokens}/{tokens}" or s["tokens"] != str(tokens)]
        reason = ("" if not missing and not bad else
                  f"missing {missing[:4]}" if missing else f"diverged {[(p, r, seen[p, r]['first_divergence']) for p, r in bad[:4]]}")
        if len(per) != samples:
            reason = reason or f"{len(per)} samples, need {samples}"
        if b in out:
            raise ValueError(f"two VLLMB lines for {label} batch {b}")
        out[b] = dict(step_ms=median_text(per) if len(per) % 2 else None, samples_ms=per, legal=not reason,
                      reason=reason or None)
    return out


def select(round0: dict[str, dict[int, dict]], batches) -> dict[int, dict]:
    """Per B, the faster legal configuration of the selection round (ties: the first configuration given)."""
    out = {}
    for b in batches:
        legal = [(c, r[b]) for c, r in round0.items() if b in r and r[b]["legal"]]
        if not legal:
            out[b] = dict(config=None, verdict="NO LEGAL BASELINE",
                          reasons={c: (r[b]["reason"] if b in r else "not measured") for c, r in round0.items()})
        else:
            c, r = min(legal, key=lambda cr: ns(cr[1]["step_ms"]))
            out[b] = dict(config=c, step_ms=r["step_ms"])
    return out


def gain(v_ms: str, e_ms: str) -> Fraction:
    """ETX's relative gain in one pair: vLLM's time over ETX's, minus one (positive = ETX better)."""
    return Fraction(ns(v_ms), ns(e_ms)) - 1


def verdict(gains: list[Fraction]) -> dict:
    """WIN / LOSS / TIE on five paired gains: every gain one sign, |median| >= 2% and |median| > 2 x spread."""
    if len(gains) != ROUNDS:
        raise ValueError(f"a verdict needs {ROUNDS} pairs, got {len(gains)}")
    g = sorted(gains)
    med, spread = g[len(g) // 2], g[-1] - g[0]
    win = g[0] > 0 and med >= MIN_GAIN and med > SPREAD_FACTOR * spread
    loss = g[-1] < 0 and -med >= MIN_GAIN and -med > SPREAD_FACTOR * spread
    pct = lambda x: float(x * 100)
    return dict(outcome="WIN" if win else "LOSS" if loss else "TIE", gains_pct=[pct(x) for x in gains],
                median_pct=pct(med), spread_pct=pct(spread))


def throughput(b: int, step_ms: str) -> float:
    """Tokens per second of a B-request step (display only; decisions use the exact gains)."""
    return float(Fraction(b * 10**9, ns(step_ms)))


def study(rounds: list[dict], selection: dict[int, dict], batches) -> dict[int, dict]:
    """rounds[k] = {"etx": {B: etx_result | {"error": ..}}, "vllm": {config: {B: vllm result} | {"error": ..}}}.
    Per B: NOT COMPLETE, NO LEGAL BASELINE, or the two verdicts with the paired data, plus the reported scalings."""
    out = {}
    for b in batches:
        sel = selection.get(b, {})
        rec = dict(batch=b)
        if sel.get("config") is None:
            out[b] = dict(rec, status="NO LEGAL BASELINE", reasons=sel.get("reasons"))
            continue
        cfg, pairs, problems = sel["config"], [], []
        for k, r in enumerate(rounds, 1):
            e = r["etx"].get(b)
            proc = r["vllm"].get(cfg, {"error": "not run"})
            v = None if "error" in proc else proc.get(b)
            if e is None or "error" in e:
                problems.append(f"round {k}: ETX {e.get('error') if e else 'not run'}")
            elif v is None or not v["legal"]:
                problems.append(f"round {k}: vLLM {cfg} {proc.get('error') or (v['reason'] if v else 'B not measured')}")
            else:
                pairs.append((e, v))
        if problems or len(pairs) != ROUNDS:
            out[b] = dict(rec, status="NOT COMPLETE", config=cfg, problems=problems or [f"{len(pairs)} pairs"])
            continue
        rec.update(status="COMPLETE", config=cfg,
                   etx_step_ms=[e["step_ms"] for e, _ in pairs], etx_latency_ms=[e["latency_ms"] for e, _ in pairs],
                   vllm_step_ms=[v["step_ms"] for _, v in pairs],
                   vllm_latency_note="vLLM advances every sequence each step: its per-request latency is its step time",
                   throughput=verdict([gain(v["step_ms"], e["step_ms"]) for e, v in pairs]),
                   latency=verdict([gain(v["step_ms"], e["latency_ms"]) for e, v in pairs]))
        rec["etx_tokens_per_s"] = throughput(b, median_text(rec["etx_step_ms"]))
        rec["vllm_tokens_per_s"] = throughput(b, median_text(rec["vllm_step_ms"]))
        out[b] = rec
    one = out.get(1)
    for b, rec in out.items():                 # reported, no rule: against the same rounds' batch-1 runs
        if rec.get("status") == "COMPLETE" and one and one.get("status") == "COMPLETE" and b != 1:
            ratio = lambda xs, ys: float(sorted(Fraction(ns(x), ns(y)) for x, y in zip(xs, ys))[ROUNDS // 2])
            rec["etx_throughput_scaling"] = b / ratio(rec["etx_step_ms"], one["etx_step_ms"])
            rec["etx_latency_inflation"] = ratio(rec["etx_latency_ms"], one["etx_latency_ms"])
            rec["vllm_throughput_scaling"] = b / ratio(rec["vllm_step_ms"], one["vllm_step_ms"])
    return out


def build_decision(rec: dict, kept_step_ms: str, extra_bytes: float | None, rate_tbs: float = 3.5,
                   factor: Fraction = Fraction(9, 10)) -> dict:
    """Section 5: after a LOSS or TIE at B = 4 or 8, build column-batched tiles only if the optimistic projection
    P = kept step + extra bytes / rate is at most 0.9 x vLLM's median step at that B."""
    if rec.get("status") != "COMPLETE":
        return dict(next=f"rerun B={rec['batch']} after a local fix ({rec.get('status')})")
    if rec["batch"] == 1:
        return dict(next="batch-1 reference point: no batch decision")
    if rec["throughput"]["outcome"] == "WIN":
        return dict(next="replication is ETX's batch mode at this B; record it; Qwen batch study next")
    if rec["batch"] not in (4, 8):
        return dict(next="no build decision from this B alone; trace it next")
    if extra_bytes is None:
        return dict(next="PENDING: the expert union was not measured, so P(B) cannot be computed")
    p_ns = ns(kept_step_ms) + round(Fraction(extra_bytes) / Fraction(str(rate_tbs)) / 1000)   # bytes / (TB/s) = ps; / 1000 = ns
    v_ns = ns(median_text(rec["vllm_step_ms"]))
    build = p_ns <= factor * v_ns
    return dict(projection_ms=p_ns / 1e6, vllm_ms=v_ns / 1e6, bound_ms=float(factor * v_ns) / 1e6, build=build,
                next="design and build column-batched tiles (stage B2), threshold before code" if build else
                     "do not pursue throughput at this B: vLLM is within 10% of the best batched ETX projection")
