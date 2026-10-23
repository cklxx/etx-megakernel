"""Experiments as data (docs/core/SESSION-ARCH-2026-10-13.md, component D).

Invariant: an arm name is one build and one rule everywhere. A model family keeps ONE registry module (e.g.
examples/dsv2lite/arms.py) of Arm records; the build, the local build check, the session's jobs, the verdicts and the
re-derivation all read it. A Rule is data, evaluated by `decide` alone, on integer nanoseconds parsed from the logs'
decimal milliseconds: no floating-point subtraction on a decision path (review 07, R02).

  python -m etx.session.experiment verify <record or result dir> <registry module> [session job file]
      the record against its SEAL (or a legacy SHA256SUMS), then every pairs_<arm>.json and stages_<arm>.json found
      re-derived from the raw logs with the same code the session ran. 0 = verified and complete, 3 = verified but
      incomplete / unsealed, 2 = something differs. Supply the job file to require all its promised results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import importlib
import json
from pathlib import Path
import statistics
import sys


@dataclass(frozen=True)
class Rule:
    """GO when every paired gain (ref - candidate) is > 0, the median >= min_median_us and, unless spread_factor is
    None, median > spread_factor x spread. With equal_to (D2's (b)): also |median(candidate - equal_to)| <=
    max(2 x spread of (ref - equal_to), equal_floor_us)."""
    basis: str
    reference: str
    min_median_us: int
    spread_factor: int | None = 2
    strict: bool = True
    text: str = ""
    labels: tuple[str, str] = ("GO", "NO-GO")
    equal_to: str | None = None
    equal_floor_us: int = 0


@dataclass(frozen=True)
class Arm:
    """A build (levers, compiler flags, plan switches) and, if `order` is given, its paired measurement: `order` is the
    round-0 invocation order (reversed on odd rounds), `references` the references reported, `optional` the builds measured
    only when present (context), `rule` the pre-registered decision (None: attribution only)."""
    name: str
    levers: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    env: dict = field(default_factory=dict)
    order: tuple[str, ...] = ()
    references: tuple[str, ...] = ("default",)
    rule: Rule | None = None
    optional: tuple[str, ...] = ()
    stamped: bool = False
    builder: str = "etx"
    scratch_allow: int = 0          # bytes/lane of scratch the build may use (a recorded, measured exception; default none)

    @property
    def cand(self) -> str:
        return next(b for b in self.order if b not in self.references)


def ns(ms) -> int:
    """Decimal milliseconds (as logged, 3 decimals) -> integer nanoseconds, exactly."""
    if isinstance(ms, float):
        raise TypeError("timings must come from decimal text, never floats")
    decimal = Decimal(ms)
    if not decimal.is_finite():
        raise ValueError(f"non-finite timing: {ms}")
    numerator, denominator = decimal.as_integer_ratio()
    value, remainder = divmod(numerator * 1_000_000, denominator)
    if remainder:
        raise ValueError(f"timing is not an exact integer nanosecond: {ms}")
    return value


def paired(gains_ns: list[int]) -> dict:
    if len(gains_ns) != 5 or any(type(g) is not int for g in gains_ns):
        raise ValueError("a paired result needs five gains")
    med = statistics.median(gains_ns)
    return dict(gains_us=[g / 1000 for g in gains_ns], median_us=med / 1000,
                spread_us=(max(gains_ns) - min(gains_ns)) / 1000, every_positive=min(gains_ns) > 0)


def gains(rounds, ref, cand, basis="device_ms") -> list[int]:
    return [ns(r[ref][basis]) - ns(r[cand][basis]) for r in rounds]


def decide(rule: Rule, rounds, cand) -> dict:
    g = gains(rounds, rule.reference, cand, rule.basis)
    stats = paired(g)
    med, spread = statistics.median(g), max(g) - min(g)
    spread_ok = rule.spread_factor is None or (med > rule.spread_factor * spread if rule.strict
                                              else med >= rule.spread_factor * spread)
    ok = min(g) > 0 and med >= rule.min_median_us * 1000 and spread_ok
    d = dict(stats, ref=rule.reference, a_ok=ok, passed=ok)
    if rule.equal_to:
        rel = gains(rounds, rule.reference, rule.equal_to, rule.basis)
        b = [-x for x in gains(rounds, rule.equal_to, cand, rule.basis)]            # candidate - equal_to
        tol = max(2 * (max(rel) - min(rel)), rule.equal_floor_us * 1000)
        b_ok = abs(statistics.median(b)) <= tol
        d.update(equal_to=rule.equal_to, b_gains_us=[x / 1000 for x in b], b_median_us=statistics.median(b) / 1000,
                 b_tol_us=tol / 1000, b_ok=b_ok, passed=ok and b_ok)
    return d


def report(arm: Arm, rounds, condition: str) -> dict:
    refs = [r for r in arm.references if r in rounds[0]]
    out = dict(arm=arm.name, rounds=rounds, condition=condition,
               **{basis: {ref: paired(gains(rounds, ref, arm.cand, basis)) for ref in refs}
                  for basis in ("device_ms", "wall_ms")})
    if arm.rule is None:
        out["verdict"] = dict(rule="attribution or measurement only: no decision", outcome="NO RULE (attribution)")
    else:
        out["decision"] = decide(arm.rule, rounds, arm.cand)
        ok = out["decision"]["passed"]
        out["verdict"] = dict(rule=arm.rule.text, outcome=arm.rule.labels[0 if ok else 1], passed=ok)
    return out


def order(arm: Arm, present) -> tuple[str, ...]:
    """The invocation order, without the optional builds that are not present."""
    return tuple(b for b in arm.order if b not in arm.optional or present(b))


def measure(arm: Arm, invoke, present=lambda _: True, tag=None):
    """Five interleaved rounds, with the registry's order reversed on odd rounds."""
    names = order(arm, present)
    return [{b: invoke(b, f"{tag or arm.name}_r{i}_{b}") for b in (names if i % 2 == 0 else names[::-1])}
            for i in range(5)]


def same(stored: dict, again: dict, arm: Arm) -> bool:
    """Legacy floats may differ by <= 1e-6 us in display fields only; rules always use the raw decimal timings."""
    def close(a, b):
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(close(v, b[k]) for k, v in a.items())
        if isinstance(a, list):
            return isinstance(b, list) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
        return a == b if isinstance(a, (str, bool)) else abs(Decimal(str(a)) - Decimal(str(b))) <= Decimal("0.000001")
    def rounds(rs):
        return [{n: {k: ns(v) if k.endswith("_ms") else v for k, v in timing.items()}
                 for n, timing in r.items()} for r in rs]
    if stored["arm"] != arm.name or rounds(stored["rounds"]) != rounds(again["rounds"]):
        return False
    for basis in ("device_ms", "wall_ms"):
        if stored[basis].keys() != again[basis].keys():
            return False
        for ref, p in again[basis].items():
            if not close(p, {k: stored[basis][ref].get(k) for k in p}):
                return False
    if "verdict" in stored and stored["verdict"] != again["verdict"]:
        return False
    if "decision" in stored:
        return close(again["decision"], stored["decision"])
    if arm.rule:
        ok = again["decision"]["passed"]
        # Old reports predate `verdict`; check their stored decision, not merely the displayed gains.
        decisions = [v["keep"] if "keep" in v else v["go"] for v in stored.values()
                     if isinstance(v, dict) and ("keep" in v or "go" in v)]
        if not decisions and "verdict" not in stored:
            decisions = [stored[arm.rule.basis][arm.rule.reference]["go"]]
        return all(v == ok for v in decisions)
    return True


def verify(d: Path, registry, session: Path | None = None) -> tuple[int, list[str]]:
    from etx.session import record
    rc, lines = record.verify(d) if (d / "SEAL").exists() else record.verify_legacy(d, registry.archives(d))
    bad, incomplete = rc == 2, rc == 3
    if session is not None:
        from etx.experiments.jobs import load_session
        jobs, _ = load_session(session)
        expected = [w for j in jobs for w in j.writes if w.startswith(("pairs_", "stages_", "batch_report"))]
        missing = [w for w in expected if not (d / w).is_file()]
        incomplete |= bool(missing)
        lines.extend(f"MISSING {w} (promised by {session})" for w in missing)
    for f in sorted(d.glob("pairs_*.json")):
        try:
            arm = registry.ARMS[f.stem.removeprefix("pairs_")]
            rounds = measure(arm, lambda b, name: registry.parse((d / f"{name}.raw.log").read_text()),
                             lambda b: (d / f"{arm.name}_r0_{b}.raw.log").exists())
            again = report(arm, rounds, registry.CONDITION)
            ok = same(json.loads(f.read_text(), parse_float=Decimal), again, arm)
            bad |= not ok
            first = again["device_ms"][arm.references[0]]
            lines.append(f"{'OK  ' if ok else 'DIFF'} pairs {arm.name:10s} {again['verdict']['outcome']:22s} vs "
                         f"{arm.references[0]}: median {first['median_us']:.1f} us spread {first['spread_us']:.1f} "
                         f"every_positive={int(first['every_positive'])}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            bad = True
            lines.append(f"DIFF {f.name}: {exc}")
    try:
        for check in registry.checks(d):
            ok, line = check()
            bad |= ok is False
            incomplete |= ok is None
            lines.append(line)
    except (OSError, ValueError, KeyError) as exc:
        bad = True
        lines.append(f"DIFF model checks: {exc}")
    lines.append("NOT VERIFIED" if bad else "RE-DERIVED: pairs and completed registry checks from raw files; "
                 "missing inputs / exclusions are listed above")
    return (2 if bad else 3 if incomplete else 0), lines


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) not in (3, 4) or a[0] != "verify":
        raise SystemExit(__doc__)
    rc, lines = verify(Path(a[1]), importlib.import_module(a[2]), Path(a[3]) if len(a) == 4 else None)
    print("\n".join(lines), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
