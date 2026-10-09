"""Report one nvref session on the pre-registered host-wall basis.

No partial performance table is emitted. gpu_jobs command comments are never
parsed as evidence. Synthetic fixtures follow tests/test_tile_ab.py (tmp_path).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from examples.vllm_llm.pairs import Refusal, kv, parse, verdict as fusion_verdict


@dataclass
class Timing:
    wall: float
    device: float | None = None
    result_wall: float | None = None


class Session:
    def __init__(self, directory: Path):
        self.directory = directory
        status = directory / "status.json"
        self.status = {j["name"]: j for j in json.loads(status.read_text())} if status.exists() else {}

    def read(self, name: str, gpu: str = "0", allow_failed=False) -> str:
        path = self.directory / f"{name}.log"
        if name in self.status and self.status[name]["status"] != "ok" and not allow_failed:
            raise Refusal(f"{name}: job {self.status[name]['status']}")
        if not path.is_file():
            raise Refusal(f"{name}: missing run/job log")
        text = path.read_text(errors="replace")
        headers = re.findall(r"^# job (\S+) gpu=(\S+)", text, re.M)
        if headers != [(name, gpu)]:
            raise Refusal(f"{name}: expected one job header on gpu={gpu}, got {headers}")
        return "\n".join(line for line in text.splitlines() if not line.startswith("#"))


def one_line(text: str, prefix: str, name: str) -> str:
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise Refusal(f"{name}: expected one {prefix.strip()} line, found {len(lines)}")
    return lines[0]


def positive(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError:
        raise Refusal(f"{name}: invalid time {value!r}") from None
    if not math.isfinite(result) or result <= 0:
        raise Refusal(f"{name}: nonpositive/nonfinite time")
    return result


def correctness(text: str, name: str) -> dict:
    fields = kv(one_line(text, "RESULT ", name))
    if fields.get("matched") != "32" or fields.get("gen") != "32":
        raise Refusal(f"{name}: requires matched=32 gen=32 (32/32)")
    if fields.get("requests") != "1" or fields.get("matched_each", "32") != "32":
        raise Refusal(f"{name}: requires one request, 32/32")
    # The host reports correctness per repeat before the single final RESULT.
    for matched, generated in re.findall(r"^run \d+(?: request \d+)?: (\d+)/(\d+) generated tokens match HF greedy", text, re.M):
        if (matched, generated) != ("32", "32"):
            raise Refusal(f"{name}: a repeat is {matched}/{generated}, requires 32/32")
    if "no token check" in text or re.search(r"^CHECK(?:-DSL)? FAILED", text, re.M):
        raise Refusal(f"{name}: incomplete run or failed correctness check")
    return fields


def etx(text: str, name: str, arm: str, steps=True) -> Timing:
    fields = correctness(text, name)
    parsed = parse(text.splitlines())
    if len(parsed.runs) != 1 or parsed.runs[0].arm != arm:
        raise Refusal(f"{name}: wrong ETX mode")
    device = positive(fields.get("median_ms", ""), name)
    wall = positive(fields.get("wall_ms", ""), name)
    if not steps:
        return Timing(wall, device, wall)
    # A new step 0 starts a repeat. Do not select a complete earlier repeat if
    # the last repeat is truncated; duplicated/missing decode steps also fail.
    last, current = [], []
    for line in text.splitlines():
        match = re.fullmatch(r"\s+step (\d+): (\S+) ms wall", line)
        if match:
            t = int(match[1])
            if t == 0:
                current = []
            if t >= 1024:
                current.append((t, positive(match[2], name)))
        if re.match(r"^run \d+: \d+/\d+ generated tokens match HF greedy", line):
            last, current = current, []
    if current:
        last = current
    if [t for t, _ in last] != list(range(1024, 1055)):
        raise Refusal(f"{name}: expected 31 measured steps t=1024..1054 in last run, got n={len(last)} or wrong indices")
    return Timing(mean(ms for _, ms in last), device, wall)


def vllm(text: str, name: str, aiter="0") -> Timing:
    f = kv(one_line(text, "VLLM ", name))
    expected = dict(label="default", aiter=aiter, batch="1", version="0.27.1", context="1024", first_divergence="32/32")
    for key, value in expected.items():
        if f.get(key) != value:
            raise Refusal(f"{name}: expected {key}={value}, got {f.get(key, 'missing')}")
    return Timing(positive(f.get("median_ms", ""), name))


def verdict(deltas: list[float]) -> str:
    med = median(deltas)
    # Decimal log values are represented as binary floats.
    if all(d < 0 for d in deltas) and (med <= -0.050 or math.isclose(med, -0.050, abs_tol=1e-12)):
        return "LEAD"
    if all(d > 0 for d in deltas) and (med >= 0.050 or math.isclose(med, 0.050, abs_tol=1e-12)):
        return "LOSS"
    if abs(med) < 0.050 and not math.isclose(abs(med), 0.050, abs_tol=1e-12):
        return "TIE"
    return "INSUFFICIENT"


def row(system: str, hardware: str, basis: str, values: list[float]):
    m = median(values)
    print(f"| this session | {system} | {hardware} | {basis} | "
          + ", ".join(f"{x:.3f}" for x in values)
          + f" | {m:.3f} | {mean(values):.3f} | {max(values) - min(values):.3f} | {1000 / m:.1f} | {len(values)}/{len(values)} |")


def report(directory: Path):
    session = Session(directory)
    def gate(name):
        text = session.read(name)
        correctness(text, name)
        if not re.search(r"^GATE-OK$", text, re.M):
            raise Refusal(f"{name}: gate failed (missing GATE-OK)")
        if name == "gate_all" and not re.search(r"^CHECK-DSL OK\b", text, re.M):
            raise Refusal(f"{name}: missing CHECK-DSL OK")

    gate("gate_all")
    # A differing launch is a correctness failure and refuses the report. A check that
    # did not run is different: the session keeps its numbers and the verdict line says
    # the gate is not met, so no claim rests on evidence that does not exist.
    PER_XCD = "--check needs a plan without per-XCD replicas (build without ETX_VL_XCD)"
    texts = {}
    for name in ("check_imp1", "check_imp"):
        if (directory / f"{name}.log").exists() or name in session.status:
            text = session.read(name, allow_failed=True) if (directory / f"{name}.log").exists() else ""
            texts[name] = (text, session.status.get(name, {}).get("status", "ok"))
    for name, (text, _status) in texts.items():
        if re.search(r"^CHECK FAILED", text, re.M):
            raise Refusal(f"{name}: CHECK FAILED; no performance verdict")
    imported = None
    ok = re.compile(r"^CHECK OK: 0 launch\(es\) differ$", re.M)
    t1, s1 = texts.get("check_imp1", (None, None))
    t0, s0 = texts.get("check_imp", (None, None))
    if t1 and ok.search(t1) and s1 == "ok":
        count = len(re.findall(r"^\s+\S+\s+L-?\d+\s+\S+\s+identical\s+original ", t1, re.M))
        # This reporter is Qwen3-30B-A3B-specific (97 planned imports). An old
        # summary without instance records must not manufacture five checks.
        imported = f"INCOMPLETE {count if count else '?'}/97, different build"
    elif t1 is None and t0 and ok.search(t0) and s0 == "ok":
        imported = "INCOMPLETE (uncertified legacy check)"
    elif t1 is not None:
        # The real check ran (or was attempted): never hide its state behind the legacy refusal.
        imported = f"NOT MET (check_imp1: {s1})"
    elif t0 is not None and PER_XCD in t0:
        imported = "n/a (host.hip:515)"
    else:
        reasons = ", ".join(f"{n}: {st}" for n, (_t, st) in texts.items()) or "no check job ran"
        imported = f"NOT MET ({reasons})"

    # Revalidate evidence, not a saved success label. Legacy timing statistics
    # remain timing-only; only this executable's complete certificate can qualify.
    from examples.vllm_llm.import_certificate import validate
    eligible = False
    certificate_reason = "no same-executable certificate"
    if (directory / "build_identity.json").exists():
        try:
            certificate = validate(directory, (0, 1040))
            imported = f"OK (certified {certificate['checked']}/{certificate['planned']}, positions 0,1040)"
            eligible = certificate['eligible']
            certificate_reason = "wall timing rule not met"
        except (Refusal, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            imported = f"INCOMPLETE (certificate: {exc})"
            certificate_reason = str(exc)

    golden = session.read("golden_sha", "None")
    if not re.search(r"(?:^|/)golden_c1024.txt: OK$", golden, re.M):
        raise Refusal("golden_sha: missing golden_c1024.txt: OK")
    record = session.read("record", "None")
    revision = one_line(record, "HF_REV ", "record").removeprefix("HF_REV ").strip()
    digest = one_line(session.read("docker_pull", "None"), "IMAGE ", "docker_pull").removeprefix("IMAGE ").strip()
    machine = " / ".join(line.strip().replace("|", "/") for line in record.splitlines()
                         if line.strip() and not line.startswith("HF_REV "))
    if not machine or not revision or not digest:
        raise Refusal("missing machine/image/HF_REV provenance")

    arms = {}
    for arm in ("vllm", "etx"):
        runs = []
        for rnd in range(1, 6):
            name = f"{arm}_r{rnd}"
            text = session.read(name)
            runs.append(vllm(text, name) if arm == "vllm" else etx(text, name, "fused" if arm == "etx" else "graph"))
        arms[arm] = runs

    fusion_unavailable = None
    arms["etxu"] = []
    try:
        gate("gate_unf")
        unfused = [etx(session.read(f"etxu_r{rnd}"), f"etxu_r{rnd}", "graph") for rnd in range(1, 6)]
    except (Refusal, OSError, ValueError, KeyError, TypeError) as exc:
        fusion_unavailable = str(exc)
    else:
        arms["etxu"] = unfused

    # Optional evidence is validated independently; failures cannot refuse the primary report.
    extras, unavailable = {}, {}
    for name in ("hostprof", "etx_noprog"):
        try:
            extras[name] = etx(session.read(name), name, "fused", steps=name != "etx_noprog")
        except (Refusal, OSError, ValueError, KeyError, TypeError) as exc:
            unavailable[name] = str(exc)
    aiter = []
    for rnd in range(1, 4):
        name = f"vllm_aiter_r{rnd}"
        try:
            aiter.append(vllm(session.read(name), name, "1").wall)
        except (Refusal, OSError, ValueError, KeyError, TypeError) as exc:
            unavailable[name] = str(exc)

    # All validation precedes output: refusal never leaves a partial table.
    print("Qwen3-30B-A3B bf16, batch 1, prompt 1024, 32 greedy tokens; same GPU 0.")
    print("| source | system | hardware | basis | runs (ms/token) | median | mean | spread | tok/s | 32/32 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    hardware = f"MI300X {machine}"
    v, f, u = ([r.wall for r in arms[a]] for a in ("vllm", "etx", "etxu"))
    row("vLLM 0.27.1 default", hardware, "host wall, (t32-t1)/31", v)
    row("ETX fused, 4 tile flags", hardware, "host wall, 31-step mean", f)
    if fusion_unavailable is None:
        row("ETX unfused, HIP graph", hardware, "host wall, 31-step mean", u)
    row("ETX fused (secondary)", hardware, "device span, not vs vLLM", [r.device for r in arms["etx"]])
    # The user requires non-log numbers to be labelled estimates. Keep the
    # section-5 source/system labels verbatim and mark these external values.
    for system, hw, rate in (("vLLM 0.10.2, CUDA 13", "H200", 189.5),
                             ("vLLM 0.10.2, CUDA 13", "B200", 180.1),
                             ("Paiton, ROCm 6.4.1", "MI300X", 189.4),
                             ("vLLM 0.10.0", "MI300X", 162.2)):
        print(f"| third party eliovp.com 2025-09-26, not reproduced | {system} | {hw} | unknown | - | - | "
              f"{1000 / rate:.2f} (estimate) | - | {rate:.1f} (estimate; spec, not session logs) | not checked |")
    print("Third party: Instruct-2507, ShareGPT, 256 output tokens, mean of 5; ms/token = 1000 / tok/s. "
          "Prefill/TTFT inclusion unknown. No H200/B200 comparison or ratio chaining is supported.")
    for arm in ("etx", "etxu"):
        for rnd, run in enumerate(arms[arm], 1):
            if abs(run.wall - run.result_wall) > 0.020 + 1e-12:
                print(f"BASIS-MISMATCH {arm}_r{rnd} mean_ms={run.wall:.3f} RESULT wall_ms={run.result_wall:.3f}")
    deltas = [a - b for a, b in zip(f, v)]
    print("DELTA r1..r5 (ETX fused - vLLM, same round) " + ", ".join(f"{d:+.3f}" for d in deltas)
          + f" median={median(deltas):+.3f} spread={max(deltas) - min(deltas):.3f} ms")
    spread = max(deltas) - min(deltas)
    med_abs = abs(median(deltas))
    unresolved = med_abs < spread or spread > 0.050 + 1e-12
    if unresolved:
        ratio = f"{med_abs / spread:.1f}x the spread" if spread else "spread is zero"
        print(f"RESOLUTION: paired delta spread {spread:.3f} ms against a 0.050 ms threshold; "
              f"|median|={med_abs:.3f} ms is {ratio}")
    if fusion_unavailable is None:
        gains = [(a - b) / a for a, b in zip(u, f)]
        rule = fusion_verdict(gains)
        print(f"FUSION (unfused - fused) median={median([a - b for a, b in zip(u, f)]):+.3f} ms "
              f"gain median={rule['median']:.2%} spread={rule['spread']:.2%} "
              f"rule {'met' if rule['established'] else 'not met'} "
              "(all gains > 0, median > 2%, median > 2 x spread)")
        hf, hu = (median([r.result_wall - r.device for r in arms[a]]) for a in ("etx", "etxu"))
        print(f"HOST-EXCESS fused median={hf:.3f} unfused median={hu:.3f} "
              f"excess(fused-unfused)={hf - hu:.3f} ms (RESULT wall_ms - median_ms)")
    else:
        print(f"FUSION unavailable ({fusion_unavailable})")
        print(f"HOST-EXCESS unavailable ({fusion_unavailable})")
    print(f"IMPORT-CHECK {imported}")
    print("FULL-CLAIM " + ("ELIGIBLE" if eligible else f"NOT ELIGIBLE ({certificate_reason}); timing-only observations"))
    result = verdict(deltas)
    if unresolved and med_abs <= spread:
        # The sign of the paired delta is not stable: no verdict can rest on it.
        result = "INSUFFICIENT"
    print(f"VERDICT {result} | import-check {imported} | image {digest} | HF_REV {revision} | machine {machine}")
    if len(aiter) == 3:
        row("vLLM 0.27.1 AITER", hardware, "host wall, (t32-t1)/31", aiter)
        if median(v) - median(aiter) > 0.050 + 1e-12:
            # Three late AITER runs cannot pair with five interleaved rounds.
            # Section 7 asks to recompute: use the measured AITER median as the
            # common reference, explicitly distinguishing the unpaired result.
            ad = [x - median(aiter) for x in f]
            print("DELTA AITER (unpaired, each fused round - AITER median) "
                  + ", ".join(f"{x:+.3f}" for x in ad) + f" median={median(ad):+.3f} spread={max(ad) - min(ad):.3f} ms")
            print(f"VERDICT {verdict(ad)} against AITER (unpaired median reference); default={result}")
    else:
        print("AITER comparison unavailable (nice-to-have runs incomplete)")
    for name, reason in unavailable.items():
        print(f"{name} unavailable ({reason})")
    for name, run in extras.items():
        print(f"{name} wall_ms={run.wall:.3f} "
              + ("(RESULT median; no per-step progress)" if name == "etx_noprog" else "(31-step mean)"))
    print("Residual not measured: vLLM includes sampling, output processing and scheduling; "
          "ETX retains tokens on device, favouring ETX. A lead is an ETX-system result; "
          "the fusion claim uses only fused versus unfused with the same tiles.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    try:
        report(args.session.expanduser())
    except (Refusal, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"REFUSED: {exc}. No performance numbers.")
        print("VERDICT NOT-COMPLETE")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
