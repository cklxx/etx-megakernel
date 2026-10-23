"""The batch 2 / 4 / 8 study on DeepSeek-V2-Lite (docs/deepseek/BATCH-2026-10-14.md; rules committed before this code, 0fc38c9).

Operations, each run by one job of examples/sessions/ds_batch.py; every invocation is a fresh process with its full log
kept (inplace_session.Session.run: deadline, progress, probes):
  gate <arm>   one invocation of a multi-request arm with every per-request gate -> bgate_<arm>.json
  boxcontrol   one B = 1 invocation each of the shipped and previous recipes -> boxcontrol.json
  vsel         vLLM round 0: both configurations, B = 1, 2, 4, 8, one engine each -> vsel.json (baseline per B)
  rounds       rounds 1-5: the kept arm, the x arms that passed their gate, the selected vLLM configuration(s);
               order reversed on even rounds -> brounds.json
  report       verdicts per B and quantity, scalings, the section-5 next step per B -> batch_report.json
No invocation here provisions a GPU or downloads; inputs come from the prep jobs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from etx.session import batch
from etx.session.experiment import ns
from etx.models.declarations import declaration as _declaration
from . import paths
BATCH_ARMS = _declaration(paths.ROOT, "dsv2lite.arms").BATCH_ARMS
BATCHES = _declaration(paths.ROOT, "dsv2lite.arms").BATCHES
X8_CANDIDATES = _declaration(paths.ROOT, "dsv2lite.arms").X8_CANDIDATES
from etx.models.dsv2lite.session import Session, write_json

IMAGE = "vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7"
CONFIGS = {"default": "0", "aiter": "1"}            # VLLM_ROCM_USE_AITER, in tie-break order
MODEL = Path.home() / "models/dsv2-lite-base"     # examples/sessions/ds_batch.py prepares and declares this path
VLLM_DEADLINE_S = 900
BOXCONTROL_PREVIOUS_ARM = "alld4r0l1"


class BatchSession(Session):
    batches = BATCHES

    def etx_args(self, b: int) -> list[str]:
        return [*self.args(), *(a for r in range(1, b) for a in ("--req-dir", str(self.data / f"r{r}")))]

    def etx(self, b: int, name: str, arm: str | None = None) -> dict:
        """One gated invocation of one arm at B requests: every request 32/32, 27/27, warm-up 32/32
        (etx/session/batch.py).

        `arm` says which binary runs: it defaults to BATCH_ARMS[b], and the batch-8 diagnostic passes its own name so
        the run is that candidate's tree (b_<arm>/run) and never another arm's. Without this, the four candidate labels
        all executed b_x8/run and the four verdicts were one measurement written four times.
        """
        arm = arm or BATCH_ARMS[b]
        try:
            # Session.run strips caller ETX_* overrides. Forward only this opt-in host diagnostic.
            options = dict(env=dict(ETX_DS_LAYER_METRICS="1")) if os.environ.get("ETX_DS_LAYER_METRICS") == "1" else {}
            text = self.run(name, [self.out(arm) / "run", *self.etx_args(b)], **options)
            return batch.etx_result(text, b)
        except (RuntimeError, ValueError) as exc:
            print(f"BATCH-INCOMPLETE invocation={name} arm={arm} b={b}: {exc}", flush=True)
            return dict(error=str(exc))

    def vllm(self, config: str, name: str, *, batches=None) -> dict:
        """One process for the requested batch sizes; every invoked full/short gate is read without filtering."""
        invoked = tuple(self.batches if batches is None else batches)
        if not invoked or len(set(invoked)) != len(invoked) or any(b not in self.batches for b in invoked):
            raise ValueError(f'invalid vLLM batch subset: {invoked}')
        requested = Path(os.environ.get("ETX_DSV2_MODEL", MODEL)).expanduser()
        if requested.resolve() != MODEL.resolve():
            raise ValueError(f"ETX_DSV2_MODEL {requested} differs from prepared model {MODEL}; refused")
        goldens = [str(self.data / "vllm" / f"golden_r{r}.txt") for r in range(max(invoked))]
        sudo = [] if os.geteuid() == 0 else ["sudo"]
        command = [*sudo, "docker", "run", "--rm", "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
                   "--ipc=host", "--shm-size", "16g", "--security-opt", "seccomp=unconfined",
                   "-e", f"HIP_VISIBLE_DEVICES={os.environ.get('GPU', '0')}", "-e", "PYTHONUNBUFFERED=1",
                   "-e", f"VLLM_ROCM_USE_AITER={CONFIGS[config]}", "-v", f"{Path.home()}:{Path.home()}",
                   "-w", str(self.root), "--entrypoint", "timeout", IMAGE, "-k", "5", str(VLLM_DEADLINE_S - 30),
                   "python3", "examples/llm/vllm_bench.py", "--model", str(MODEL), "--goldens", *goldens,
                   "--batches", ",".join(map(str, invoked)), "--tokens", "32", "--repeats", str(batch.SAMPLES),
                   "--label", config, "--echo-config", "--json", str(self.logs / f"{name}.json")]
        try:
            result = batch.vllm_results(self.run(name, command, timeout=VLLM_DEADLINE_S), config)
            if batches is not None and set(result) != set(invoked):
                raise ValueError(f'vLLM batch outputs {sorted(result)} differ from invoked subset {invoked}')
            return result
        except (RuntimeError, ValueError) as exc:
            print(f"BATCH-INCOMPLETE invocation={name} vllm={config}: {exc}", flush=True)
            return dict(error=str(exc))

    def fresh(self, path: Path) -> Path:
        if path.exists():
            raise RuntimeError(f"{path.name} already exists: use a fresh ETX_SESS_LOGDIR")
        return path

    def boxcontrol(self):
        """Two invocations, estimated ~1 GPU minute, on this study's box. Reuse etx()'s gates and exact host
        BATCHREQ device median / BATCH step-wall median, never the human wall mean or token interval.
        This is a within-session recipe difference (shipped minus previous), not a machine offset.
        One invocation per recipe cannot establish a performance verdict or convert times between machines.
        pct is the wall difference / previous wall median x 100.
        """
        out = self.fresh(self.logs / "boxcontrol.json")
        arms = (BATCH_ARMS[1], BOXCONTROL_PREVIOUS_ARM)
        measurements = {}
        for arm in arms:
            result = self.etx(1, f"boxcontrol_{arm}", arm=arm)
            if "error" in result:
                raise RuntimeError(f"BOXCONTROL incomplete arm={arm}: {result['error']}")
            values = dict(device_median_ms=result["requests"][0]["device_ms"], wall_median_ms=result["step_ms"])
            if any(ns(value) <= 0 for value in values.values()):
                raise RuntimeError(f"BOXCONTROL incomplete arm={arm}: nonpositive median {values}")
            measurements[arm] = values
        shipped, previous = (measurements[arm] for arm in arms)
        device_ns = ns(shipped["device_median_ms"]) - ns(previous["device_median_ms"])
        wall_ns = ns(shipped["wall_median_ms"]) - ns(previous["wall_median_ms"])
        difference = dict(device_ms=f"{device_ns / 1_000_000:+.6f}", wall_ms=f"{wall_ns / 1_000_000:+.6f}",
                      pct=f"{100 * wall_ns / ns(previous['wall_median_ms']):+.6f}", between=list(arms))
        write_json(out, dict(arms=measurements, recipe_difference=difference,
                             measurement_kind="within_session_recipe_difference", invocations_per_arm=1,
                             machine_offset_measured=False))
        # Emit only after both arms passed: an absent/failed arm must never leave a partial control number.
        for arm, values in measurements.items():
            print(f"BOXCONTROL arm={arm} b=1 device_median_ms={values['device_median_ms']} "
                  f"wall_median_ms={values['wall_median_ms']} tokens=32/32 layers=27/27", flush=True)
        print(f"BOXCONTROL recipe_difference device_ms={difference['device_ms']} wall_ms={difference['wall_ms']} pct={difference['pct']} "
              f"between {arms[0]} and {arms[1]}", flush=True)

    def gate(self, arm: str):
        # the study's own arms carry their batch in BATCH_ARMS; the batch-8 recipe diagnostic's four candidates
        # (X8_CANDIDATES) are all eight-request arms, so they gate at B = 8
        b = next((b for b, a in BATCH_ARMS.items() if a == arm), 8 if arm in X8_CANDIDATES else None)
        if b is None:
            raise SystemExit(f"gate: {arm} is not a batch arm and not a batch-8 candidate")
        out = self.fresh(self.logs / f"bgate_{arm}.json")
        r = self.etx(b, f"bgate_{arm}", arm=arm)
        write_json(out, dict(arm=arm, batch=b, passed="error" not in r, result=r))
        print(f"BGATE arm={arm} b={b} {'PASS' if 'error' not in r else 'NOT COMPLETE: ' + r['error']}", flush=True)

    def vsel(self):
        out = self.fresh(self.logs / "vsel.json")
        round0 = {c: self.vllm(c, f"vsel_{c}") for c in CONFIGS}
        sel = batch.select({c: r for c, r in round0.items() if "error" not in r}, self.batches)
        write_json(out, dict(round0=round0, selection={str(b): s for b, s in sel.items()}))
        for b, s in sel.items():
            print(f"VSEL b={b} " + (f"baseline={s['config']} step {s['step_ms']} ms" if s["config"] else
                                    f"NO LEGAL BASELINE {s['reasons']}"), flush=True)

    def rounds(self):
        out = self.fresh(self.logs / "brounds.json")
        sel = {int(b): s for b, s in json.loads((self.logs / "vsel.json").read_text())["selection"].items()}
        def passed(arm):                 # a gate that never ran (its build failed) did not pass
            f = self.logs / f"bgate_{arm}.json"
            return f.exists() and json.loads(f.read_text())["passed"]
        gated = {b for b, a in BATCH_ARMS.items() if b == 1 or passed(a)}
        configs = list(dict.fromkeys(s["config"] for s in sel.values() if s["config"]))
        order = [*(("etx", b) for b in self.batches if b in gated), *(("vllm", c) for c in configs)]
        rounds = []
        for k in range(1, batch.ROUNDS + 1):
            r = dict(etx={}, vllm={})
            for kind, x in (order if k % 2 else order[::-1]):
                if kind == "etx":
                    r["etx"][x] = self.etx(x, f"b{k}_{BATCH_ARMS[x]}")
                else:
                    r["vllm"][x] = self.vllm(x, f"b{k}_vllm_{x}")
            rounds.append(r)
            write_json(out.with_suffix(".partial.json"), dict(rounds=rounds))
        write_json(out, dict(order=[list(o) for o in order], skipped=[BATCH_ARMS[b] for b in self.batches if b not in gated],
                             rounds=[dict(etx={str(b): e for b, e in r["etx"].items()}, vllm=r["vllm"]) for r in rounds]))
        print(f"BROUNDS {batch.ROUNDS} rounds; arms {[BATCH_ARMS[b] for b in self.batches if b in gated]}; vLLM {configs}", flush=True)

    def report(self):
        from .batch_floor import step_bytes
        sel = {int(b): s for b, s in json.loads((self.logs / "vsel.json").read_text())["selection"].items()}
        raw = json.loads((self.logs / "brounds.json").read_text())["rounds"]
        rounds = [dict(etx={int(b): e for b, e in r["etx"].items()},
                       vllm={c: (v if "error" in v else {int(b): x for b, x in v.items()}) for c, v in r["vllm"].items()})
                  for r in raw]
        res = batch.study(rounds, sel, self.batches)
        routing = self.logs / "routing.json"
        union = json.loads(routing.read_text()).get("executed_prefix", {}) if routing.exists() else {}
        kept = res.get(1, {})
        for b, rec in res.items():
            extra = None
            if str(b) in union:
                extra = step_bytes(b, union=float(union[str(b)]))["batched"] - step_bytes(1)["one"]
            kept_step = batch.median_text(kept["etx_step_ms"]) if kept.get("status") == "COMPLETE" else None
            rec["next"] = (batch.build_decision(rec, kept_step, extra) if kept_step else
                           dict(next="the kept batch-1 arm is NOT COMPLETE: no projection"))
        write_json(self.logs / "batch_report.json", dict(condition="docs/deepseek/BATCH-2026-10-14.md", union=union,
                                                          batches={str(b): r for b, r in res.items()}))
        for b, rec in res.items():
            if rec["status"] != "COMPLETE":
                print(f"BREPORT b={b} {rec['status']} next: {rec['next']['next']}", flush=True)
                continue
            t, l = rec["throughput"], rec["latency"]
            print(f"BREPORT b={b} vs vLLM {rec['config']}: throughput {t['outcome']} median {t['median_pct']:+.2f}% "
                  f"spread {t['spread_pct']:.2f}% ({rec['etx_tokens_per_s']:.0f} vs {rec['vllm_tokens_per_s']:.0f} tok/s); "
                  f"per-request latency {l['outcome']} median {l['median_pct']:+.2f}% spread {l['spread_pct']:.2f}%; "
                  f"next: {rec['next']['next']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("operation", choices=("gate", "boxcontrol", "vsel", "rounds", "report"))
    ap.add_argument("arm", nargs="?")
    a = ap.parse_args(argv)
    if a.operation != "gate" and a.arm is not None:
        ap.error("an arm argument is only valid with gate")
    s = BatchSession()
    if a.operation == "gate":
        allowed = [BATCH_ARMS[b] for b in s.batches if b > 1] + list(X8_CANDIDATES)
        if a.arm not in allowed:
            ap.error(f"gate arm must be one of {allowed}")
        s.gate(a.arm)
    else:
        getattr(s, a.operation)()


if __name__ == "__main__":
    main()
