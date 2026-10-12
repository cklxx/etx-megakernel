"""Trace-calibrated replay of the DeepSeek plan through etx.sim.xcd: the "simulated plan" basis of stage_table.py.

Every task gets its MEASURED body (done - ready) as a fixed cost, and etx.sim.xcd.ReplayTiming carries the measured
residuals: the worker's fetch after a task (next taken - done) and the task's ready delay beyond its inputs and its
worker (ready - max(last input event fired, taken)). With handoff and release at 0 (the residuals hold them), the
replay reproduces the measured schedule; `check` asserts it. A what-if then shrinks chosen task bodies and keeps every
residual (fetch, notification, ready delay are moved, never deleted), so the dependency graph decides how much of a
body saving reaches the token's end.

  python -m examples.dsv2lite.replay          # calibrate on the stride-32 trace and print the reproduction error
"""
from __future__ import annotations

import json
from pathlib import Path

from etx.sim.xcd import Cost, Hw, ReplayTiming, simulate
from etx.tools import timeline as tl
from examples.dsv2lite.pinned import data_sets


class Replay:
    def __init__(self, plan_path: Path, trace_path: Path):
        self.plan = tl.load_plan(str(plan_path))
        raw = tl.read_raw(str(trace_path))
        trace = tl.build_trace(self.plan, raw)
        self.tasks = trace.by_id                                   # id -> measured Task (us, origin = first taken)
        descs = self.plan["descs"]
        n = len(descs)
        missing = [i for i in range(n) if i not in self.tasks]
        if missing:
            raise ValueError(f"{len(missing)} plan tasks have no measured stamps")
        # Static queues: the worker runs its list in order; the measured fetch gap follows each task.
        wpd = self.plan["workers_per_domain"]
        queue_after = [0.0] * n
        self.owner = {}
        for key, ids in self.plan["static_queues"].items():
            d, s = (int(x) for x in key.split(":"))
            for a, b in zip(ids, ids[1:]):
                queue_after[a] = self.tasks[b].taken - self.tasks[a].done
            for t in ids:
                self.owner[t] = d * wpd + s
        # Event fired = last producer's measured done; ready delay = measured ready - max(inputs fired, taken).
        fire = [max(self.tasks[p].done for p in prods) for _, _, prods in self.plan["ev_inst"]]
        ready_after = [0.0] * n
        for t in range(n):
            inputs = max((fire[e] for e in self.plan["task_in"][t]), default=0.0)
            ready_after[t] = self.tasks[t].ready - max(inputs, self.tasks[t].taken)
        self.min_ready_after = min(ready_after)
        first_taken = {}
        for t, w in self.owner.items():
            first_taken[w] = min(first_taken.get(w, float("inf")), self.tasks[t].taken)
        # A negative residual (a stamp before its input fired, clock skew between CUs) cannot be replayed: clamp it
        # and report the clamped total, so the reproduction error says how much skew the calibration absorbed.
        self.clamped_us = -sum(min(0.0, r) for r in ready_after)
        self.timing = ReplayTiming([max(0.0, q) for q in queue_after], [max(0.0, r) for r in ready_after],
                                   {w: first_taken[w] for w in first_taken})
        self.names = {t["id"]: name for name, t in self.plan["types"].items()}
        self.measured_span = max(t.done for t in self.tasks.values()) - min(t.taken for t in self.tasks.values())

    def body(self, t: int) -> float:
        task = self.tasks[t]
        return task.done - task.ready

    def run(self, saving_us=None) -> float:
        """Makespan with each task's body reduced by saving_us(type name, task id) (never below zero)."""
        def cost_of(name, desc):
            t = desc["id"]
            cut = saving_us(name, t) if saving_us else 0.0
            return Cost(chain_us=max(0.0, self.body(t) - cut))
        hw = Hw(handoff_us={"DOMAIN": 0.0, "DEVICE": 0.0}, release_us=0.0, lat_us=0.0)   # residuals carry the handoffs
        rows, stats = simulate(self.plan, cost_of, hw, replay=self.timing)
        if stats["unfinished"]:
            raise RuntimeError(f"replay left {stats['unfinished']} tasks unfinished")
        return stats["makespan_us"]


def default_replay() -> Replay:
    sess = data_sets() / "results_2026-09-27d" / "sess2"
    return Replay(sess / "x1s_plan.json", sess / "x1s_trace.bin")


def check(replay: Replay) -> dict:
    simulated = replay.run()
    return dict(measured_span_us=replay.measured_span, replay_span_us=simulated,
                error_us=simulated - replay.measured_span, clamped_skew_us=replay.clamped_us,
                min_ready_residual_us=replay.min_ready_after)


if __name__ == "__main__":
    print(json.dumps(check(default_replay()), indent=2))
