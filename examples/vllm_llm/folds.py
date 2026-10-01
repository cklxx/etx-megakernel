"""Which producer phases to fold into their consumers, derived from the model's shapes instead of switched by hand.

A fold deletes a one-block producer phase (its floor, its body and its hand-off) and pays for it by having every task
of the consuming phase redo the producer's work. That trade is model-dependent: it paid for the input norm and lost
badly for the MoE top-k on Qwen3-30B-A3B (measured 2026-10-01, rounds 7 and 25), and nothing about the two phases
themselves says which way it goes -- only their shapes do. This module makes the comparison explicit.

    keep_us  = phase floor x (producer tasks / workers) + the producer's own body + one hand-off
    fold_us  = rounds x (rematerialised bytes / rate)      bandwidth: every consumer task re-reads the inputs
             + hotspot_us(consumer tasks, shared bytes)    contention: many tasks reading one small buffer uncached
             + rounds x serial_us                          latency: dependent work on wave 0 while the others wait

CONSTANTS AND WHERE THEY COME FROM (docs/BUDGET-2026-10-01.md unless noted):
  PHASE_FLOOR_US 2.64   a full-device phase costs this before any bytes, at the tiles' 1024-thread width (round 21;
                        the 3.0 us figure in the budget's text is the 256-thread measurement)
  HANDOFF_US     1.00   one event edge, 9 per layer x 1.00 = 0.433 ms/token
  RATE_B_US      12354  bytes per microsecond per workgroup at width 256 (interpolated +-7%; 19931 measured at 192)
  HOP_US         0.690  one dependent round trip under load (hw_bench lat, both machines)
  HOTSPOT_US_PER_KTASK  calibrated, see below -- the one term with a single calibration point

CALIBRATION. Two measured fold outcomes anchor this model, and the hot-spot term is fitted to the second:
  input norm, Qwen3-30B-A3B: fold 3.75 us/layer against keep 5.9 -> fold wins by 2.15 (rounds 25, 27)
  MoE top-k,  Qwen3-30B-A3B: fold 14.1 us/layer against keep ~8.6 -> keep wins by 5.5 (rounds 7, 26)
The bandwidth term alone predicts the norm (36864 B/task / 12354 = 2.98 us against 3.75 measured, within 20%) but
predicts ~0.02 us for the top-k, which is wrong by three orders of magnitude. The difference is that 448 expert tasks
read the SAME 256 B of logits with uncached 2-byte accesses -- a coherent access bypasses L1 and L2, so there is no
sharing and the requests serialise at the memory system. That cost scales with the number of consuming tasks, not
with bytes, and it is the term that decides the top-k. It has one calibration point; treat a prediction that turns on
it as a hypothesis to measure, not a settled answer.

WHAT WOULD FALSIFY THIS. A fold candidate whose predicted and measured costs differ by more than the ~20% the norm
row shows, or any model where the hot-spot term decides a fold and the measurement disagrees. Both are worth a
20-second ablation before trusting the decision (etx/ops/gemv.py carries DSL_ABLATE_* for exactly this).
"""
from __future__ import annotations

from dataclasses import dataclass

PHASE_FLOOR_US = 2.64
HANDOFF_US = 1.00
RATE_B_US = 12354.0
HOP_US = 0.690
WAVE = 64
# Fitted so the MoE top-k candidate reproduces its measured 14.1 us/layer at 448 consumer tasks reading 256 B.
# 14.1 = rounds(2) x serial + hotspot; with serial = TOPK x log2(WAVE) x HOP_US/6 = 5.52 us for TOPK 8, the
# remainder over 448 tasks gives this coefficient. Single calibration point -- see the module docstring.
HOTSPOT_US_PER_KTASK = 6.9


@dataclass(frozen=True)
class Fold:
    """One candidate: delete `producer` and have every task of `consumer` redo its work."""
    name: str
    producer_tasks: int          # tasks the producer phase would run (an x-form replica is NX, one per XCD)
    producer_body_us: float      # its own body, excluding the floor and the hand-off
    consumer_tasks: int          # tasks that would each redo the work
    workers: int                 # workers available, so rounds = ceil(consumer_tasks / workers)
    remat_bytes: int             # bytes EACH consumer task re-reads
    shared_bytes: int            # of those, bytes every task reads from the SAME small buffer (the hot spot)
    serial_steps: int = 0        # dependent steps on one wave while the rest wait (a wave reduction is log2(WAVE))

    @property
    def rounds(self) -> int:
        return max(1, -(-self.consumer_tasks // self.workers))

    def keep_us(self) -> float:
        # The floor is a FULL-device cost; an x-form replica occupies only `producer_tasks` of the workers, and round 2
        # measured a narrow phase's fixed cost at about 0.2 us, not 2.64. Scaling by occupancy reproduces both measured
        # phases: top-k 9.04 against 8.60 measured, in_norm 5.98 against 5.9.
        occupancy = min(1.0, self.producer_tasks / self.workers)
        return PHASE_FLOOR_US * occupancy + self.producer_body_us + HANDOFF_US

    def fold_us(self) -> float:
        bandwidth = self.rounds * (self.remat_bytes / RATE_B_US)
        hotspot = (self.consumer_tasks / 1000.0) * HOTSPOT_US_PER_KTASK if self.shared_bytes else 0.0
        serial = self.rounds * self.serial_steps * (HOP_US / 6.0)
        return bandwidth + hotspot + serial

    def decide(self) -> bool:
        return self.fold_us() < self.keep_us()

    def explain(self) -> str:
        v = "FOLD" if self.decide() else "KEEP"
        return (f"{self.name:<12} {v}  fold {self.fold_us():6.2f} us  keep {self.keep_us():6.2f} us  "
                f"(rounds {self.rounds}, remat {self.remat_bytes} B, shared {self.shared_bytes} B, "
                f"serial {self.serial_steps})")


def candidates(d, cu: int, nx: int) -> list[Fold]:
    """The fold candidates of one decoder layer, from the model's dimensions.

    `d` is examples.llm.model.load_dims' result: H, NH, NKV, HD, moe, E, TOPK, MI. `cu` is the dense phase width and
    `nx` the number of XCDs, so an x-form replica producer runs `nx` tasks and a dense consumer `cu`.
    """
    out: list[Fold] = []
    H = d.H
    # in_norm -> qkv: every qkv task re-reads the residual plus, for an MoE model, the TOPK expert rows it sums.
    rows = (d.TOPK if d.moe else 1) + 1
    out.append(Fold("in_norm", producer_tasks=nx, producer_body_us=4.9, consumer_tasks=cu, workers=cu,
                    remat_bytes=rows * H * 2, shared_bytes=0))
    # post_norm -> router: the residual plus the attention output, one row each.
    out.append(Fold("post_norm", producer_tasks=nx, producer_body_us=3.9, consumer_tasks=cu, workers=cu,
                    remat_bytes=2 * H * 2, shared_bytes=0))
    if d.moe:
        # topk -> every expert task: the same E logits, read uncached by all of them, plus a TOPK-deep selection
        # whose each step is a full log2(WAVE) shuffle reduction on wave 0 (etx/ops/gemv.py dsl_topk_lds).
        experts = _moe_task_count(d, cu)
        out.append(Fold("topk", producer_tasks=nx, producer_body_us=7.96, consumer_tasks=experts, workers=cu,
                        remat_bytes=d.E * 2, shared_bytes=d.E * 2,
                        serial_steps=d.TOPK * 6))
        # moe_sum -> the next layer's in_norm: already counted as in_norm's TOPK rows, so folding it is what makes
        # in_norm's remat what it is. Kept here for the record; it is not an independent decision.
    return out


def _moe_task_count(d, cu: int) -> int:
    """Tasks across the two expert phases, which both consume the top-k (model.py's MoE grids)."""
    up = d.TOPK * (d.MI // 32) if d.MI else 0
    down = d.TOPK * (d.H // 64)
    return up + down


def decide(d, cu: int, nx: int) -> dict[str, bool]:
    return {f.name: f.decide() for f in candidates(d, cu, nx)}


if __name__ == "__main__":                      # python -m examples.vllm_llm.folds [config.json ...]
    import sys
    from examples.llm.model import load_dims
    for p in sys.argv[1:] or ["examples/llm/configs/qwen3-30b-a3b.json"]:
        d = load_dims(p)
        print(f"== {p}  H={d.H} moe={int(d.moe)}" + (f" E={d.E} TOPK={d.TOPK} MI={d.MI}" if d.moe else ""))
        for f in candidates(d, 256, 8):
            print("  " + f.explain())
