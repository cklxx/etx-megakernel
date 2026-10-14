# Five team proposals, 2026-10-14: classification against what is measured

Forwarded by kailun. The proposals are quoted verbatim in the request; one-line paraphrases appear below. Labels: [M] measured,
[E] estimate, [U] unknown. All proposals concern Qwen3-30B-A3B (128 experts) at bf16, batch 1, unless a row says
otherwise.

The KEEP bar is 0.030 ms per token, which is 0.625 us per layer over 48 layers.
- **Classes:** implemented and kept; measured and lost; genuinely open.
- **Where open proposals go:** the four open ones are listed in docs/EXPERIMENTS-2026-10-14.md section "Open proposals"
  with rules written before any code.
- **None is built, queued or run.**

| # | Proposal | Class | Deciding evidence | Index entry |
|---|---|---|---|---|
| 1 | Router: 128 useful tasks, 16 per XCD | Narrowing **implemented and kept**; the grid and placement part is **open, not worth a box** | Kept narrowing: C1 + C5, +0.0626 + 0.0556 ms [M] (`docs/GAINS-LEDGER.md:97`). Grid 256 with 128 empty tasks: `examples/vllm_llm/model.py:298-304`, `etx/ops/gemv.py:783` | P1 |
| 2 | Pre-shuffle weights | **Implemented and kept**; the rest is covered by the queued `moerows` study or closed by the bar | Qwen: GU1 shipped (`docs/MASTER-TABLE-2026-10-11.md:445`). DeepSeek: Fleet's packer (`fleet-3fb67a6/src/host/pack_weights.py:71`) | none |
| 3 | Runtime bundle: epoch events, static-only reset, device token loop | Device loop **closed by the bar**; device reset **measured and lost**; epoch + static-only reset **open, not worth a box** | 0.012 ms [M], Fleet 0.005 ms [M]; −0.008 ms wall [M] (`docs/RESULTS-2026-10-01.md:107`) | P3 |
| 4 | Producer-side combine of the 8 expert outputs | Two forms **measured and lost**; the row-partitioned form is **open, gated on attribution** | Separate combine phase +4.10 us/layer (`docs/PHASES-2026-10-01.md:33`). Fleet one-writer fold 4.80 vs 4.38 / 4.06 ms (`fleet-mi300x/docs/STATUS.md:66,68`) | P4 |
| 5 | Last router task does the top-k | Fleet's form **measured and lost**; the Qwen form is **open, microbenchmark-gated** | Fleet `--topk-published` slower, router 12 → 18 us (`fleet-mi300x/docs/STATUS.md:29`) | P5 |

## 1. Router: schedule only the 128 useful tasks, 16 per XCD

**Already implemented and kept.**
- `ETX_VL_ROUTER_NARROW=1` is in the kept environment (`examples/vllm_llm/kept_build.py`).
- It gives each of the 128 useful tasks one row and a body specialised for H=2048, E=128 (`etx/ops/gemv.py:779-806`).
- It is banked with the norm hoist C5: 0.0626 + 0.0556 ms [M] (`docs/GAINS-LEDGER.md:97`).

**The literal remainder is true today.**
- The grid is still `(CU, 1, 1)` = 256 tasks (`examples/vllm_llm/model.py:303`).
- Tasks 128-255 return at once (`etx/ops/gemv.py:783`).
- The 128 real rows sit on XCD 0-3, 32 per XCD (`docs/XCD-TIMELINE-2026-10-11.md:60`).

**Why the remainder does not reach the bar** [E]:
- **The empty tasks are off the critical path.** They take 0.28 us, against a 5.20 us body on XCD 0-3 [M]. They are
  "not a 256-way serial cut" (`docs/LEVERS-MORE-2026-10-12.md:134`). Each runs on an otherwise idle worker of XCD 4-7.
- **The router's stage time is one task's serial chain** (norm → block sum → weight load → dot), not its bytes:
  - working tasks take 5.44 us at the median [M];
  - block 0, with 8 KB more stores, is never the path task (`docs/MASTER-TABLE-2026-10-11.md:266-268`).
  - Spreading the rows to 16 per XCD does not shorten that chain: every task already has its own compute unit.
- **The router → top-k event does not change.** It still needs all 128 producers on every XCD, so it stays a
  device-wide event.

**What would justify a GPU study:** a trace (X1/X6, `docs/MASTER-TABLE-2026-10-11.md:380,385`) showing that the router
path task's body exceeds the median working body by ≥ 0.625 us/layer. That would mean contention among the 32 tasks
sharing one XCD's L2, which spreading would cut. Without that, the expected gain is about 0 [E].

**Existing coverage:**
- The CU/workgroup count is `ETX_VL_CU=256` (`docs/OPTIMISATIONS.md:25`). It is the reason the grid is 256.
- Changing the count for one phase is this proposal, not a new count study.

## 2. Pre-shuffle weights

**Already implemented and kept for both models.**
- **Qwen:** the MoE class III levers SCALAR, GU1 and WSUM shipped; the class is closed
  (`docs/MASTER-TABLE-2026-10-11.md:445`). GU1 lays gate and up out so one task streams both rows together.
- **DeepSeek (Fleet's tiles):** the packer interleaves gate and up offline (`interleave_gate_up`,
  `~/code/etx-cache/fleet-3fb67a6/src/host/pack_weights.py:71-75`).
- **Every dense GEMV task** already streams a contiguous row block in its own order (one task per workgroup,
  `examples/vllm_llm/model.py:58-61`).

**The remaining layout questions are covered or closed.**
- **Which workgroup owns which expert rows** (contiguous vs interleaved) is the queued study 5, `qwen_moerows`
  (docs/EXPERIMENTS-2026-10-14.md). Do not add a duplicate.
- **The LDS-side layouts** are priced below the bar: the XOR layout S at 0.0234 ms and segment reuse X at 0.0345 ms
  (`docs/LEVERS-MORE-2026-10-12.md:103-104`). The MoE-up body row A6 is "Met: named mechanism 0.023 < 0.030"
  (`docs/PLAN-300-CERTAIN-2026-10-12.md:46`).

**What would justify a new experiment:** a layout that changes HBM traffic. `moerows` decides exactly that. A pass
there (median ≥ 3%) would be the trigger for a full-model layout arm. A re-shuffle inside the current ownership would
not be.

## 3. Runtime bundle: epoch-tagged events, static-only reset, device-side token loop

**Device-side token loop: closed by the bar.**
- The increment beyond the reset term is 0.012 ms [M] (launch 11.994 us + residency 0.474 us,
  `results_nvref_2026-10-09/hostprof.log:2120`).
- Fleet's only precedent is 0.005 ms/token [M] (one launch per token 3.665 ms vs one per 32 tokens 3.660 ms,
  `~/code/fleet-mi300x/docs/STATUS.md:175-176`).
- Sources: `docs/PLAN-300TOKS-2026-10-11.md:194`; row H in `docs/PLAN-300-CERTAIN-2026-10-12.md:54`.
- It reopens only on ≥ 0.030 ms of attributable launch, completion or readback round trips.

**Device reset (the kernel rewrites its own per-step state): measured and lost.**
- `ETX_DEV_RESET` gave −0.008 ms wall (0.16%) [M] (`docs/RESULTS-2026-10-01.md:107`).
- The host's 40.8 us of enqueues disappeared, but:
  - 14 us returned as device time (one workgroup copies the initial images);
  - the rest was absorbed by the wait.
- It is off and listed as "do not resurrect without a new mechanism" (`docs/OPTIMISATIONS.md:26`).

**Epoch-tagged events with static-only reset: genuinely open, not worth a box now.**
- **Never built.** The epoch field is never set (`docs/SYNC-TAXONOMY-AUDIT-2026-10-11.md:43`), and the host resets
  every event, queue and slot each step (`etx/runtime/include/etx/host.h:259-312`).
- **Ceiling:** reset submission 0.0386 ms [M] in HOUR/W and 0.0380 ms [M] in HOUR/J
  (`docs/PLAN-300-CERTAIN-2026-10-12.md:41,56-59`).
- **The `ETX_DEV_RESET` decomposition is the closest measured precedent.**
  - Epochs avoid the 14 us of returned device time, because nothing is copied.
  - The absorbed part would be absorbed again.
  - Best case: 0.008 + 0.014 = **0.022 ms [E from M]**, below 0.030.
  - The `d_ctrl` reset stays in any design: the host reads `ctrl_done` (`host.h:276`).

**What would justify building it:** a diagnostic on the kept build showing that ≥ 0.030 ms of wall time follows the
reset submissions. For example, wall minus kernel falls by ≥ 0.030 ms in a run whose reset work is pre-enqueued
before the timer. That diagnostic is cheap (host only, rides any Qwen box). The epoch build itself is runtime and
kernel work with epoch, abort and rollover proofs (`docs/PLAN-300TOKS-2026-10-11.md:193`).

## 4. Combine the eight expert outputs once, on the producer side

**Two forms already measured and lost.**
- **A separate combine phase.** vLLM's `moe_sum_k8` after `fused_moe_down` costs 4.10 us/layer more than ETX's fold
  into the consumer [M] (`docs/PHASES-2026-10-01.md:33`). ETX deleted the phase (`docs/SYNC-TAXONOMY-AUDIT-2026-10-11.md:39`).
- **One writer combines** (Fleet v0.11, "fold once by the last-arriving down worker"): 4.80 ms vs 4.38 ms without
  tiling, and 4.06 ms in v0.12, where every consumer folds in parallel [M] (`~/code/fleet-mi300x/docs/STATUS.md:66-68`).
  - Fleet's serial one-writer fold took ~4 us, against ~1 us for the parallel consumer fold.
  - Fleet's v0.12 fold is what ETX does now: all 256 qkv tasks sum the residual and 8 slot rows (QK2,
    `docs/MASTER-TABLE-2026-10-11.md:185,195`).
  - A single last-arriver publisher "would serialize the fold" (`docs/LEVERS-MORE-2026-10-12.md:152`).
- **Atomics into one vector.** Not considered: the order of floating-point additions would vary between runs, which
  breaks bit-reproducibility.

**The untried form is open, gated on attribution.** moe_down tasks partitioned by output row would sum all 8 experts
for their rows and add the residual: no atomics and no single writer.
- **Ceiling:** QK2's share of the qkv body, at most the qkv body allowance of 1.51 us/layer = **0.073 ms [E]**
  (`docs/MASTER-TABLE-2026-10-11.md:411`; A4 in `docs/PLAN-300-CERTAIN-2026-10-12.md`).
- **The cost it adds** [U]:
  - each moe_down task would wait for all 8 experts' moe_up outputs, not for its own;
  - the XCD-local moe_up → moe_down edge (DOMAIN, 2.60 us/layer [M], `docs/DEP-GAP-LEDGER-2026-10-11.md:103`) would
    become a device-wide one;
  - cross-XCD DEVICE edges measure 1.53-2.67 us/layer [M] (rows 3, 5 and 7 there).
- **What would justify building it:** X6 in-body stamps (`docs/MASTER-TABLE-2026-10-11.md:385`, not built) showing
  QK2 exposed ≥ 0.625 us/layer on the kept build. Without X6, the ceiling is only an upper bound, and the added edge
  could cancel it.

## 5. Last router task to finish performs the top-k

**Fleet measured its form and lost.**
- `--topk-published` ("top-k published once per XCD by the router's last arriver") measured 3.670-3.678 ms, against
  3.661-3.665 ms [M]: slower (`~/code/fleet-mi300x/docs/STATUS.md:29`; listed as done and slower at :520-521).
- The gate_up prologue fell 5.7 → 1.5 us, but the router phase grew 12 → 18 us: one worker reduced while 36 waited.
- In Fleet the routing had been computed in parallel in every consumer's prologue, so the last arriver added serial
  work to the path.

**The Qwen form differs, so it is genuinely open.**
- ETX's Qwen graph already runs top-k as a separate serial task, one replica per XCD, behind a 256-producer device event
  (`docs/DEP-GAP-LEDGER-2026-10-11.md:109,114`).
- A last-arriver fold adds no serial work. It removes the router → top-k handoff: **74.0 us/step [M]**, 59.1 cross-XCD
  + 14.9 same-XCD, 1.54 us/layer, always "consumer already waiting" (`docs/DEP-GAP-LEDGER-2026-10-11.md:200`).

**Costs, two of them** [E from M]:
- **The top-k → moe_up edge turns cross-XCD.** The router's rows sit only on XCD 0-3, so the top-k result must reach
  all 8 XCDs. Today the edge is same-XCD DOMAIN at 1.08 us/layer (row 10). Measured cross-XCD DEVICE edges are
  1.53-1.78 us/layer (rows 5 and 7). That is +0.45-0.70 us/layer.
- **Visibility is still paid.** The last arriver must still see the other 127 producers' scores from other XCDs.
  Cross-XCD one-way visibility is 0.74-0.9 us [M] (`docs/RESEARCH-sync-hardware.md:24`).

**Net:**
- Upper bound, if the whole handoff disappears: 74.0 − 33.6 … 21.6 = **0.040-0.052 ms**.
- If the visibility round trip remains: **−0.003 … +0.017 ms**.

The difference is a property of the sync path that a microbenchmark measures in minutes.

**What decides it before any build:** a `bench/sync/sync_bench.hip` mode, not yet written. 128 producers on 4 XCDs
each write one score. The last arriver, detected by the returned count of the arrival atomic, acquires the scores and
publishes a 64-B record to one consumer per XCD. Compare against today's arrive → poll → consumer → publish chain on
the same box.
- **GO** for a model arm only if the modeled per-layer saving is ≥ 0.625 us.

## Counts after this change

**docs/EXPERIMENTS-2026-10-14.md lists 11 entries.**
- **7 prepared studies:** built, queued, rules committed. **0 GPU-measured.**
- **4 open proposals:** design only, not built, not queued.
- The batch 2/4/8 study is separate: prepared, not run, not counted there.

**DeepSeek four-column log** (`examples/dsv2lite/results/EXPERIMENT-LOG.md`): **29 rows.**
- 21 GPU-measured.
- 3 measured locally only.
- 5 not yet run: K1, M1, batch 2/4/8, noQ, noA.

**Qwen four-column log** (`examples/vllm_llm/results/EXPERIMENT-LOG.md`): **6 rows, none run on the GPU.**
- 2 queued: `kept_l1`, `moerows`.
- 4 design only: P1, P3, P4, P5.
