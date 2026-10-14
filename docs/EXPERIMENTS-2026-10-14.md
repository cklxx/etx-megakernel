# Experiment index, 2026-10-14: DeepSeek and Qwen, seven studies and four open proposals, no GPU yet

Labels: [M] measured, [E] estimate, [U] unknown. bf16, batch 1, context 1024, 32 tokens throughout. Larger batches
(2/4/8) are a separate, prepared study with its own session and rules: docs/BATCH-2026-10-14.md,
`examples/sessions/ds_batch.py` (DeepSeek; not run, not counted below).

**Count:** 7 experiments = 6 decision studies + 1 attribution-only trace.
- **DeepSeek (5):** K1, M1, two ablations, one trace.
- **Qwen (2):** moerows, the common cache mode.
- **Not counted:** `certify_kept` / `certify_kept_l1` (correctness gates), `moegather` (moerows' cross-check),
  preflights and builds.
- **Also listed, not counted as studies:** 4 open team proposals (P1, P3, P4, P5), design only: not built, not queued
  (section "Open proposals"; classification of all five in docs/PROPOSALS-2026-10-14.md).

**Four-column logs:** DeepSeek `examples/dsv2lite/results/EXPERIMENT-LOG.md`; Qwen `examples/vllm_llm/results/EXPERIMENT-LOG.md`.

**Queues:**
- DeepSeek alone: `examples/sessions/ds_k1.py`.
- Qwen alone: `examples/sessions/qwen_g6.py`.
- Both on one box: `examples/sessions/joint_ds_qwen.py`. It builds no job of its own; it combines the two files'
  jobs, with Qwen prefixed `qwen_` and recorded under `qwen/`.

## The seven

**1. DeepSeek K1: kv_a computed once** (`pairs_alld4r0l1kvs`)
- **Mechanism:** each XCD computes 1/8 of kv_a behind a global qkv event, instead of 8 full replicas.
- **Rationale:** qkv is 21.5 us vs a 3.5 us floor [C]. Issued bytes explain it (MODEL-dsv2.md:154), and the replica
  half was never measured.
- **Status:** built, plan and ISA checked; kept ISA byte-identical; K1 0 scratch, buffer_wbl2 +54.
- **Baseline:** `alld4r0l1` (kept), device time.
- **Rule (2ec2d29):** KEEP if every pair is faster, median >= 30 us and > 2 x spread; otherwise DROP.
- **GPU-only unknown:** the net sign (qkv -5..-7 vs handoff/attention +1..+4 us per layer, both [E]).
- **Sources:** DSV2-D1-D5.md "K1"; tests/test_dsv2_kva.py.

**1b. DeepSeek M1: merge's W_UV GEMV on all four waves** (`pairs_alld4r0l1mw`)
- **Mechanism:** lever `merge_uv_waves`, `gemv_rows_pipelined<2, 1>` instead of `<8, 1>`.
- **Rationale:** M3 is 5.24 us per task, and wave 0 alone does all 8 rows while 3 waves idle (OPS-dsv2.md:82).
- **Status:** built (ROCm 7.2.4: 78 AGPR, 0 scratch, plan unchanged). Per-wave M3 code 1082 -> 365 lines: weight loads
  24 -> 6, reduction shuffles 96 -> 24. Outputs bit-identical by construction.
- **Baseline:** `alld4r0l1`.
- **Rule (3d27ff8):** KEEP if every pair is faster, median >= 30 us and > 2 x spread; otherwise DROP.
- **GPU-only unknown:** whether the 3x shorter per-wave chain shortens the phase (one shared memory round trip could
  dominate). Estimate 27-80 us/token [E].
- **Sources:** DSV2-D1-D5.md "M1"; tests/test_dsv2_lever_merge_uv_waves.py.

**2. DeepSeek ablation: remove gemv_no_terminal_all** (`pairs_alld4r0l1noQ`)
- **Mechanism:** the kept recipe without that lever.
- **Rationale:** it was dead alone and cost 14 us inside `all` in an earlier session, yet it ships in the kept arm.
- **Status:** built; 87 AGPR (+17), 0 scratch.
- **Baseline:** `alld4r0l1`.
- **Rule:** REMOVE if every pair is faster and median > 2 x spread (no 30 us bar); otherwise it stays.
- **GPU-only unknown:** the interaction with l1 + D4r0.
- **Sources:** same.

**3. DeepSeek ablation: remove argmax_tree** (`pairs_alld4r0l1noA`)
- **Mechanism:** the kept recipe without that lever.
- **Rationale:** +2 us, inside the noise, when measured alone; still shipping.
- **Status:** built; +21 barriers, 0 scratch.
- **Baseline:** `alld4r0l1`.
- **Rule:** REMOVE if every pair is faster and median > 2 x spread; otherwise it stays.
- **GPU-only unknown:** the sign. The ceiling is low: argmax is 5.1 us in total [M].
- **Sources:** same.

**4. DeepSeek K1 trace** (`trace_alld4r0l1kvs`)
- **Mechanism:** one stamped token of K1.
- **Rationale:** if K1 drops, it shows whether qkv shrank and attention grew.
- **Status:** stamped build defined in the registry.
- **Rule:** none; attribution only, never a timing.
- **GPU-only unknown:** the stage shifts.
- **Sources:** ds_k1.py.

**5. Qwen moerows: row ownership between MoE workgroups** (`qwen_moerows`, `hw_bench moerows`)
- **Mechanism:** contiguous vs interleaved rows, Qwen's moe_up shape, cold, checksummed. One kernel for both modes.
- **Rationale:** decides Q2 before any hot-loop rewrite. The old 0.50 ms was a cache-contaminated model bound.
- **Status:** device and host compile (ROCm 7.2.4); the CPU coverage/coldness oracle passes.
- **Baseline:** the contiguous mode, same launch pairs.
- **Rule:** PASS if every pair is faster, median >= 3% and > 2 x spread. A checksum mismatch means INCOMPLETE.
- **GPU-only unknown:** HBM / MALL behaviour under each pattern.
- **Sources:** QWEN-READINESS-2026-10-14.md section 4; bench/hw/moerows_oracle.py.

**6. Qwen common cache mode** (`qwen_pairs_kept_l1`)
- **Mechanism:** `ETX_COHERENCE=l1` on the kept build: one compiler switch for both models.
- **Rationale:** the reuse criterion, "one mode, two models, one rule". DeepSeek's half: +330 us, KEEP [M].
- **Status:** built and ISA-checked: 82 system-scope loads move to agent scope; 0 acquires removed.
- **Baseline:** `kept`, device time.
- **Rule:** KEEP if every pair is faster, median >= 30 us and > 2 x spread; otherwise NO-GO. Both builds pass the
  same-build import certificate + 32/32 first.
- **GPU-only unknown:** correctness on the device and any gain. Expect far less than DeepSeek's [E].
- **Sources:** INVENTORY 4.1; tests/test_qwen_g6.py.

## Open proposals (team, 2026-10-14; design only)

Classified in docs/PROPOSALS-2026-10-14.md. Proposal 2 (pre-shuffle weights) is implemented and kept, and its remainder
is covered by study 5. The other four are open in the form stated here; the forms already measured and lost are not
reopened. All four are Qwen, bf16, batch 1. Each rule is written before any code. A model arm needs the same-build
import certificate and 32/32 first.

**P1. Router grid of 128 tasks, 16 per XCD** (`router128`)
- **Mechanism:** a router grid of 128 tasks with rows remapped so each XCD owns 16, instead of 256 tasks with rows on
  XCD 0-3 and 128 empty tasks.
- **Rationale:** the empty tasks and the 4-XCD placement are literal today (`etx/ops/gemv.py:783`,
  `examples/vllm_llm/model.py:303`, `docs/XCD-TIMELINE-2026-10-11.md:60`).
- **Status:** not built. The narrowing itself is kept (C1, +0.0626 ms [M]).
- **Baseline:** `kept`, device time.
- **Rule:** KEEP if every pair is faster, median >= 30 us and > 2 x spread; otherwise DROP.
- **GPU-only unknown:** whether 32 router tasks sharing one XCD's L2 lengthen the path task's body.
- **Box:** **not worth a box.** The empty tasks are 0.28 us and off the path. The stage is one task's serial chain
  (`docs/MASTER-TABLE-2026-10-11.md:266-268`), which placement does not shorten; [E] about 0. Reopens only if an
  X1/X6 trace shows the path task's body >= 0.625 us/layer above the median working body.
- **Sources:** PROPOSALS-2026-10-14.md section 1; `docs/LEVERS-MORE-2026-10-12.md:134`.

**P3. Epoch-tagged events with static-only reset** (`epoch_reset`)
- **Mechanism:** event words compare against a per-step epoch, so the host stops resetting events, queues and slots
  every step. Only static state and `d_ctrl` are written.
- **Rationale:** reset submission is 0.0386 / 0.0380 ms [M] in two sessions (`docs/PLAN-300-CERTAIN-2026-10-12.md:41,56-59`).
- **Status:** not built. The device token loop is closed by the bar (0.012 / 0.005 ms [M]). The device reset was
  measured and lost (−0.008 ms wall [M], `docs/RESULTS-2026-10-01.md:107`).
- **Baseline:** `kept`, **host wall** (a device-time rule cannot see host work).
- **Rule:** KEEP if every pair is faster, median >= 30 us and > 2 x spread; otherwise DROP. Epoch, abort and rollover
  proofs come first.
- **GPU-only unknown:** how much of the reset submission lies on the wall path.
- **Box:** **not worth a box now.** By the `ETX_DEV_RESET` decomposition, the best case is 0.008 + 0.014 =
  0.022 ms [E from M], below the bar. It reopens on a host-only diagnostic on the kept build: wall minus kernel falls
  by >= 0.030 ms when the reset work is enqueued before the timer. That diagnostic is minutes on any Qwen box.
- **Sources:** PROPOSALS-2026-10-14.md section 3; `etx/runtime/include/etx/host.h:259-312`;
  `docs/PLAN-300TOKS-2026-10-11.md:193-194`.

**P4. Producer-side combine, row-partitioned moe_down** (`moe_rowsum`)
- **Mechanism:** each moe_down task owns output rows across all 8 selected experts, sums them and adds the residual
  once. The qkv tasks then read one finished vector instead of folding residual + 8 slot rows (QK2).
- **Rationale:** QK2 is repeated by all 256 qkv tasks (`docs/MASTER-TABLE-2026-10-11.md:185,195`).
- **Status:** not built. A separate combine phase lost 4.10 us/layer [M] (`docs/PHASES-2026-10-01.md:33`). Fleet's
  one-writer fold lost, 4.80 vs 4.38 / 4.06 ms [M] (`~/code/fleet-mi300x/docs/STATUS.md:66-68`). Atomics are
  excluded because they break bit-reproducibility.
- **Baseline:** `kept`, device time.
- **Gate before code:** X6 in-body stamps (`docs/MASTER-TABLE-2026-10-11.md:385`, not built) show QK2 exposed
  >= 0.625 us/layer. Otherwise no build.
- **Rule:** KEEP if every pair is faster, median >= 30 us and > 2 x spread; otherwise DROP.
- **GPU-only unknown:** QK2's exposed time [U] (ceiling 1.51 us/layer = 0.073 ms [E]). Also the cost of turning the
  XCD-local moe_up -> moe_down edge (2.60 us/layer [M]) into a device-wide wait (cross-XCD edges 1.53-2.67 us/layer
  [M], `docs/DEP-GAP-LEDGER-2026-10-11.md:103-109`).
- **Box:** the gate rides the next Qwen box once X6 is built (+5 min, +$0.17 [E], MASTER-TABLE:385). No timed arm
  before the gate passes.
- **Sources:** PROPOSALS-2026-10-14.md section 4.

**P5. Last-arriver top-k** (`topk_lastarr`)
- **Mechanism:** the router task whose arrival completes the 128 producers runs the top-k itself and publishes one
  record to moe_up on all 8 XCDs. This replaces the 8 per-XCD top-k replicas behind the device event.
- **Rationale:** the router -> top-k handoff is 74.0 us/step [M], consumer always already waiting
  (`docs/DEP-GAP-LEDGER-2026-10-11.md:109,114,200`). Unlike Fleet's `--topk-published` (slower, router 12 -> 18 us,
  `~/code/fleet-mi300x/docs/STATUS.md:29`), Qwen's top-k is already serial, so the fold adds no serial work.
- **Status:** not built; the deciding microbenchmark is not written.
- **Baseline:** for the microbenchmark, today's arrive -> poll -> consumer -> publish chain on the same box; for a
  model arm, `kept`, device time.
- **Microbenchmark rule:** GO for a model arm only if the modeled saving is >= 0.625 us/layer after adding the
  cross-XCD top-k -> moe_up edge (+0.45-0.70 us/layer [E from M]).
- **Model rule:** KEEP if every pair is faster, median >= 30 us and > 2 x spread; otherwise DROP.
- **GPU-only unknown:** whether the last arriver still pays the cross-XCD visibility round trip (0.74-0.9 us one-way
  [M], `docs/RESEARCH-sync-hardware.md:24`). Net 0.040-0.052 ms if it does not; −0.003 to +0.017 ms if it does [E].
- **Box:** a `bench/sync/sync_bench.hip` mode (minutes of GPU on any box) once written. It is the only one of the
  four worth preparing next.
- **Sources:** PROPOSALS-2026-10-14.md section 5.

## The joint run (`examples/sessions/joint_ds_qwen.py`)

**GPU lane, strictly in order, never two at once:**
`qwen_moerows` -> `prep_ref` -> DeepSeek's 4 pairs studies -> its trace -> `qwen_stage_pack` -> 2 Qwen certificates
-> `qwen_pairs_kept_l1`.

**Isolation, proven on the real graph** (tests/test_joint_session.py; the runner's own `_state` on settled graphs):
- a failed Qwen download skips only Qwen;
- a failed microbenchmark, or a fully failed DeepSeek, leaves the Qwen cache-mode chain ready;
- no shared names or files;
- the two models' environment variables are disjoint.

**Qwen staging** runs inside the runner (logged and sealed), every step idempotent and verifying its output:
- environment with torch for ROCm;
- vLLM cloned at v0.27.1 and its commit verified;
- HF download pinned to revision `ad44e777...` (recorded by both nvref sessions), refusing an unpinned earlier copy;
- packing on the GPU as before, then the pinned golden, then the config.

## Budget (one MI300X; no provisioning without the owner's approval)

| part | basis |
|---|---|
| DeepSeek prep to preflight | 3 min 59 s on a fresh droplet [M, 2026-10-13]; prep_ref 42 s [M] |
| DeepSeek GPU studies | 40 invocations x 5.9 s + 1 traced token: about 4.1 min [E from M per invocation] |
| Qwen moerows | 10 launches, seconds [E]; job cap 600 s |
| Qwen environment / download / pack / builds | **[U]: no record measured them.** Caps 1800 / 2400 / 2400 / 1500 s (nvref and session caps). The 61 GB download at DeepSeek's measured rate (30 GB in 29 s) would take about 1 min [E] |
| Qwen GPU (2 certificates x 3 + 10 timed invocations) | 16 host invocations: expected about 120 s each (qwen300_hour's step estimate) = about 32 min [E]; worst 16 x 240 s = 64 min (deadlines) |

- **Runner cap:** 7200 s for the joint run, 1800 s for DeepSeek alone.
- **Recommended:** one box, at most **2.5 h including staging, about $5 at $1.99/h [E]**. DeepSeek's four pair studies
  finish within its first ~10 min [E]. The largest uncertainty is Qwen's staging and builds, which are unmeasured.
- **Still missing before any rental:** the owner's approval; and on the box, network access to Hugging Face, PyPI
  (torch for ROCm) and GitHub.
- **Absent on this Mac (not prepared, not needed locally):** staged Qwen weights and a GPU.
