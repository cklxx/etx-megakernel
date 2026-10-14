# Experiment index, 2026-10-14: DeepSeek and Qwen, seven studies, no GPU yet

Labels: [M] measured, [E] estimate, [U] unknown. bf16, batch 1, context 1024, 32 tokens throughout. Larger batches
(2/4/8) are the next phase: neither ready nor tested.

**Count:** 7 experiments = 6 decision studies + 1 attribution-only trace.
- **DeepSeek (5):** K1, M1, two ablations, one trace.
- **Qwen (2):** moerows, the common cache mode.
- **Not counted:** `certify_kept` / `certify_kept_l1` (correctness gates), `moegather` (moerows' cross-check),
  preflights and builds.

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
