# CPU verification and next-session registration (2026-10-12)

Condition: one MI300X VF, Qwen3-30B-A3B, bf16, batch 1, context 1024,
32 free-running tokens. [M] is retained measurement or a CPU observation with
its source, [E] is an estimate, [U] is unverified and contributes zero.
No GPU, provisioning or GPU Docker execution is part of this change.
The six mechanisms in LEVERS-MORE add zero ledger ceiling. KEEP remains
0.030 ms, every adjacent pair faster, median greater than twice max-minus-min
spread; atom bar 0.625 us/layer, S1 0.088 ms, S2 0.324 ms, cap 8400 s.

## 1. Reset: settled by CPU source reading

`etx/runtime/include/etx/host.h:hour_profile_run` records reset_start on the
stream, starts CPU clock a, calls reset_step, stops CPU clock b, then records
reset_stop. The host-submit mean b-a is **38.63 us (W) / 38.00 us (J)** [M],
32 HOUR-STAGE records (steps 1023..1054), in the two retained report.json files
under `results_qwen300_hour_20261011/session-{wklfw6bc,jc7gb969}`.
The stream event mean is **74.98 us (W) / 73.69 us (J)** [M]. It brackets the
reset commands: ctrl clear, event initial counts, domain slot clears, remaining
counts, per-domain arrival counters, mirror if enabled, local/global queue
initialization; execution-counter clear only when enabled. It includes stream
starvation between host enqueues and event overhead, not just DMA/memset
service. The CPU-submit and event intervals overlap: **do not add them**.

`launch()` records ev_start only after reset_stop and the residency query;
ev_stop follows the megakernel. Thus the 74.98 us is outside kernel_device_us
and outside the token_time prologue→argmax device span. It is inside the
host step w0→wms (`examples/vllm_llm/host.hip`, HOUR-STAGE producer).
Both kept seal and raw/hour_build.json explicitly have ETX_DEV_RESET=0;
the latter is the stamps build, not the kept timing build. Profile numbers
come from stages (ETX_HOUR_PROFILE), whose extra events perturb submission.
They are not a measured ordinary-wall saving.

The old **0.008 ms** is a historical *net wall gain* of ETX_DEV_RESET, not
existing reset device service: `docs/RESULTS-2026-10-01.md:107` explicitly
says -0.008 wall, 40.8 us fewer enqueues, 14 us returned inside the kernel.
It cannot price today's event interval. A1's 0.049 is the old estimated
host-submission ceiling. Account the whole reset path once in A1, replacing
that price only with a fresh ordinary-wall pair; do not add 0.075+0.039 to it.
Host H excludes pending reset service already inside wait-minus-kernel
(`second_stop`), and excludes kernel-outside-token time as a separately
recoverable round trip. This conservative host attribution is not a device
service subtraction proof. Cross-step residency would overlap A1; its old
0.005 estimate is not a bound on reset avoidance. Its actual gain is [U].

### Registered for the session: RESET.keep

One new build directory `reset`, kept recipe with **only ETX_DEV_RESET=1**;
all existing directories and flags stay as registered. `prepare` builds and
seals it before the measurement clock. `reset_pair.check_pair` verifies
compiler, import/DSL/original hashes, overrides and schedule; the generated
reset tail/header are the intentional differences. Import certificates now
record ETX_DEV_RESET rather than silently omitting this plan-affecting flag.
Fresh host --check 48 at 0/1040 and CHECK-DSL/32/32 precede three alternating
adjacent kept/reset pairs (31 post-first-token wall steps each), using
`reset_pair.run` → `candidate_pairs`; no profile/stamps in timing.

Measurement: ordinary wall gain and token-device delta on the same pairs.
Budget: **120 s / 2 min / $0.0663 [E]**, reserved inside B; correctness is
included. A cold load or slow checks may exhaust this, yielding NOT-COMPLETE,
never a shortened population. Expected [E] direction on reset ON: fewer
host enqueues, higher kernel-event time from the reset tail (outside the
token timers), lower wall if reset removal dominates. Reverse the flag and
those signs reverse. Token-device time should be roughly unchanged [E];
resource/scheduling effects are [U]. 74.98 us of wall recovery is not promised.
KEEP only at the unchanged bar; reject returns zero reset credit. Falling
back to host resets or changed math/schedule falsifies this experiment.
A passing singleton is measured again in D before claiming any final target.

## 2. Load phases: implemented and CPU checked

The retained v1/v2 records expose poll, final-arrival, tile-return and drain;
there is **no load phase**. M.J/QK6/OP3 interleave weight-load groups with
M.N/QK7/OP4 dot work. A timestamp around the entire stream would still count
dots and row flushes. 2415.919 MB / 770.3 us = 3.136 TB/s is a body rate [E],
not a measured load rate; 2415.919 MB / 4.3 TB/s = 561.84 us remains a
logical byte floor [E]. Neither decides which work binds this body.

`etx/codegen/load_stamps.py` adds begin→all-values-ready stamps around each
full or tail load group, using each loaded dword as an asm operand before
the end tick so compiler waits must finish first. Dots, norm, reductions,
rounding and SiLU are outside those local brackets. `gemv.emit_tiles` applies
the overlay only with ETX_INBODY_EMIT=1; ordinary generated tiles are unchanged.
ABI fields exist only under ETX_INBODY. One wave owns each diagnostic row.
The CSR buffer selects only moe_up, moe_down, qkv, o_proj, with eight groups
per wave (up needs eight, qkv three, down three, o_proj two at this shape).
Overflow remains visible and invalid, never clipped. Standalone import-check
ctx pointers are null. Load stores run only at the **existing step-1024
snapshot**, not throughout the prompt. v3 adds load CSR to v1; v4 adds it
after the v2 producer ledger. Old versions still read losslessly.

`etx/tools/inbody.py:load_phases` checks every task, wave, order, byte count
and all 48 layers. Per layer task counts/weight bytes per task are
moe_up=192/262144, moe_down=256/98304, qkv=256/81920, o_proj=256/65536,
from MASTER-TABLE M.J/QK6/OP3 and the frozen plan. The numerator counts the
complete grid once. The denominator is the **union of load-group brackets
across concurrent waves/tasks within that layer and role**, summed over
layers, not a sum of parallel durations and not a critical body denominator.
Rows report bytes, load_union_us and logical_TB_s for each layer and role.
These are logical weight rates, not exclusive memory latency or physical
HBM traffic: TOUCH/cache hits, concurrent dot work and timestamp overhead
remain. Rates above hardware bandwidth require auditing those effects.

### Registered for the session: A.loads, zero additional GPU minutes

Command is the existing `python -m examples.sessions.qwen300_hour --analyze
STAMPS/plan.json OUT/inbody.bin OUT/A_edges.json`. Its compact summary now
contains load_phases independently of G; a missing load stamp refuses rates
without erasing valid edge evidence. The already-paid stamps correctness
gate also captures step 1024 with ETX_INBODY_LOADS=0. The same bounded
analysis reports absolute ON/OFF body changes on the union of their critical
task IDs; **>0.5 us/body invalidates the load inference**. This adds no host
invocation, build arm or budget. Original/OFF spread validity is still [U]
(the session only has one noninterleaved control); outputs explicitly stay
non-actionable pending that registered validity gate, rather than silently
claiming a perturbation pass. GPU resource/spill effects of the new stamps
are [U] until native compile and ON/OFF evidence are reviewed.

Decision once validity holds: **>=4.0 TB/s → target non-byte body work;
<=3.3 TB/s → target memory path; (3.3,4.0) → unresolved**. These thresholds
are the owner's task, not new performance acceptance bars. No new ledger
credit; any later implementation must pass KEEP and byte/token gates.
Falsification: incomplete byte/grid coverage, reversed/missing stamp,
overflow, failed correctness or perturbation validity yields NOT-COMPLETE or
INVALID, never a bandwidth number usable for pricing.

CPU command `.venv/bin/python -m pytest -q tests/test_load_phases.py
 tests/test_inbody.py tests/test_qwen300_hour.py`: **137 passed in 0.80s** [M].
Tests cover a healthy concurrent grid, missing stamps/old schema, overflow,
clock order, wrong bytes, missing task, binary truncation and ordinary-source
identity. Local gfx942 compilation of an actual emitted up tile with the
real ROCm device libraries passed [M], using `etx/importer/hip_ir_local.sh`.
The full native stamps build check is recorded in section 5.
Retained `raw/inbody.bin`: reader OK, v1, 60362 tasks [M CPU]; load analysis
refuses "missing load-phase stamps: rebuild ... (inbody v3/v4)". Its unfixed
clock cannot support **any** timing number here.

## 3. X5: CPU inventory executed; isolated probe registered

CPU command:

```sh
.venv/bin/python -m examples.vllm_llm.x5 \
  --inventory /Users/ckl/code/vllm \
  --retained results_qwen300_2026-10-11/bf16_pricing.json \
  --output docs/evidence/verify-x5-2026-10-12.json
```

Observed vLLM source revision **f46a9dfe2c5f57bebbd29556cbbb25eabd874226** [M].
The evidence JSON retains hashes and knob-bearing lines of the installed
sources. A cached AITER unified_attention.py is byte matched by Git blob SHA
against the cached tree at **cdb1950e1dc4d864cae54c8a86a61cde999ce1e7** [M CPU].
It exposes TILE_SIZE, BLOCK_M/Q, num_segments, warps/stages, K_WIDTH and
SHUFFLED_KV_CACHE. **No full AITER installation/source tree or the pinned
image's dispatcher is present on this Mac** [U for their exact candidate
universe]. Matching the local vLLM/current cached AITER to the old pinned
image is also [U]. Completeness is therefore explicitly false; the following
is the source-visible family inventory, not a fabricated exhaustive binary
catalogue. No GPU vendor sweep is repeated.

The retained vendor result is 6.7686176 attention + 4.6516855 reduce =
**11.4203031 us/layer [M intervals, E mean]**, 48 layers × 31 decode samples,
from `bf16_pricing.json` and `job1_pricing.log:47-95`. The task calls it a
22-minute search; that was a budget, not the retained elapsed time:
`status.json` records Job 1 **274.041 s [M]**, including image/download/bench/
trace work. It is S3's AITER pair, not a sweep of all backend knobs.
Its 32/32 full-system gate is retained in bf16_bar.log. This proves that
candidate is slower on the historical cross-basis comparison; nothing else
gets a measured pre-price from it.

| Candidate / source-visible knobs | Pre-price us/layer | Exact-condition legality and next action |
|---|---:|---|
| Native ROCM_ATTN `attn_mfma16_g8` + `attn_reduce`; PART=256, block=16, HD=128, MFMAType::F16 | 10.645 [E critical body] | Legal control, already exported in importer/spec.py; sole currently executable survivor. Measure in isolated harness; not a new vendor win. |
| Native GQA 1..7, 9..16, HD=64, or mfma4 G1..4 | [U] | Excluded: real NH/NKV=32/4=8 and HD=128. Changing template GQA misaddresses heads. Native dispatcher chooses mfma16 for G8; mfma4 G8 is not a registered vendor dispatch/export. |
| Native PART=128 or 512 via host parts | [U] | Excluded: T_PAR_SIZE remains 256 inside body. 128 just adds early-exit tasks; 512 fails maxlen coverage. model.py:74-78 and native attention.cu:362 (retained slice). |
| Native block=32 | [U] | bf16/G8 shape supported, but **not legal under current block-16 import contract**; must repack KV writer/cache strides and tables, change specialization and certify. No ready adapter. |
| Generic paged v1 (no reduce) / v2 (partitioned + reduce) | [U] | Potential bf16 shape candidates; different symbols, Q-head grids, outputs/partials and numerics. Existing ETX export/host contract does not implement them; legality [U], not an exclusion by speed. |
| vLLM TRITON_ATTN 2D / 3D | [U] | Source `ops/triton_unified_attention.py`: num_par_softmax_segments=None vs segmented path, tile sizes, BLOCK_M/Q, launch warps/stages. Flash K/V layout and reduction order require new adapters/independent original checks. No executable contract here. |
| AITER unified 2D / 3D, unshuffled / shuffled KV, TILE_SIZE, BLOCK_M/Q, segment count, warps/stages, K_WIDTH | 11.4203 [M/E] for retained dispatch only; other tuples [U] | Current wrapper `rocm_aiter_unified_attn.py`, cached kernel blob; compatible bf16 GQA family but full dispatch/allowed tuning tuples are [U]. Shuffled/flash layouts are not interchangeable with current KV buffers. |
| AITER `paged_attention_common` HIP/ASM / `pa_fwd_asm` | [U] | `_aiter_ops.py:3000-3087` forwards scales, strides, temporary outputs. Implementations and allowed bf16 dispatch knobs absent: [U]. Must export immutable original and pass byte/32-token gates before becoming a survivor. |
| AITER FA `flash_attn_varlen_func` (bf16), split-K variants | [U] | Ordinary GQA family, but metadata/layout and entry point differ. Wrapper alone proves no exact import eligibility; dispatch and adapter [U]. |
| MLA / sparse MLA / distributed-context / prefix-cascade changes | [U] | Excluded for this ordinary GQA single-request causal contract; no MLA latent, sparsity, distributed context or shared-prefix test is present. |
| CUDA-only FlashAttention/FlashInfer/CUTLASS, XPU, Navi WMMA | [U] | Excluded: wrong hardware target for gfx942 CDNA; platform/rocm.py selects native/AITER/Triton families. |
| FP8 attention/KV/out, TurboQuant/INT4, sinks, ALiBi, sliding window | [U] | Excluded: changes precision or attention semantics of this bf16 workload. MFMAType::F16 is the existing vendor path despite bf16 storage; enabling its FP8 alternate is not legal. |

A knob is not an independently legal candidate until its dispatcher supports
this shape and its exact immutable-original import contract is staged. This
CPU cannot assert the remaining numerical/ABI legality. In particular,
**"no ready adapter" is not "no legal faster implementation exists"**.
`inventory.complete=false` prevents an incomplete list from closing B1.
The next pre-staging task must pin the full AITER source/export inventory and
stage any additional adapters **before** starting the paid session; absent
ones remain [U] in this session. Existing nine recipes stay unchanged.

### Registered for the session: X5, before S2

`x5_builds=['kept']` is the runnable inventory now; additional source-pinned
survivors use separate sealed artifacts, never mutate the nine directories.
Exact native command for each staged survivor:

```sh
.venv/bin/python -m examples.vllm_llm.import_certificate \
  --run-build BUILD --model MODEL -- \
  --check 48 --check-pos 1024 --price-attention
```

`host.hip` prices only attn/reduce after that launch's byte check: 10 warmup
launches of each original/import, then three alternating original/import
batches of 100 isolated replays per role per layer, HIP events. Watchdog is
5 s per batch; runner bounds **30 s per survivor**, **120 s total** within
B's existing 1320 s. That is **0.5 min/$0.0166 each [E]**, at most
2 min/$0.0663 for the registered probe slot. No downloads/container engine
startup. Fresh 0/1040 checks and 32/32 CHECK-DSL certification precede pricing;
new adapter correctness work must also fit B, or remain NOT-COMPLETE.
This replaces the speculative 20-minute vendor sweep; it adds no B minutes.

`x5.prices` requires all **288 records** (48 × 2 × 3), exact position and
100 replays, finite positive durations and bytes_equal. Average of each
layer's sum of role medians must reach **<=7.69 us/layer**. A passing isolated
price gives `R2=max(0,48*(10.645-price)/1000)` [E, cross-basis], never a
certified fused wall gain. Whole-grid replay includes import-wrapper work,
whereas 10.645 is critical bodies. An eventual model-level pair/certificate
must validate transfer. Incomplete population fails closed; warm reuse is
reported as a limitation, not a cold-HBM price.

Runner now measures X5 **after S1 and before S2**, so R2 can influence
`G+R2+H >=0.324`; C1/C5 body pairs remain after S2. If all legal candidates
have been certified/enumerated and none reaches the bar, **B1=0, R2=0**,
then re-evaluate S2. Today's missing dispatch/adapters mean **B1 [U]**, with
zero R2 for spending decisions, not an exhaustive CLOSED verdict. Failed
byte/token gates disqualify a candidate; missing coverage is NOT-COMPLETE.
