# Three prepared Qwen experiments, 2026-10-15

Preparation only. No GPU experiment was executed by this integration; GPU spend is $0.
All three paths remain OFF in the normal model. This is not a combined mechanism,
a fourth experiment, a performance result or authorization to spend.

Source parents: producer main `bcc3c98b421164ae70e42d566bf836d0da35425f`,
attention `db1694bcdbd2345401f42a58e5f41b4b4ee43d02`, expert repair
`11743ed0c18ce4a2acc4af5493c678af1a62691a` (includes `172a22f`).
The integration commit, when review permits it, records those parents; source maps bind executable inputs.

| Prepared path | Mechanism and decision | Actual first GPU gate | Registration |
| --- | --- | --- | --- |
| Producer score | o_proj owners compute router contributions; replaces router/top-k task boundaries. Publication cost and net gain are unknown. | Full-model incremental host RESULT wall: five AB/BA pairs, every gain positive, median >=0.030 ms and >2 times range. Separate fusion rule: every relative gain positive, median >2% and >2 times range. | [Producer preparation](QWEN-PRODUCER-PREP-2026-10-15.md), [runbook](QWEN-PRODUCER-RUNBOOK.md), [readiness repairs](QWEN-PRODUCER-READINESS-2026-10-15.md) |
| C6 attention | Prep owner publishes early; last drained partition runs original ordered reduce. Fixed-body local advance 101.82 us is an estimate; registered gross allowance 96 us, combined target 50 us. | Local resident protocol screen, all 48 layers at each position 0/1023/1024/1040: five interleaved paired span sums, every gain positive, median >=50 us and >2 times range at ALL four positions. General 30 us KEEP is reported separately and does not pass 50 us. | [C6 registration](../../examples/vllm_llm/attention_early_merge/EXPERIMENT.md) |
| Expert merge | Early expert contributions, elected ordered fold, scalar/normalized input publication. Qkv and head roles are separate isolated screens. | Five complete interleaved DEVICE pairs for one input: every gain positive, median >=0.625 us and >2 times range. Synthetic pass only permits real-input screening. Later full-model KEEP requires >=0.030 ms HOST wall plus all-positive/noise and project baselines. | [Expert registration](EXPERIMENT-qwen-expert-merge-2026-10-15.md) |

These gates have different scopes and units. Passing local spans or an isolated
synthetic screen establishes neither model wall gain nor hardware correctness of
another path. No thresholds are combined or relaxed.

Producer correctness precedes timing: fresh exhaustive same-build original/import
checks at positions 0/1040, CHECK-DSL, 32/32 HF greedy for fused and same-kernel
HIP graph in both builds, exact IDs/mix/indices/residual/normalized input. Raw
logits are diagnostic. HF must resolve to packed-model/hf; checkpoint, golden
and prompt hashes bind S3 to the same inputs. S3 is a same-session estimator,
with 0.050 ms lead and noise rule; no exhaustive vLLM search is added.

C6 requires original/import and CHECK-DSL, ordinary AND hybrid 32/32 model checks,
all 15 buffers exact for both arms/binaries across all layers/positions, repeated
epochs, skew and duplicate/stale/future/timeout faults. The native full-dispatch
resource gate is <=128 VGPR, <=29752 B LDS and >=1 static 1024-thread block/CU;
actual residency/placement is checked on device. Retained kept dispatcher is
compiled but never executed in the local probe. The unbound resource envelope
MUST NOT run. Its resident separate-task control is neither the full ETX
scheduler nor HIP graph. Reset/copies precede both windows. Dynamic spills,
queue interference, incremental reset and whole-step costs remain unknown.

Expert requires fused and same-kernel graph exact equality through 12 resets,
including buffers, partials, scalar, ownership and counters, before timing. Both
qkv/head diagnostic control paths include the reviewed all-thread LDS join;
timed paths do not gain that diagnostic join. Selected-slot packed layout is
not original model allocation. Real qkv inputs are layers 0/23/46 and head 47,
each at positions 0/1040. Full-model implementation/import/DSL/tokens/system
baselines remain later gates; the isolated screen does not enable a default.

The expert final re-review (`astra-expert-fixes-rereview.md`) closes both prior
defects at `11743ed`; all expert implementation/test/job bytes remain identical
in this integration, and all eight generated tiles and plan headers match that
reviewed preparation. Attention remains bound to `db1694b`; its final review
decision is recorded with the copied reports in the integration evidence.
A missing or blocking attention review prevents main integration and publication.

Integration evidence, review status, refreshed identities, compilation comparison
and raw actual-exit logs live in
`examples/vllm_llm/results/2026-10-15-three-integration/`.
Native executable HIP runtime linking, load, visibility, real captures and all GPU
performance remain open. See the [bounded GPU-only queue](QWEN-THREE-GPU-QUEUE-2026-10-15.md).
