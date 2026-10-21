# Qwen o_proj producer scoring: implementation preparation

## Decision and scope

This work implements the original producer-scoring hypothesis.
The candidate is an experiment. The default switch is off.
No GPU was rented. No remote command was run. GPU spend is $0.
The [final acceptance record](../../examples/vllm_llm/results/2026-10-15-producer-prep/validation.md) is complete.
The full suite passed 2,052 tests. It skipped 17 tests for recorded environment or data reasons.
All 16 matrix builds and both compiler paths passed.
No real-model token pass or speed gain is claimed.

The target is Qwen3-30B-A3B, bf16, batch 1, context 1024.
The model must produce the same 32 greedy tokens as the retained HF golden.
The hardware target is one MI300X with eight XCDs.
The candidate keeps the existing o_proj GEMV primitive and its row ownership.
Each output owner scores its completed residual slice before the global RMS scale.
The last producer reduces the scores, applies the scale, and calls the kept top-k.
This is not the separate P5 last-router-arriver proposal.
KEY32 is reused. It is not the new mechanism.

The [prior study](QWEN-ROUTER-FUSION-STUDY-2026-10-15.md) remains valid.
Deferring the scale changes bf16 rounding. It is not an exact rewrite.
That result does not prove a real-model failure or a performance loss.
This implementation keeps the sparse counterexample and all acceptance gates.

## Sources and emitted path

- [Producer implementation](../../etx/ops/producer.py)
- [Shared HIP and CPU arithmetic](../../etx/ops/producer_math.h)
- [Plan integration](../../examples/vllm_llm/model.py)
- [Host argument setup and actual-input capture](../../examples/vllm_llm/host.hip)
- [CPU numerical and protocol replay](../../examples/vllm_llm/producer_replay.py)
- [GPU session operations and report](../../examples/vllm_llm/producer_session.py)
- [Same-build trace attribution](../../examples/vllm_llm/producer_trace.py)
- [Local acceptance driver](../../examples/vllm_llm/producer_local.py)
- [Session job graph](../../examples/sessions/qwen_producer.py)
- [Tests](../../tests/test_producer.py)

`ETX_VL_PRODUCER_SCORE=1` selects the new export `dsl_gemv_producer`.
The baseline uses the kept narrow router and the kept KEY32 selector.
The flag is absent from the kept recipe.
A subprocess test compares default-off tile source and import layouts with `c432c77`.
They must be byte-identical.

The candidate plan has 48 o_proj grids with 256 owners each.
Each grid must name `etx_tile_dsl_gemv_producer` in `plan.json`.
The plan must have no separate router or top-k grid.
The source must contain the shared scoring function and the existing top-k call.
The plan audit checks these facts. An arm name alone is not proof.

The producer argument block is 192 bytes.
The first 160 bytes keep the DSL GEMV layout, including the grid at byte 144.
The trailing pointers are router weights at byte 160, normalized input at 168,
and token-expert indices at 176.
The host rejects incompatible shapes before launch.
The planner rejects other model shapes, batch modes, UC, L1 mode, and local-fold mode.
The prepared arm uses the kept environment and changes only the producer switch.

## Data, ownership, visibility and reuse

Owner `p` computes o_proj rows `[8p, 8p+8)` with the baseline GEMV.
The GEMV rounds each output to bf16 at the same point as the baseline.
The owner then computes `z[h] = bf16(o[h] + residual[h])`.
It writes that slice to the canonical `resid2` buffer.
The owner preserves the learned norm weight `gamma[h]`.

For each expert, the owner computes eight ordered fp32 FMAs:
`partial[p,e] = sum_j W[e,8p+j] * (z[8p+j] * gamma[8p+j])`.
It also computes eight ordered square FMAs for its norm contribution.
The shared header fixes these arithmetic steps for HIP and CPU replay.
The candidate does not reduce the router's 262,144 matrix FMAs per layer.
The implementation also performs gamma multiplies in the scoring lanes.
Their cost is part of the unknown producer cost.

One scratch buffer serves all layers in sequence.
It has 33,344 four-byte words, or 133,376 bytes.
It holds 256 records of 128 scores plus one square sum.
It also holds 256 epoch tags, one arrival counter, one epoch, and padding.
The generic host allocates and clears this buffer once.
The buffer is not a new cached cross-XCD tensor.
Its payload uses the DSL's system-scope coherent accesses.

Each payload writer executes a system fence.
All waves join before the leader increments the counter.
The increment is a system-scope acquire-release atomic operation.
The last leader acquires the release chain and shares that result through a block join.
The last producer checks all epoch tags.
It folds score records in ascending owner order, regardless of arrival order.
There is no producer spin loop.
A producer that is not last returns and lets its worker retire.

The norm uses the baseline ownership and reduction order.
There are 256 active square sums, a wave butterfly, and a fixed wave-order sum.
The last producer applies `rsqrt(ss/2048 + 1e-6)`.
It writes the normalized expert input with the baseline multiply order and bf16 round.
It rounds each scaled router score to bf16.
It calls `dsl_topk_lds`, including the existing KEY32 path.
It publishes all eight copies of IDs, fp32 mixing weights, and token-expert indices.

The folded expert tiles read one canonical `ox` vector.
The other seven allocated `ox` slabs are not inputs to this path.
The candidate writes the canonical vector and the capture gate checks that vector.
Writing eight normalized copies would add unused stores and create a false byte gate.
This fact comes from the host's `dsl_moe_up_fold` argument setup.

The finalizer fences all output writers before it resets the counter.
It advances the epoch and sets the count to zero before returning.
The normal ETX completion event includes this finalizer and every o_proj owner.
The expert grids wait on that event.
The next layer cannot reuse scratch until the current expert work completes.
The next decode cannot reuse it until the current invocation retires.
Epoch wrap traps and requires a new process.
A timeout or trap aborts the process. The session never reuses that process's state.
A retry creates new buffers and a new evidence directory.

The CPU protocol model checks the logical happens-before conditions.
It is not an AMD cache model. Hardware must still test visibility and progress.

## Executed CPU simulation and evidence reuse

The CPU replay compiles and calls `producer_math.h` through a native shared library.
It does not replace the production scoring loop with a Python approximation.
The baseline norm, router dot and KEY32 models come from the prior study.
The CPU uses NumPy for rsqrt and exp. These are not GPU instruction-bit certificates.

The fixed-seed simulation runs 32 synthetic input and weight cases.
It also reuses the retained sparse near-tie witness.
Each case runs three random completion orders and scratch reuse cycles.
The fixed owner fold gives the same result after each completion order.
Failure tests cover missing release, missing acquire, duplicate owners, stale epochs,
missing normalized input, missing fanout, and epoch wrap.
The tests also cover capture corruption, incomplete records, deadlines, and failed gates.

The simulation uses synthetic weights and synthetic inputs.
It does not use real Qwen activations.
The prior inventory found no retained router matrix, gamma, o_proj output and residual set.
The scoped local search found no such complete set either.
Retained routing outputs cannot reconstruct those inputs.
Actual-model replay is pending. No synthetic tensor is labelled as a capture.

The host capture runs only inside the sequential `--check` path.
For each layer it saves o_proj output, residual input, gamma, router matrix,
canonical normalized input, residual output, bf16 logits, IDs, mixing weights and indices.
Files are created exclusively. A retry cannot overwrite them.
The session seals each complete capture with hashes, build identity and position.
The reader rejects missing files, extra files, wrong sizes, changed bytes and wrong provenance.
Tests execute this format with explicitly labelled synthetic fixtures.
The session requests actual captures at positions 0 and 1040 for both executables.
It saves CPU diagnostic replay before the production byte gate.
A CPU rsqrt/exp diagnostic mismatch is reported separately from GPU acceptance.

## Ceiling and gain model

The router matrix has 524,288 bytes per layer.
At the project's assumed achievable 4.3 TB/s, its bandwidth floor is 0.122 us.
The o_proj matrix has 16,777,216 bytes. Its corresponding floor is 3.902 us.
These are bandwidth estimates. They omit dependencies, cache effects and instructions.
The producer does not remove either matrix.

The score and square records add 264,192 bytes of publication plus readback per layer.
Tags, the counter, residual publication and routing fanout add more traffic.
The canonical normalized expert input still has 4,096 bytes.
The fixed score reduction has 128 times 256 additions, including each initial zero add.
Two waves fold all owner records in fixed order.
This exposed tail can erase the saved scheduling gaps.
A loss would apply to this implementation. It would not close every possible producer-scoring layout.
No byte floor is used as a latency prediction for coherent scalar accesses.

The current retained kept wall median is 4.051 ms in the October 14 session.
There is no matching retained current task trace for that timing build.
The historical October 11 handoffs, including the old 209.8 us total, are not a predicted gain.
The study's historical trace and counterexample are reused without rerunning their questions.
Fleet's lost last-arriver publication cases remain relevant negative evidence.
They do not measure this Qwen producer candidate.

The schedule model uses matching endpoints:
o_proj start to all required normalized input and routing outputs ready.
It includes producer scoring, producer interference, exposed last-owner scoring,
publication, readback and reduction, scale, normalized input, top-k, and fanout.
The implemented finalizer runs the tail in sequence. The model does not assume a parallel norm tail.
The illustrative CPU schedule finishes at 22.1 us under its stated assumed costs.
That value is not a GPU prediction.
The current net gain is unknown.

After correctness, the session collects uninstrumented wall pairs first.
It then collects a same-build trace for each arm if the budget permits.
The raw trace columns are type, four coordinates, taken (5), ready (6),
body done (7), and worker. Body done is stamped before phase-three release and event arrival.
The trace report gives o_proj ready/start to routing body completion and to all expert tasks ready.
A task can be taken before the producer finishes and become ready afterward.
The body-done-to-consumer-ready tail includes unresolved publication and scheduling work;
it is not an isolated publication cost. No exact output-publication stamp exists.
All endpoints use one trace clock. These diagnostics do not change the untraced KEEP basis.
The report does not subtract device stage intervals from host wall medians.
Scoring, interference, counter traffic, scale and top-k are not stamped separately.
Those individual costs stay unknown. Trace overhead is not calibrated.

## Registered GPU rules

The incremental rule is unchanged:
five alternating A/B wall pairs; every saving is positive;
median saving is at least 0.030 ms; median saving exceeds twice the range.
The deciding field is host `RESULT wall_ms`.
The earlier G6 device basis is not substituted for this producer wall experiment.
The prior producer study already registered this wall basis and threshold.

The session orders each round as baseline fused, candidate fused, candidate graph.
It reverses that order on alternate rounds.
Thus both the incremental pair and the candidate fusion pair are adjacent.
It runs S3 on the same GPU in each round with the existing vLLM wall estimator.
S3 uses the pinned runtime image and `ROCM_AITER_UNIFIED_ATTN`.
It is the strongest legal configuration in the prior tested set.
The report does not call this a new exhaustive vLLM search.

Candidate versus its same-kernel HIP graph has a separate fusion verdict.
Every relative wall gain must be positive.
The median must exceed 2% and twice the gain range.
An incremental KEEP does not imply ESTABLISHED fusion.
The report also gives the five candidate-minus-S3 wall deltas.
The system LEAD rule uses the existing 0.050 ms and twice-spread bar.

Before any measured pair, both executables need fresh full import certificates at 0 and 1040.
Both need 32/32 HF greedy tokens in fused and HIP-graph modes.
Both need the existing CHECK-DSL gate in fused mode.
The selected IDs, mixing weights, indices, residuals and normalized expert input must match byte for byte.
The input matrices and source activations must also match for attribution.
Raw router-logit bits are diagnostic. The existing CHECK-DSL numerical gate still applies.
The session does not add an exact gate on all 128 logits.
An actual-model mismatch blocks timing. A missing or failed gate also blocks timing.
All timed invocations must again return 32/32 tokens.
An incomplete sample population has no performance verdict.
The final report revalidates raw logs, build stamps, certificates and capture hashes.
The incremental rule uses integer nanoseconds from the original decimal log text.
Both the packed inputs and the HF reference files are hashed before and after timing.
It recomputes the S3 estimator and both comparison rules.

## Limits and launch

See the [short runbook](QWEN-PRODUCER-RUNBOOK.md).
The source must pass the complete local driver before a GPU start.
The driver checks the full repository suite, all 16 builds, both ROCm arms,
LLVM 23 device IR and assembly, host object compilation, simulation and shell syntax.
For the native compiler, it maps container-only include paths to the same mounted source files.
It preserves the original source and records each include mapping.
The final acceptance binds the checked source hashes.
A changed source invalidates local readiness.

Compilation is not executable linking, loading, or hardware execution.
macOS cannot link the Linux HIP runtime in this environment.
The prepared VM build must link the real executables before capture.
No imported-launch, actual-model, memory visibility, token or speed result is inherited.
The GPU session can fail the actual byte gate even if all local checks pass.
This is the experiment's intended decision.
