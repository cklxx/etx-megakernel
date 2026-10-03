# ETX roadmap (written 2026-09-30)

This roadmap states the current position, next steps, and work rules.
Its numbers come from docs/MODEL-vllm-kernels.md, sections 10-18, docs/DATA.md, and docs/SURVEY-2026-09-29.md.

## Where we are

Qwen3-30B-A3B bf16, one MI300X, batch 1, greedy, 32/32 tokens against HF:

| | ms per token | of the 5.3 TB/s peak |
|---|---|---|
| vLLM (default<br> aiter on is 4.92) | 4.67 | 25% |
| ETX measured best (qknr, unfused / fused) | 5.62 / 5.86 | 21% |
| ETX build 3 + touch, predicted (sim x 1.11) | 3.56 | 33% |
| streaming floor at 3.56 / 4.3 TB/s | 1.71 / 1.41 | |

The whole gap is the per-task cost: 1.4-2.9 us, close to a kernel boundary of 1.61 us.
Several hundred tasks per token produce about 2 ms of this cost.
No published batch-1 result on gfx942 exceeds 36%, from the closed Kog system.

The correct mechanisms already include XCD-affine placement from XCC_ID and two-level event scope, with one flush from the last worker per XCD.
They also include AOT static per-worker queues and a simulator calibrated to 11%.
Lean and TLA+ cover the GEMV loop and event protocols.

## Near term: the queued session (examples/sessions/coh.py)

The dsl_gemv session ran on 2026-09-30. See docs/RESULTS-2026-09-30.md.
Build 3 measured 5.895 against a prediction of 3.56. Fold + touch saved 0.7%.
The diagnosis replaced the model with a phase floor, acquire cache invalidation, and row layout.
The next session is queued, with one run per number:

1. Microbenchmarks first: hw_bench restart (the inv's cost to a warm phase), cohload, the fixed width sweep,
   lfx. These are the diagnosis whatever the build does.
2. build 4 = build 3 + ETX_VL_COH. Readings (section 20): ~5.9 means the inv was never the cost; 5.6-5.8 means
   the overhead moved but touch is still dead; 4.9-5.3 means both landed. Independent of the number: fused <
   unfused (5.660) flips the 0.235 ms fusion penalty, the reading that matters.

## Mid term (1-2 months): kill the fusion penalty, then batching

1. Coherence is built and queued in examples/sessions/coh.py, section 20.
   The acquire's `buffer_inv sc1` made each phase start cold and removed the touch benefit.
   ETX_VL_COH uses coherent activation access in DSL tiles without the invalidate.
   TLA checks and the container build pass, with spill 9.
   The pre-registered range is 4.9-5.8 against build 3 at 5.895.
   Fused time below unfused time, 5.660, removes the 0.235 ms fusion penalty and is the decisive result.
   The Kog sentinel handoff was SLOWER than our events in sdat on 2026-09-30. This option is closed.
2. Phase width. The model assigns 0.56 ms/token to router's 0.5 MB across 304 tasks and predicts 2.22 ms of savings from appropriate widths.
   This extrapolates a 296-workgroup rate table to 88 workgroups from one narrow data point.
   The queued hw_bench stream width sweep must decide this before implementation.
3. Prefetch to the experts. The expert ids are known when the router's event fires; the down projection's weights
   can stream while the up projection runs. Experts are 60% of the bytes per token; larger than the dense touch,
   and only worth building once COH proves the touched lines survive.
4. Batching B <= 8 with dynamic expert scheduling (the union of the requests' experts per layer). Target about 800 tok/s
   at B = 8 (analytic: 779) against vLLM's 560-600. Batch 1 can win by 20-30%; batching is where the gap opens.
5. Generality. Build 3 combines manually selected ETX_VL_* switches.
   For any model, the compiler must derive the GEMV stream tiles, norm / top-k folds, kv-head placement, and autoconf configuration.
   Llama dense and DeepSeek-V2-Lite must require no code changes.
   This distinguishes a DSL from a model port.
6. Simulator. Fit the tax from its four parts (fence, atomic, descriptor fetch, wave start; no public gfx942 breakdown
   exists), bring the x1.11 under 5%, and gate every GPU run on a predicted gain.

## Long term (3-6 months)

- FP8 weights: half the bytes, floor about 0.8 ms. The tax then dominates even more, which is why it comes first.
- gfx950 / MI355X port (Fleet's ground; a direct comparison).
- Multi-GPU: TP over 8 MI300X with in-kernel collectives (Kog's KCCL under 3 us against RCCL's 8 us).
- Serving: ETX as a decode backend behind vLLM (prefill stays in vLLM). Without this the numbers stay a benchmark.

## How the work is done

- Research before design. The 2026-09-29 survey should have preceded the tax work by weeks; Kog and Fleet had the
  answers.
- A GPU session is one best configuration, one reading table, and the microbenchmark that decides the next step.
  Nothing is queued until all three exist.
- No experiment whose result is already expected. A run that yields no new knowledge is cut.
- Knowledge goes into docs/ (MODEL, SURVEY, this file), not only into results_*.
- Correctness gates never move: 32/32 tokens against HF greedy; --check / --check-dsl within 2 bf16 ulps.

## Risks

- Hardware: Hot Aisle had no 1x or 2x MI300X for 22 hours on 2026-09-29/30. A second provider is a backup worth setting up.
- The last millisecond at batch 1 may be hard: the best NVIDIA systems keep 15-25% of a pass in sync and setup.
- The baseline moves: aiter / FlyDSL are AMD's active MoE path on gfx942.
