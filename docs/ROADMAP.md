# ETX roadmap (written 2026-09-30)

Where we are, what comes next, and the rules for how the work is done. Numbers are from docs/MODEL-vllm-kernels.md
(sections 10-18), docs/DATA.md and docs/SURVEY-2026-09-29.md.

## Where we are

Qwen3-30B-A3B bf16, one MI300X, batch 1, greedy, 32/32 tokens against HF:

| | ms per token | of the 5.3 TB/s peak |
|---|---|---|
| vLLM (default; aiter on is 4.92) | 4.67 | 25% |
| ETX measured best (qknr, unfused / fused) | 5.62 / 5.86 | 21% |
| ETX build 3 + touch, predicted (sim x 1.11) | 3.56 | 33% |
| streaming floor at 3.56 / 4.3 TB/s | 1.71 / 1.41 | |

The whole gap is the per-task tax: 1.4-2.9 us per task, about a kernel boundary (1.61 us), over several hundred tasks
per token, about 2 ms. Nobody has published more than 36% at batch 1 on gfx942 (Kog, closed). What is already right:
XCD-affine placement by reading XCC_ID, two-level event scope (the last worker per XCD flushes), AOT static per-worker
queues, a simulator calibrated to 11%, Lean and TLA+ on the GEMV loop and the event protocols.

## Near term: the queued session (examples/sessions/dsl_gemv.py)

One run each, best configuration first, decisions pre-registered:

1. build 3 + touch: read the number against the sensitivity table (section 17). About 3.6: the model holds. About
   4.5: the prologue loads see loaded latency. About 4.8: the hand-offs are what evx measured, not the trace. Two
   requests: 2.42 ms per token predicted (413 tok/s; vLLM batch 2: 360).
2. hw_bench sdat against evx: the value of the sentinel hand-off before it is built.
3. hw_bench lfx: one XCD's bandwidth; closes the pipelining question (section 17 says batching wins at every f).

## Mid term (1-2 months): kill the fusion penalty, then batching

1. Coherence (built, queued: examples/sessions/coh.py, section 20). The acquire's `buffer_inv sc1` made every
   phase start cold and nullified the touch; ETX_VL_COH gives DSL tiles coherent activation access and no inv.
   TLA-checked, container-built (spill 9). Pre-registered: 4.9-5.8 against build 3's 5.895; fused < unfused
   (5.660) is the reading that matters -- the 0.235 ms fusion penalty is the number this exists to kill.
   The sentinel hand-off (Kog) measured SLOWER than our events (sdat, 2026-09-30) and is dead.
2. Phase width. The model says spreading router's 0.5 MB over 304 tasks costs 0.56 ms/token and the right widths
   save 2.22 ms -- but that extrapolates a 296-workgroup bandwidth table down to 88 workgroups on one narrow data
   point. The fixed width sweep (hw_bench stream, in the queued session) decides it before any implementation.
3. Prefetch to the experts. The expert ids are known when the router's event fires; the down projection's weights
   can stream while the up projection runs. Experts are 60% of the bytes per token; larger than the dense touch,
   and only worth building once COH proves the touched lines survive.
4. Batching B <= 8 with dynamic expert scheduling (the union of the requests' experts per layer). Target about 800 tok/s
   at B = 8 (analytic: 779) against vLLM's 560-600. Batch 1 can win by 20-30%; batching is where the gap opens.
5. Generality. Build 3 is a set of ETX_VL_* switches assembled by hand. The compiler must derive the streaming GEMV
   tiles, the norm / top-k folds, the placement by kv head and the configuration (autoconf) for any model: Llama
   dense and DeepSeek-V2-Lite without code changes. This is what makes ETX a DSL rather than a port.
6. Simulator. Fit the tax from its four parts (fence, atomic, descriptor fetch, wave start; no public gfx942 breakdown
   exists), bring the x1.11 under 5%, and gate every GPU run on a predicted gain.

## Long term (3-6 months)

- FP8 weights: half the bytes, floor about 0.8 ms; the tax then dominates even more, which is why it comes first.
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
