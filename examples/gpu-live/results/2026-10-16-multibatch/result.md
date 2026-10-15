# Multi-request continuation result

COMPLETE ATTRIBUTION; SHARED-WEIGHT CANDIDATE NOT IMPLEMENTED / NOT MEASURED.

The prepared code has no DeepSeek shared-weight/column-batched tile (`docs/BATCH-2026-10-14.md`, Stage B2 is designed, not built). A new tile with arithmetic/LDS validation and paired full-model acceptance was not safely achievable in the remaining authorized window. The explicitly authorized fallback filled the missing B2/B4 task attribution, without retrying the crashing profiler or rerunning settled unprofiled pairs.

## Actual performance baselines, reused evidence

One MI300X VF, gfx942, ROCm7.14; DeepSeek-Coder-V2-Lite-Base, bf16, context1024,32decode tokens. Five same-box interleaved paired rounds from the earlier completed B4 timing subjobs. AITER is strongest only among the two tested legal configurations (default/AITER), selected separately at each B.

ETX step is the registered direct host bracket; vLLM step is the registered host full-minus-short estimator. Total throughput uses those step times. Paired gain is vLLM/ETX−1; spread is max−min of the five paired percentage gains.

| B | ETX median step ms | AITER median step ms | ETX total tok/s | AITER total tok/s | Paired throughput gain % | Spread pp | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
|1|3.222786|4.097652|310.290537|244.042198|+27.110935|1.043288|WIN|
|2|4.532310|4.732610|441.276082|422.599792|+4.419380|0.527359|WIN|
|4|8.921324|6.165512|448.363942|648.770126|−30.930675|9.314794|LOSS|

Median worst-request token interval: ETX B1/B2/B4=3.23944/4.57104/9.00740ms; vLLM advances every sequence per step, so its request interval equals its step. Paired request-latency gain/spread: B1 +26.435836%/1.005319pp, B2 +3.456906%/0.692543pp, B4 −31.568217%/9.336125pp. Exact five samples and individual request records are retained in baseline/batch_report.json and baseline/brounds.json. Every selected full/short vLLM timing-call gate and all ETX per-request gates passed in these prior timing subjobs.

These are reused measurements, not fresh continuation timings. The earlier sealed parent record is incomplete because its subsequent traffic-attribution job crashed; that does not relabel its completed timing subjobs.

## New deciding measurements

One new serialized GPU0 attempt, three rounds in B1/B2/B4, B4/B2/B1, B1/B2/B4 order. All nine invocations passed per-request32/32greedy tokens,32/32warm-up tokens and27/27hidden boundaries. All raw task rows passed exact coverage/type/coordinate/timestamp/static-worker validation before timeline analysis. No missing tasks or timestamp clamping accepted.

The trace covers the final decode token of each invocation, with ETX_TRACE instrumentation. It is not an unprofiled wall-performance comparison. B1 uses D4; B2/B4 use the same replicated recipe, making B2→B4 the cleaner scaling comparison.

| B | Complete tasks | Median traced span ms | Min–max ms |
|---|---:|---:|---:|
|1|53022|3.094430|3.093460–3.100780|
|2|109892|4.339200|4.324760–4.347170|
|4|219784|8.670160|8.664580–8.696920|

B4/B2: task count2.000000×; summed task-body work1.982145×; traced span1.998101×; CP-body contribution1.969555×. Thus B4 replicates twice the task work and obtains almost no extra throughput over B2 in these traces. This observation does not identify HBM saturation as the cause.

The B4 median-span run (round2) decomposes exactly into7.340840ms CP bodies +0.963120ms dependency gaps +0.366200ms queue gaps =8.670160ms. Across all three B4 runs, independently summarized medians are7.356920ms bodies,0.955080ms dependency gaps,0.366200ms queue gaps; their medians need not sum to the median span. Bodies include internal tile waits. Queue-induced serialization also appears as bodies of queue predecessors, so the queue-gap column is not all scheduling cost.

B4 median CP-body attribution by family:

| Family | CP body ms |
|---|---:|
|gate_up (routed/shared combined)|2.021840|
|down (routed/shared combined)|1.174800|
|attention|0.973080|
|qkv|0.926280|
|router/norm|0.621600|
|output projection|0.513360|
|merge/UV|0.439960|
|qabs/UK|0.282880|
|lm_head|0.225200|

All families, ranges, body-work sums and per-request phases are in summary.json/trace_report.json. Static source (`graph.py`, `_Req`) maps r to one of two worker slots using r%2; at B4 r0/r2 and r1/r3 share queues. The task counts and recorded critical paths are consistent with this repeated-work/serialization structure. No counterfactual scheduling variant was measured, so this is not proof that queue placement alone causes the loss.

## Decision

The registered B4 target is5.5489608ms (0.9×the retained6.165512ms vLLM median). From8.921324ms ETX, this requires3.3723632ms,37.801151%, improvement. The complete observed CP gaps are only1.307660–1.336720ms. Even ideal removal of all observed gaps is much smaller than the required host-step reduction. This is an instrumented scale comparison under fixed task durations/queue edges, not a host-wall guarantee; schedule changes can alter those conditions.

Prioritize reducing repeated tile-body work rather than a gap-only cleanup. No single observed family has enough CP body to supply the entire target reduction even under an ideal deletion model. QKV is a bounded first column-batched correctness/microbenchmark candidate (0.926280ms CP body to address), but cannot reach the full B4 target alone. Shared-expert and other request-independent projections must also be considered; gate_up/down combine shared and routed work and need finer separation before assuming their full cost is reusable.

No actual repeated traffic reduction was measured. Partial prior rocprof CSV remains rejected; no profiler command was retried. HF distinct-expert means6.0/11.496394/20.631010 and the5.130547ms column-batching projection are conditional prior estimates, not native ETX pick equality or measured implementation speedup.

## Source, acceptance and evidence

Executed driver/session commit: edd7457a333fd25dcf90402c91b8af24cb563cfe, isolated branch task/multibatch-continuation-20261016.
Accepted frozen executable/source commit:554133d744b4591c9cf87c68b6817df781137c9b; archive abee09fcda357bb982cfba4d85bb387a6da97ec2385d3d4574f1f7cd4282d5da. Binary/plan/generatedHIP/resources identities match all prior gates before and after the nine runs. Per-request producer receipts/payloads and pinned inputs were verified. Fleet remained read-only. Deployed driver hashes also match after execution. No executable edits/rebuilds, broad suite or unchanged compiler matrix reruns.

Local acceptance:9trace-validator/aggregation tests passed,3summary tests passed; retained real B1 trace dry-analysis passed; session declaration and git diff checks passed. Prior2190-pass and compiler preparation remain reused evidence, not rerun here.

New remote record: /root/sess/multibatch-continuation-20261015-edd7457/20261015T164723Z-d0e0cb.
Full local export: /LOCAL_WORKSPACE/.hermes/cache/scratch/etx-multibatch-continuation-20261016-evidence/raw/20261015T164723Z-d0e0cb.
Stock record verify exit0:32files match,0bad,0omitted; complete=True. MANIFEST <HOST_KEY_FINGERPRINT>. Full raw logs and all nine binary traces retained. Incremental exports completed before the original17:22:10Z stop. Curated repository receipt is not represented as a complete raw record.

Runner exit0, job exit0; started16:47:23Z, sealed16:48:59Z,96s execution envelope. At16:50:02Z GPU use0%, no continuation runner/driver/model processes remained; no GPU job will be restarted. No new machine, cloud action, cron loop, publish or push. Amber owns teardown; machine deletion is NOT CLAIMED. Incremental allocation cost was not queried;96s at the earlier illustrative$1.99/h would be about$0.053, an estimate excluding idle/setup/allocation time.

Limits retained: same-kernel graph baseline UNAVAILABLE because of LDS lifetime; no full fusion certification. Prior x8 and matching single-request hidden-state failures remain failures; these B1/B2/B4 results do not extend to B8. No accepted hardware traffic counters, native multi-request pick-union equality, candidate arithmetic/LDS/correctness evidence or paired candidate measurements exist. Those are the next GPU prerequisites after a reviewed local implementation. Original absolute stop and export reserve were preserved.
