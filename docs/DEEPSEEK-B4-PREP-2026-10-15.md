# DeepSeek B4 controlled attribution: preparation

Status: prepared, not run. CC review found B4 R1/R2 blockers; the local repairs at `40ddf36` passed acceptance. Independent scoped repair recheck CLOSED both findings with no blocking repair regression; serialized publication is authorized. This is the existing B4 follow-up in BATCH-2026-10-14.md section 12. It adds no kernel
candidate. It is separate from the x8 correctness session. No GPU use, rental or spending is authorized by this file.
The current provider is DigitalOcean. Its available one-GPU shape, tariff and billing minimum have not been checked.

## Question and retained evidence

Does the B4 slowdown occur with the repaired correctness and baseline protocol? If it does, which measured costs grow:
external read traffic, task bodies or event/queue waits? Exact-prefix HF routing supplies the expert-union input to the
existing column-batching estimate. It does not prove the expert choices made by ETX.

The historical ETX medians labelled as wall time were 3.649397, 4.960463 and 9.760234 ms at B1/B2/B4. The corresponding vLLM medians
were 4.852954, 5.652581 and 7.053838 ms. Those records lack the repaired short-call vLLM gates and producer receipts.
The old ETX BATCH step used `hst.last_ms`, which is a HIP-event device span on both fused and unfused paths.
Thus the historical application comparison also mixed ETX device time with vLLM's host full-minus-short estimator.
They are evidence for this question, not certified host-wall baselines. No historical absolute time is combined with a new
session. The B4 useful-payload floor is an estimate: 19.738 GB / 4.3 TB/s = 4.590 ms. That floor is below the historical
vLLM step, so bytes alone do not force a loss. Actual traffic and body/wait costs are unknown.

The retained Fleet STATUS.md and ETX model/experiment records already cover occupancy, prefetch, split workers,
visibility and local-fold studies. This session does not repeat those optimizations. It reuses FETCH_SIZE, ETX task
traces, the timeline latency report, repaired vLLM gates and exact-prefix routing tooling.

## Frozen executable protocol

The registered queue is `examples/sessions/ds_b4_attribution.py`. Run it only from a clean committed archive after
separate GPU authorization and native toolchain/runtime checks. The job has one GPU lane, pinned to GPU 0. The planned
duration is 60–90 minutes; this is an estimate. The runner stops the complete session at 7200 seconds, including setup.
This cap is independent of the x8 correctness cap. There is no quoted dollar budget.

1. Reuse the pinned environment, Fleet export and model/weight preparation. Verify producer receipts before reuse.
   Generate missing references in the fresh result directory. Preserve old cache data. Bind all four request tuples,
   the model snapshot and the producer execution. A reference failure stops the dependent model work.
2. Build the kept B1 arm `alld4r0l1kvsmw`, x2 and x4. Record source, plan, generated source, resource and executable
   identities. x2/x4 retain K1 and M1; B1 also has D4. Thus B1-to-B4 scaling includes the stated D4 difference.
3. Gate all three fused invocations. Each needs all requests to have 32/32 greedy tokens, 32/32 warm-up tokens
   and 27/27 consecutive hidden boundaries. A failed correctness or identity gate prevents timing and captures.
   No imported vLLM launch is used in this model. Check the pinned same-binary graph baseline locally before launch.
   All current arms are UNAVAILABLE: x2/x4 use `fleet_expert_down_cached`, and B1 uses `fleet_d4_routed_down`.
   These bodies read gate_up's LDS routing after a per-grid graph launch has ended that LDS lifetime.
   Unknown task symbols also fail closed. No known-invalid graph command is executed. This is a retained limitation:
   ETX_Technical_Design.md section 15.7 already records wrong layers even with routing recomputed. Do not retry an
   uncached substitution, waive a gate, or present an accidental pass as a replay-safety proof.
4. Select the faster legal vLLM configuration separately at B1/B2/B4 from default and AITER in the pinned image
   `vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7`.
   Every full and one-token subtraction call is checked, including warm-up. This is strongest only among the two
   tested configurations. Preserve the effective configuration and exact image identity from the existing jobs.
   In timed rounds each configuration invokes only the B values assigned to it by selection. The prompt/golden set
   covers those calls; returned batch keys must match the invoked subset exactly. Captures invoke only their selected B.
   Unselected, illegal configuration/B combinations are excluded; every invoked full/short gate remains mandatory.
5. Run five interleaved rounds of fused ETX and selected vLLM. Reverse the order on even rounds. Keep every
   independent correctness gate. A failed timed gate stops the remaining timing rounds and captures. The host now
   brackets both fused and unfused calls with the same steady host clock; HIP event durations are not used as wall
   time. The interval includes fused reset/residency/completion work. Host parameter copies precede the timer.
   Device per-request timestamps remain separate. Same-kernel graph pairs may run only after complete replay safety
   is established; this queue refuses them for every current arm. It records the blocked baseline beside the
   corrected vLLM comparison. Attribution can complete, but a performance claim requiring both baselines cannot.
6. After unprofiled rounds, run fresh gated attribution invocations. Enable ETX_TRACE on the same fused binaries and
   retain the complete last-token raw trace, each task's body/wait span and the timeline critical-path report. Trace
   timings include instrumentation. A body includes tile-internal waits; ETX event waits do not isolate those waits.
7. Capture FETCH_SIZE per dispatch for ETX and selected vLLM at B1/B2/B4. Retain the profiler version, complete available
   metric output, all counter CSV files and raw invocation logs. ETX must have exactly 64 megakernel dispatches
   (32 tokens, two repetitions). Report the raw derived FETCH_SIZE values in the target metric's KB units, dispatch
   durations and device spans. These are measured counter values, not shape-byte estimates or unprofiled bandwidth.
   They exclude host DMA and do not distinguish spill, repeated weight and activation traffic.
8. Mark vLLM warm-up/full/short calls with synchronized ROCTx ranges. Require the retained kernel-dispatch trace.
   Within every nonoverlapping call range, FETCH_SIZE and trace records must match exactly by Process_Id, Agent_Id,
   Dispatch_Id and start/end timestamps. Missing, extra, duplicate, misscoped or boundary-crossing records refuse
   attribution. Worker and marker processes may differ; trace/counter scope binds the dispatch identity.
   An explicit zero value is a measured record, including an all-zero short call. Missing records cannot become zeros.
   Retain full and short traffic separately; all five full-minus-short estimates must remain positive.
   `(full traffic - short traffic) / 31` is explicitly an estimate from measured calls: cancellation of prefill/cache
   effects is not independently established. No profiled time enters a performance verdict.
9. Run HF routing separately at B1, B2 and B4, using that run's first B bound prompts. Check its entire 33-token greedy trajectory against the receipts and
   retain all top-k picks for all 26 MoE layers and 32 decode steps at each B. Compute each union from that B's own
   HF run over requests 0 through B-1, not B4's first-B picks or an average over subsets. Identical tokens do not
   imply identical routing under different numerical batch shapes. The ETX input window is golden[0:32], output window golden[1:33]; the vLLM output
   window is golden[0:32]. Record that distinction. Missing or divergent trajectories leave the projection incomplete.
10. Write `b4_result.json` after normal job completion, including failed or skipped dependencies. A hard session
    deadline can interrupt the finalizer; `status.json` and partial files then show the incomplete run. Retain partial round and attribution JSON,
    raw logs and runner exit codes. A failed job cannot become a completed experiment through file hashes alone.

## Acceptance and decision branches

The correctness conditions above are mandatory for each measurement. The existing throughput and latency rule remains:
five same-session pairs, every gain of one sign, median magnitude at least 2%, and magnitude greater than twice the
spread. Report WIN, LOSS or TIE independently for throughput and request latency. The same-kernel graph baseline is pinned to the same binary/plan but locally refused for the LDS handoff.
Its absence keeps `performance_claim_status` and the overall result INCOMPLETE, even if attribution completes.
No DeepSeek fusion verdict is available, and no new Qwen/vLLM import claim is made.

- If B4 does not reproduce a certified loss, report the new outcome. Do not transfer the old loss's cause to this run.
- If any counter, marker, complete task trace or bound routing prefix is missing, report attribution INCOMPLETE even
  if a separately certified performance comparison completed. Do not substitute the byte model for a counter.
- If B4 still loses, report measured traffic growth and changes in body, event wait and critical-path spans separately.
  A longer body alone does not identify HBM saturation, load issue or tile-internal synchronization. These are possible
  explanations until additional evidence separates them. This queue does not authorize a new kernel experiment.
- Compute the existing column-batching projection only from the exact-prefix union and this session's certified B1
  and vLLM data. It remains a conditional HF-based estimate; native ETX pick equality is unmeasured. Its existing trigger is projected ETX step <= 0.9 times vLLM's step. A crossed
  trigger justifies a later design decision; it is not evidence that column-batched tiles were built or are faster.

## Remaining gates

Local CPU fixtures exercise complete and refused counter paths, prefix mismatch, failed dependencies, binary identity,
five-round order and the session dependency dry run. They do not test ROCm or vLLM. Real producer-bound inputs are not
available locally. Native HIP executable linking, GPU loading, target compiler compatibility, profiler counter/marker
availability and all model correctness/performance results remain real-input or device gates. Target ROCm may differ
from the retained 7.2.4 device evidence. Reuse of device evidence requires an exact generated-source/plan identity check;
the changed host requires its own compiler check. No fresh device-code pass is implied by a retained artifact.

Local R1/R2 repair evidence, before/after regressions and exact commands are in
[the repair acceptance record](evidence/deepseek-prep-2026-10-15/b4-review-repair/acceptance.json).
Independent scoped repair review CLOSED R1/R2; see the retained repair review and source bindings. CPU acceptance is not profiler/device certification.
