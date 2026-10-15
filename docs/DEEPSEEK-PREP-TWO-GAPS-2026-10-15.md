# DeepSeek: two prepared experiments, 2026-10-15

Status: implemented; local acceptance is recorded below. No GPU run, rental, or spend. This change prepares only the two open
experiments in `NEXT-GPU-2026-10-15.md`. It does not recover the lost historical r5/r7 payloads.

## Queue 1: eight-request correctness

Session: `examples/sessions/x8_correctness.py`. Hard limit: 3600 seconds on one GPU. Estimated device work:
5–10 minutes for six diagnostics, plus 10–20 minutes if all references need generation. The current provider is
DigitalOcean. Price, billing minimum, capacity, and spending approval are unknown. The session does not provision.

The new `x1km` arm has the five existing levers, L1 loads, shared kv_a, and four-wave merge, as `x8` does. D4 is off.
It uses one request. This controls the numerical recipe, not binary or schedule identity. One request uses the
single-request graph names, one request parameter block, unpaired resource allocation, and one worker slot per CU.
Eight requests use separate request grids, two slots per CU, and request r on slot r mod 2. Graph size, register
pressure, spills, and overlap can differ. `-DETX_SLOT_BY_CU` remains explicit in the host control build.

The session reuses the pinned Fleet export and the existing model and weight caches. It verifies the measured
weight-pack hashes. For each reference, it requires a tuple binding and a producer receipt. The receipt binds the
pinned producer source, rotation wrapper, complete model-file hashes, package versions, HIP version, GPU architecture,
and explicit bf16/eager/context-1024/decode-32/seed-0 recipe. It also binds a retained full producer log and exit 0.
An old byte binding alone is insufficient. The cache-reuse path verifies the producer log hash.

If a reusable reference is absent, the registered GPU preparation job creates it in a fresh result directory.
It certifies the tuple only after the pinned producer returns 0. It does not add a certificate to an old tuple.
The original cache remains intact. `references_ready.json` records reuse or generation per request and refuses
missing provenance without `--regenerate-missing`. Generation uses a 180-second per-request deadline, based on the
retained approximately 42-second reference run plus load and execution margin. It is a new-input reproduction.

The model diagnostics run in this order:

1. `x1km` on bound r5 and r7 separately.
2. The kept `alld4r0l1kvsmw` arm on those same two references as a secondary recipe control.
3. `x8` twice, on the same full eight-request reference set.

All six invocations retain raw output, exit codes, executable and plan hashes, recipes, input receipts, and
27 `LAYERGATE` rows per request. There are no performance rounds. A completed hidden-state failure is retained;
the remaining declared diagnostics still run. A missing input binding stops model diagnosis.

A pass needs 32/32 free-greedy tokens AND 32/32 warm-up tokens AND 27/27 consecutive hidden-state boundaries.
The unchanged boundary thresholds are relative max error <=0.02 and cosine >=0.999. Missing, duplicate, or
nonfinite metrics are INCOMPLETE. Tokens alone cannot pass. Hidden boundaries cover the first decode step only.

The structured classifier returns:

- A matched single-request failure: concurrency is not necessary for this failure.
- Both single-request controls pass, but x8 fails twice: failure depends on the multi-request path. This is not
  proof of a race. Scheduling, spills, and visibility remain separate explanations.
- x8 passes twice: failure not reproduced on these certified inputs. The historical failure is not repaired.
- Incomplete evidence or differing repeats: unresolved.

`x8_correctness.json` records each diagnostic; the final lane writes `x8_summary.json` even after normal dependency
failure. The runner hard deadline can skip pending final jobs. In that case its raw logs and `status.json` are the
incomplete record. No diagnostic timing has a performance verdict.

## Queue 2: four-request attribution and corrected baseline

Session: `examples/sessions/ds_b4_attribution.py`. It has a separate 7200-second cap. See
`DEEPSEEK-B4-PREP-2026-10-15.md` for its commands, limits, and acceptance conditions. It runs only B1/B2/B4.
It does not consume the correctness queue's cap and contains no new tile design.

The historical useful-payload model gives 19.738 GB at B4, or 4.590 ms at an assumed 4.3 TB/s. Historical ETX/vLLM
reported step medians were 9.760/7.054 ms. The ETX field used a HIP-event device span; the vLLM field used a
host full-minus-short estimate. The old label did not establish a matched wall-time comparison. These values set context only. Actual traffic and the current gap/body/wait
ceiling remain unknown until the registered captures pass. The old baseline omitted short-call gates and cannot
be transferred into a new performance verdict.

The two required baseline identities are fixed: the same compiled tiles through HIP-graph replay, and the
fastest legal configuration among pinned-image vLLM default and AITER at the same B. The existing same-tile replay
path has a structural blocker for these DeepSeek arms: `fleet_expert_down_cached` reads gate_up's LDS state, whose
lifetime ends at a separate kernel boundary. The queue must refuse that path locally. No replay-safe variant or new
kernel is part of these two experiments. This is an existing implementation limit, not an unmeasured bandwidth cost.

The queue can collect corrected ETX/vLLM system pairs and attribution data after their strict gates pass. The full
two-baseline speed claim remains INCOMPLETE while same-kernel replay is unavailable. Instrumented counters and
traces cannot replace uninstrumented paired timings. Five pairs use matched host clocks. No strongest-over-all-vLLM-
configurations claim is permitted.

## Local acceptance and remaining gates

The [local evidence index](evidence/deepseek-prep-2026-10-15/README.md) records exact tested source hashes, raw logs,
actual exit codes, generated-source comparisons, compiler identities, device objects, and host objects.
The frozen source archive of `a0c3e72` passed **278 focused tests**, with 14 graph-lowering tests initially excluded under the
Qwen heavy lock, in 15.26 seconds. SOL_CLOSE then passed all 14 (6.90 seconds) and the complete suite:
**2168 passed, 14 skipped in 508.28s (0:08:28)**, exit 0. The full run has no test deselection. The two Qwen archive transport cases use the
established real committed source fixture without historical GPU data; see the evidence index. Both queue dependency dry runs and outer/embedded shell syntax checks passed.
The new x1km device/assembly/full-host checks passed with ROCm 7.2.4 and native LLVM 23. All five relevant final
host objects compile. The CPU runner success and dependency-failure paths pass. Real r5/r7 cache checks refuse
with exit 1. Their original payloads remain absent. Unchanged retained builds are reused only after
explicit generated source, plan, patched Fleet source, and artifact checksum comparisons. Reuse is not a fresh
compiler pass.

No local compiler result proves device correctness. No local reference was generated on a GPU. The local cache
has no reusable producer-certified full reference set. The registered producer path remains a native GPU gate.
The installed offline compiler is ROCm 7.2.4; the future DigitalOcean image and ROCm version must be checked before
execution. Missing local HIP runtime libraries prevent final HIP executable linking here. Native link, runtime load,
real-input correctness, profiler metric/marker support, and all device measurements remain explicit execution gates.

The implementation and registered queues are committed. SOL_CLOSE adds final local acceptance and the independent review status
to this record for serialized main integration and dual-remote publication after a nonblocking review report. Local preparation is complete within the
two existing experiment scopes; a GPU run still needs a separate spending decision. Same-kernel graph
replay is a named existing implementation blocker, not a placeholder or a proposed third experiment.
