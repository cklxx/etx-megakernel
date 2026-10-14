# ETX on MI300X: measured status

**One line.** On one MI300X serving DeepSeek-V2-Lite (bf16, 1024-token context, 32 greedy tokens per request), ETX
beats the strongest vLLM configuration we could build at one and two concurrent requests and loses at four. Every
number below was produced in one session on one box, with the two engines measured in interleaved pairs.

## 1. The result

| Concurrent requests | ETX tok/s | vLLM tok/s | Difference | Verdict |
|---|---|---|---|---|
| 1 | 274 | 206 | **+33.0%** | WIN (spread 1.5%) |
| 2 | 403 | 354 | **+14.0%** | WIN (spread 0.6%) |
| 4 | 410 | 567 | **-27.7%** | LOSS (spread 6.2%) |
| 8 | — | — | — | not measured; the arm is incorrect (section 4) |

Best single-request latency observed, from a separate session at the current configuration: **3.068–3.075 ms per token
on device** (about 326 tok/s). The table's 274 tok/s is the decision number: it is the same-session, interleaved
comparison against vLLM, so it is the only one that supports a claim about the other engine.

**Total throughput and per-request latency move in opposite directions**, so both are reported. At batch 2, total
throughput rises 14% while per-request latency rises 10%.

## 2. How the numbers were produced

- **One box, one runner, one cap.** One MI300X VF (192 GB, ROCm 7.14). Both engines run in the same session, in
  alternating order, on the same inputs.
- **Pairs, not single runs.** Five interleaved rounds; the reported figure is the median of the paired differences with
  its spread. Every verdict rule was committed to the repository **before** the code it judges.
- **Per-request correctness gates.** Each of the 32 tokens of each request must match a vLLM-produced reference
  (32/32 tokens, 27/27 layers). A number from an arm that failed its gate is not reported at all.
- **Sealed record.** Session logs are hashed and sealed on the box, verified on fetch file by file, and the verdicts can
  be recomputed from the raw logs without the summary JSON.

## 3. Why batch 4 loses while batch 1 and 2 win

The batch arms replicate work per request. Bytes read per decode step therefore grow linearly with the number of
requests, while the measured step time stops improving: at batch 2 ETX already sits at a bandwidth ceiling, and the
extra requests of batch 4 add bytes without adding throughput. This is a structural property of the current
replication, not a scheduling accident, and the fix is already named in the design: process the batch by columns so the
weights are read once for all requests of a step. That work item now has its trigger measurement.

## 4. What is not yet true

- **Batch 8 has no verdict.** The current eight-request arm fails its correctness gate deterministically, at the same
  request and the same layer. A first diagnostic that appeared to compare four candidate configurations was withdrawn:
  it ran one binary under four labels, so it compared nothing. The tooling defect is fixed and tested locally; the
  correct comparison needs a fresh box.
- **Cross-model portability is not free.** The same compiler switch that is worth **+330 µs** per token on DeepSeek is
  worth **+9 µs** on Qwen, with no pair faster — a measured NO-GO, not a tuning detail. A reusable inference system
  therefore has to earn its per-model gains; it cannot assume them.
- **Untested:** models larger than DeepSeek-V2-Lite, batch 8, contexts beyond 1024, and anything on a second GPU.

## 5. Cost of the evidence

The most recent box was alive **5 h 47 min and billed 6 hours (~US$11.94) for roughly 40 minutes of computation**:
DigitalOcean bills the hour that has started, and the gap was idle time between operator turns. An earlier box in the
same period has no recorded destroy time, so its cost is not claimed here. Both holes are now closed at the mechanism
level: a watchdog destroys any rented GPU box ten minutes before the next billed hour unless a deadline says otherwise,
and it reports the billed hours when it does.

## 6. Where the time goes next

1. **Batch 8 correctness** — find and fix the eight-request defect, then gate it. Local work; one gate costs a minute of
   GPU once a candidate exists.
2. **Column-batched tiles** — the fix that makes batch 4 and above pay, with its rules written before its code.
3. **A second model on the same switches** — Qwen is measured; the next model tests whether the lever set is a system
   or a one-model artefact.
