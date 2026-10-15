# ETX on MI300X: measured status, 2026-10-15

**One line.** On one MI300X serving DeepSeek-V2-Lite (bf16, 1024-token context, 32 greedy tokens per request), ETX
beats the strongest vLLM configuration we could build at one and two concurrent requests and loses at four. Every
number below was produced in one session on one box, with the two engines measured in interleaved pairs.

## 1. The result at the application level

| Concurrent requests | ETX tok/s | vLLM tok/s | Difference | Verdict |
|---|---|---|---|---|
| 1 | 274 | 206 | **+33.0%** | WIN (spread 1.5%) |
| 2 | 403 | 354 | **+14.0%** | WIN (spread 0.6%) |
| 4 | 410 | 567 | **-27.7%** | LOSS (spread 6.2%) |
| 8 | — | — | — | not measured; the arm is incorrect (section 5) |

**Every absolute number below belongs to one machine, and the machines differ by more than any optimisation we have
measured.** The batch table's 274 tok/s is the shipped recipe on the batch box. The same binary on an earlier box
measured **3.068–3.075 ms per token on device (about 326 tok/s)** across five rounds, while on the batch box it
measured **3.609–3.615 ms** across its five rounds: a **+17.6% offset (+542 µs/token)**, stable to 6 µs inside each box
and larger than every change in section 2 combined (41–120 µs each). The batch table's 274 tok/s is the decision number
because it is the same-box, interleaved comparison against vLLM, and only that comparison supports a claim about the
other engine; the 326 tok/s figure is a different machine's capability and must not be read against that table.

**A box offset is now a measured quantity, not an assumption.** The batch session never re-ran the previous
configuration on its own box, so it cannot say whether the offset is the machine, the batch harness at batch 1, or
both. Every future session runs a **box control**: the shipped recipe at batch 1, plus the previous shipped recipe,
on the same box as the study, so the machine's offset is measured and every absolute number can be converted between
boxes instead of assumed equal.

**Total throughput and per-request latency move in opposite directions**, so both are reported. At batch 2, total
throughput rises 14% while per-request latency rises 10%.

## 2. The experiment list, item by item

Eleven studies: six on the single-request recipe, two on a second model, one batch study, and two design items that
consumed no GPU time. Each one holds only if its own pre-registered rule passes, on interleaved pairs, behind
per-request correctness gates; a number from an item that failed its gate is not reported.

### 2.1 DeepSeek-V2-Lite per-token cost — six items, all measured 2026-10-14

| Item (the mechanism, not a code name) | Measured pairs, µs | Median / spread | Outcome |
|---|---|---|---|
| Replicate the kv projection once per chiplet instead of eight times | 40, 40, 41, 48, 44 | **+41** / 8 | **KEEP** |
| Spread one weight matrix's GEMV over all four waves | 76, 65, 55, 74, 66 | **+66** / 21 | **KEEP** |
| Remove the terminal pass in the GEMV path | -29, -29, -23, -32, -22 | **-29** / 10 | **STAYS** — removing it costs 29 µs; the switch is load-bearing inside the current recipe |
| Remove the argmax tree | -13, -6, -21, -6, -14 | **-13** / 15 | **STAYS** — every pair slower, inside twice the spread |
| Both kept levers together | 48, 52, 50, 58, 54 | **+52** over the better lever alone; **+120** over the previous baseline | **KEEP** — today's shipped configuration |
| Single-request trace (attribution only) | span 3380.9 µs | — | recorded; decides nothing by itself |

Rules, all committed before the code: every pair faster, median >= 30 µs, median > twice the spread. The two numbers in
the fifth row measure different things and that distinction is the point: **+52 µs** decides what ships, **+120 µs** is
attribution against the old baseline. Both were recomputed from the 15 raw run logs rather than from the summary file.
**Net effect of the six: the shipped single-request configuration is 120 µs/token faster than the one it replaced**
(3.19 -> 3.07 ms device), with every correctness gate green (32/32 tokens, 27/27 layers, 15 runs).

### 2.2 Qwen — the same switches on a second model: two items, both measured, both negative

| Item | Measured | Outcome |
|---|---|---|
| Reassign MoE row ownership between workgroups | median **-6.01%**, spread 44.7%; **+3.42%** on a second session (raw records; the pair of medians quoted for weeks, -0.12% and +14.29%, appears in no raw record and was retracted on 2026-10-15) | **NO-GO twice with opposite medians** — the microbenchmark cannot decide it, which is a finding about the measurement, not the lever |
| One compiler switch that makes cached blocks coherent at a tighter scope | median **+9.0 µs**, spread 19.0 µs; pairs 9.0 / 10.0 / -5.0 / 14.0 / 0.0 (three faster, one slower, one equal) | **NO-GO** — the median fails the >= 30 µs and > 2x spread rule |

The second row is the platform-level result: the **same switch is worth +330 µs on DeepSeek and +9 µs on Qwen** (and
three of its five Qwen pairs were faster, so this is a failed threshold, not a uniformly slower switch), and
both models compile and pass their correctness gates (32/32, import certificates at two positions). Cross-model
portability is therefore a work item with a measured negative example, not an assumption.

### 2.3 Batch 2 / 4 / 8 — one study, three arms

Section 1's table. Two of its three correctness gates pass (batch 2 and batch 4). Batch 8's arm fails
deterministically, at the same request and the same layer; its recipe comparison was withdrawn after review because
the harness ran one binary under four labels, so it compared nothing.

### 2.4 Design-only items — four proposals, no GPU time

A router grid with rows remapped per chiplet; an epoch-tagged event scheme with a static-only reset; a producer-side,
row-partitioned combine; and a last-arriver top-k. Each has a written rule and a planned local check. None is built,
none is queued, and none has been on a GPU. The fifth proposal of that set (pre-shuffling weights) was implemented
earlier and is kept.

### 2.5 The preceding stage, 2026-10-13

Five decisions on the same model were taken through the same rule-first process on a rented MI300X; that record is
`docs/DSV2-D1-D5.md` with its decision register. One item (a payload-load variant of the cache experiment) is
**incomplete**: the box was destroyed mid-session. It is recorded as a known loss together with what a re-run costs.

## 3. How the numbers were produced

- **One box, one runner, one cap.** One MI300X VF (192 GB, ROCm 7.14). Both engines run in the same session, in
  alternating order, on the same inputs.
- **Pairs, not single runs.** Five interleaved rounds; the reported figure is the median of the paired differences with
  its spread. Every verdict rule was committed to the repository **before** the code it judges.
- **Per-request correctness gates.** Each of the 32 tokens of each request must match a vLLM-produced reference
  (32/32 tokens, 27/27 layers). A number from an arm that failed its gate is not reported at all.
- **Sealed record.** Session logs are hashed and sealed on the box, verified on fetch file by file, and the verdicts can
  be recomputed from the raw logs without the summary JSON.

## 4. Why batch 4 loses while batch 1 and 2 win

The batch arms replicate work per request: every request of a step streams its own weight set, on a fixed overlap
capacity. Bytes per step therefore grow with the number of requests while the useful rate stops rising — 1352, 1990,
2022 GB/s at one, two and four requests — and a descriptive fit (`s(B) ~ (B/2) x s(2)`: 9.92 ms predicted against 9.76
ms measured) is what nearly doubled queued work looks like.

**That is the measurement; the cause is not isolated, and we do not claim it is.** The same numbers are consistent with
HBM saturation, with serialized task bodies under a longer queue, with load-issue limits, and with the spill traffic the
batch arms carry (64 B/lane of scratch at batch 4, none in the single-request recipe). An internal attribution pass
also retired two things this report used to imply: the "5.20 GB per request" figure is an older measurement from a
different engine version rather than traffic measured for these arms, and the byte floor does not force the loss at all
(19.7 GB at the assumed 4.3 TB/s is 4.59 ms, below vLLM's measured 7.05 ms).

The fix named in the design is unchanged — process the batch by columns so the weights are read once for all requests
of a step — and the decisive next measurement is now written down: measured memory traffic and device span per arm,
plus per-phase body/wait spans and spill resources, which separates a true bandwidth plateau from serialized work.

## 5. What is not yet true

- **Batch 8 has no verdict.** The eight-request arm fails its hidden-state gate deterministically — requests 5 and 7
  both, with their greedy tokens still 32/32 correct, and the "3 layers" in the log is where the gate stopped counting,
  not a count of bad layers. The four-recipe comparison was withdrawn (one binary under four labels); that tooling
  defect is fixed and tested locally, and the correct comparison needs a fresh box.
- **Cross-model portability is not free** (section 2.2).
- **Untested:** models larger than DeepSeek-V2-Lite, batch 8, contexts beyond 1024, anything on a second GPU, and the
  four design-only proposals.

## 6. Cost of the evidence

The most recent box was alive **5 h 47 min and billed 6 hours (~US$11.94) for roughly 40 minutes of computation**:
DigitalOcean bills the hour that has started, and the gap was idle time between operator turns. An earlier box in the
same period has no recorded destroy time, so its cost is not claimed here. Both holes are now closed at the mechanism
level: a watchdog destroys any rented GPU box ten minutes before the next billed hour unless a deadline says otherwise,
and it reports the billed hours when it does.

## 7. Where the time goes next

1. **Batch 8 correctness** — find and fix the eight-request defect, then gate it. Local work; one gate costs a minute of
   GPU once a candidate exists.
2. **Column-batched tiles** — the fix that makes batch 4 and above pay, with its rules written before its code.
3. **A second model on the same switches** — Qwen is measured; the next model tests whether the lever set is a system
   or a one-model artefact.

An independent audit of the harness behind these numbers found eight blocking defects (job scheduling, input binding,
gate parsing, verdict verification). They are being fixed in the same rule-first way, because a confident wrong number
is worse than no number.
