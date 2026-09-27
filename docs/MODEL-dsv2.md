# DeepSeek-V2-Lite decode on MI300X: an operation-level model, its calibration, and what we do when it is wrong

Status: pre-registration, written 2026-09-27 before the measurements it describes. Everything here is either measured (with its
source) or marked as a prediction; the predictions and the alternative explanations are fixed now, so that a mismatch leads to
a planned measurement instead of an after-the-fact story.

## 1. Method

1. **Top-down ledger (done):** `examples/dsv2lite/ledger.py` assigns every microsecond of a traced token to handoff, ramp,
   staging, prologue, bandwidth floor, stream excess and tail, per phase; the categories add up to the token exactly. Results
   for fleet, ETX (one request) and ETX (two requests): design doc 15.13.
2. **Bottom-up operation inventory:** for every task kind, every global read (bytes, source, shared or unique, load width,
   loads in flight), write, LDS access, barrier, reduction, arithmetic, atomic and in-body wait, with formulas in the model
   dimensions (`docs/OPS-dsv2.md`, from the fleet source, file:line for every step).
3. **Hardware model:** a cost for each operation class as a function of its parameters and of what else runs at the same time
   (section 2), every parameter calibrated by a microbenchmark (section 4).
4. **Prediction and reconciliation:** the model predicts each ledger category per phase; the residual per category is what is
   still unexplained. A residual above the thresholds of section 3 triggers that category's discriminating measurements.

## 2. Hardware parameters

| # | parameter | what depends on it | value now | source | status |
|---|---|---|---|---|---|
| H1 | HBM stream rate vs loads in flight per CU (304 CUs streaming) | stream floor per phase | 2.67 (depth 1) .. 4.33 TB/s (depth 8) | fleet microbench (c) | measured |
| H2 | HBM stream rate vs number of active CUs (8 .. 304) and bytes per workgroup (short streams) | stream excess of small phases (o_proj 28 KB/task, merge, router) | - | - | **to measure** |
| H3 | idle latency ladder L1 / L2 / MALL / HBM | round trips in prologues | 57 / 100-108 / 205-218 / 333-350 ns | pointer chase, memory note | measured |
| H4 | loaded latency vs background bandwidth | prologue round trips during another phase's stream | 1.40 -> 6.16 us cross-XCD event at 1.73 TB/s | fleet microbench (e) | partial |
| H5 | N workgroups reading the same 64 KB, vs N and vs where it lives (L2 / other XCD via MALL) | staging (folds, kv_a, routing logits) | 1.12 us (8 WGs), 3.24 us (304 WGs) | overlap_bench fold | partial: 2 points |
| H6 | same-XCD hand-off (release, arrive, poll, acquire), 37 producers -> 37 consumers | handoff and ramp of XCD-local events | 0.96 us one-way, one producer | sync_bench xcd1 | partial |
| H7 | device-wide hand-off, 296 producers -> 296 consumers, with the last-arriver flush | handoff of o_proj -> router, down -> qkv | barrier 7.6-7.8 us idle, 21 us with dirty L2 | sync_bench barrier | partial |
| H8 | poll backoff granularity (s_sleep 8 = ~256 cycles) and wake-up spread | ramp | - | - | **to measure** |
| H9 | `buffer_wbl2 sc1` cost vs dirty bytes; `buffer_inv sc1` cost vs L2 contents | handoff; the 7% cost of the sc1 domain acquire | - | - | **to measure** |
| H10 | `__syncthreads`, LDS write/read, wave and cross-wave reductions | prologue (norm, top-k, fold reduce) | - | - | **to measure** |
| H11 | VALU / MFMA throughput per CU for the tile mixes (bf16 -> fp32 dot, exp, rsqrt) | attention, router, top-k | - | - | **to measure** |
| H12 | two workgroups per CU: stream + stream, stream + latency chain, chain + chain | every category of the two-request build | fold x1.00-1.04 beside a stream; real phases x1.4 | overlap_bench; ledger | **contradiction, to explain** |
| H13 | GPU clock under full load (s_memtime cycles / s_memrealtime) | everything compute- or issue-bound | - | - | **to measure** |
| H14 | instruction / scalar cache misses of the megakernel (code size, 110k-task dispatch) | per-task overhead, ETX vs fleet | - | - | **to measure** |
| H15 | TLB reach over the 31 GB weight blob | stream excess | - | - | **to measure** |
| H16 | scratch traffic of spilled registers (32-448 B/lane) | per-task overhead of the patched builds | - | - | **to measure** |

## 3. Predictions and the alternatives, decided now

Each row: the prediction, the thresholds that count as a mismatch, the alternative explanations, the measurement that tells
them apart, and the signature each alternative would leave.

### 3.1 Stream excess (ETX one request: 47 us per MoE layer above the 38 us floor)

Prediction: the excess is the sum of per-phase stream start-up (first bytes after the prologue) and tail (the last batch of
each worker's rows), from H1/H2 at the phase's bytes per task and active CU count. Mismatch: residual > 20% of the excess.

| alternative | measurement | signature if true |
|---|---|---|
| A1 short streams: per-task bytes too small to reach the rate (o_proj 28 KB, router 0.9 KB, merge 8 KB per task) | H2 sweep at the real bytes/task | isolated phase time = H2 prediction; excess concentrated in small phases |
| A2 depth: one wave per SIMD with 8 x 16 B in flight is below H1's knee under load | in-situ phase benchmark with FLEET stream depth variants (patched copy) | excess falls with depth |
| A3 HBM channel / bank conflicts from 37 workers striding rows by 37 | H2 with the real row stride vs contiguous | stride-dependent rate |
| A4 TLB misses (31 GB blob) | counters (UTCL2 / TLB miss) if exposed; huge-page allocation variant | excess scales with distinct pages touched |
| A5 clock throttling under full load | H13 during the phase | clock below nominal during streams |
| A6 contention with other traffic (KV cache, partial writes, polls) | in-situ phase alone vs in the token | phase faster alone |
| A7 megakernel-specific overhead (registers, code size, scalar loads of descriptors) | same phase in ETX's unfused mode (one launch per grid) and in fleet | unfused / fleet faster per task |

### 3.2 Staging (fleet 20.4 us, ETX 14.0 us per MoE layer)

Prediction: staging = one round trip of the shared vector at the H5 cost for the number of concurrent readers plus the
source's location (the producer's XCD L2, or MALL / HBM after the last-arriver write-back). Mismatch: residual > 1 us per phase.

| alternative | measurement | signature |
|---|---|---|
| B1 reader contention on the same lines (304 readers of 64 KB) | H5 curve vs N | cost grows with readers, not bytes |
| B2 source location: lines written by another XCD come from MALL / HBM after `buffer_inv sc1` | H5 with a remote producer + wbl2 vs local | remote >> local |
| B3 loaded latency: staging overlaps the tail of the previous phase's stream | H4 at the measured background rate | staging tracks the background bandwidth |
| B4 ETX vs fleet difference (-6 us in ETX) comes from where the fold reads its partials (ETX acquire order, fewer readers at once because of ramp) | per-task staging vs ready time within a phase (trace + stamps) | early starters stage faster |

### 3.3 Handoff and ramp (ETX 5.1 + 11.6 us, fleet -3.0 + 6.9 us per MoE layer)

Prediction: handoff = release + arrive + poll interval + acquire (H6/H7/H8/H9), ramp = the spread of the 37 (XCD-local) or
296 (device) consumers' wake-ups, which is the poll interval plus XCD skew of the producers. Mismatch: residual > 1 us per
event.

| alternative | measurement | signature |
|---|---|---|
| C1 poll interval (s_sleep 8) | event microbenchmark with s_sleep 0 / 1 / 8 / 32 replicating 37 -> 37 | ramp tracks the sleep |
| C2 serialization of 37 arrivals on one counter | arrivals-only microbenchmark vs producer count | handoff grows linearly with producers |
| C3 release write-back (wbl2) on device events | H9 with the phase's dirty bytes | device hand-offs cost more with more dirty lines |
| C4 acquire (inv sc1) refilling L2 | H9; the sc0 vs sc1 A/B (0.25 ms/token on DeepSeek) | cost per acquire x acquires per layer = the A/B |
| C5 fleet's early start: the fastest XCD begins before the slowest ends (negative handoff) where ETX waits on a device event | ledger per XCD (first ready per XCD vs that XCD's producer end) | ETX's device events vs fleet's XCD-local ones |
| C6 descriptor fetch and dispatch in the ETX loop between tasks | trace: taken -> ready when deps are already met | constant per-task overhead |

### 3.4 Tail (about 10 us per MoE layer)

Prediction: the slowest worker's extra rows (ceil(N/296) vs N/296) plus the slowest XCD's start skew. Mismatch: residual > 1 us.

| alternative | measurement | signature |
|---|---|---|
| D1 uneven rows per worker | inventory (rows per worker) vs per-worker done times | the late workers own more rows |
| D2 XCD imbalance (HBM stacks, placement) | per-XCD phase end from the trace, repeated runs | the same XCD is always last |
| D3 random stalls (refresh, contention) | distribution over layers and runs | tail varies run to run, not by XCD |

### 3.5 Two requests per CU (H12's contradiction)

Prediction: co-resident tasks slow each other by the resource both use most; a pure memory round trip (the fold) does not slow
beside a stream (measured), so the observed x1.4 comes from the non-memory parts. Mismatch: the in-situ pair benchmark shows a
slowdown the per-resource model does not predict within 15%.

| alternative | measurement | signature |
|---|---|---|
| E1 VALU / MFMA issue: attention and top-k compute share the SIMD with a GEMV | pair benchmark: attention (real tile) beside gate_up (real tile), vs beside an idle workgroup | slowdown ~ compute share |
| E2 LDS bandwidth / bank conflicts (norm staging, top-k, attention partials beside gate_up's LDS use) | pair benchmark with an LDS-only chain | slowdown with LDS traffic |
| E3 memory issue slots: two waves per SIMD each with 8 loads in flight halve per-wave bandwidth | pair benchmark stream + stream at the real depth | per-workgroup rate halves |
| E4 L1 / L2 thrashing: the stream evicts the latency phase's working set (partials, KV) | pair benchmark; L2 hit counters | the latency task's loads miss more |
| E5 spills / noinline calls of the patched build | x1 with the patch and 1 WG/CU (x1cap, measured +2.6%) vs x1 | a fixed per-call cost |
| E6 i-cache misses (two requests' task types interleaved, 110k-task megakernel) | counters; x2 with identical requests (same prompt) vs different | misses scale with code footprint |

## 4. Microbenchmarks

**Hardware primitives** (`bench/hw/hw_bench.hip`, one binary, each mode a RESULT line): H2 stream rate over active CUs x bytes
per workgroup x row stride; H4 loaded latency; H5 shared-read cost over readers x bytes x source (local L2, remote XCD after
wbl2); H6-H8 the ETX hand-off (37 -> 37 XCD-local, 296 -> 296 device, s_sleep sweep, relay); H9 fence costs vs dirty and
resident bytes; H10 barrier, LDS and reduction costs; H11 VALU / MFMA rates for the tile mixes; H13 clock under load; pairs of
two workgroups per CU for E1-E4.

**In situ, with the real tiles** (`ETX_DSV2_ISOLATE`): the ETX graph built from a chosen subset of grids (for example only
`qkv_5`, or `r0_attn_5` with `r1_gate_up_5` on the other workgroup of each CU), no cross dependencies, real weights and
activations, traced. It measures a phase alone, next to one other phase, or with one variable changed, with fleet's own code,
which is the only way to separate "the phase is slow" from "the phase is slow here".

**Counters** (rocprofv3 --pmc on one-token runs; one value per dispatch, so for whole tokens only): clock, VALU / LDS / VMEM
instruction counts, L2 hit rate, TLB and instruction-cache misses where the counters exist on gfx942.

## 5. Reconciliation (2026-09-27, sessions ds_model and ds_evstride; logs examples/dsv2lite/results/2026-09-27d)

Measured parameters: H2 (stream replica) at the phases' bytes per task, rows layout: router 3.2, merge 3.4, o_proj 4.9,
q/kv_a (51 KB) 6.1, down (160 KB) 16.4, gate_up (320 KB) 32.7 us phase span (contiguous per-WG layout is slower: q/kv_a
10.1). H5: a 64 KB vector read right after another XCD wrote and wrote it back: 1 reader 3.9 us, 296 readers 6.0 us.
H6/H8: XCD-local hand-off 3.28 us with the 8 XCDs' counters in one cache line, **1.00 us padded**; 5.3 -> 1.4 us with 64 KB
dirty per producer; device-wide 1.2-1.4 us; s_sleep 0/1/8/32 and acquire sc0/sc1 do not change it. H10: `__syncthreads`
77 ns, one bpermute ~56 ns (6-step reduce 337 ns), 8 KB LDS write+read 613 ns. H13: 1.99 GHz under a full FMA load.
Counters (one token): instruction-cache miss rate 0.2%, UTCL1 translation miss rate 0.1%.

In situ, the real tiles, each phase of layer 5 alone (us): q/kv_a 20.1, q_absorb 2.9, attention 14.1, merge 9.9, o_proj 8.8,
router 10.4, gate_up 34.0, down 20.5: 117.8 in sum against 136.3 per layer inside the token, so 18.5 us per layer are
hand-offs.

| category (per MoE layer) | prediction / question | result | verdict |
|---|---|---|---|
| handoff + ramp (ETX 5.1 + 11.6 us, fleet -3.0 + 6.9) | C1-C6 | none of C1-C5 explains it (sleep and acquire do not matter); the hand-off replica with the counters padded took 1.0 us where packed took 3.3 us: **C7, false sharing of the event counters, not in the pre-registered list**, found because the XCD-local and device replicas disagreed | fixed: `ETX_EV_STRIDE` 32 (machine model default on gfx942). One request 3.72 -> 3.47 ms/token (fleet 3.55); ledger handoff + ramp 16.7 -> 9.3 us per layer, layer 136.3 -> 124.6 us |
| stream excess of q/kv_a (ETX 11.1 us above the 3.5 us floor) | A1-A7 | issued bytes, not rate: every XCD computes all 576 kv_a rows (no `FLAG_KVA_SHARED`) and each wave over-issues one group, 180 KB per task where 50 KB are unique; the replica at 160-180 KB per task gives 12-16 us, the stamped stream is 14.6 | explained (A1 generalised to "issued bytes"); A4 TLB and A7 i-cache rejected by counters |
| stream excess of gate_up / down | A1-A7 | replica at 344 KB / 180 KB issued per task: 22.7-32.7 / 12.1-16.4 us; measured 26.5 / 16.3 | explained |
| staging (fleet 20.4, ETX 14.0 us per layer) | B1-B4 | a fresh remote 64 KB costs 3.9 us to one reader, 6.0 us to 296: the source (written back by another XCD) dominates (B2), reader count adds ~50% (B1); ETX's readers start spread (ramp), so they see fewer concurrent readers (B4) | explained within 1 us per phase |
| prologue: norm ~2.5 us, gate_up routing 4.7 us | from the inventory | rms_norm = 10-level LDS tree with a barrier each (~1.5 us) + a gamma round trip (~1 us); routing = 84 serial bpermutes x 56 ns = 4.7 us, exactly fleet's own stamp | explained |
| tail (~10 us) | D1-D3 | rows per worker differ by one (e.g. q/kv_a 16 vs 15 kv_a rows); consistent with D1; D2 / D3 not separated | partly |
| two requests x1.4 (H12) | E1-E6 | pairs: a compute-bound workgroup slows x1.74 beside a stream, a latency chain x1.14, stream beside stream x1.8; in situ: q/kv_a x1.76, attention x1.94, router x1.39 beside gate_up, gate_up x1.08-1.26 beside them, same-phase pairs x1.35-1.55; SQ_WAIT_INST_ANY 11x with two requests; any pairing saves only 16-30% over running the two phases one after the other, which bounds two requests at ~1.3-1.4x | E1 (issue contention of compute-heavy phases) confirmed; E4 (L2) and E6 (i-cache) rejected by counters |

With the counter fix the two-request build runs 4.81 ms per step, **416 tokens/s**: 1.47x fleet at batch 1 and 10.6% above
vLLM at batch 2 (376 tokens/s, same prompts, measured 2026-09-27); per-request latency 4.76 ms against vLLM's 5.32.

Still open: (1) a streaming workgroup runs ~10% faster beside a compute-bound one than beside an idle one (pair test, with
the Infinity Cache evicted between configurations): H13 under pure streaming (a clock drop when the SIMDs idle) is the
first candidate, to be measured with s_memtime inside the stream. (2) The ETX_EV_STRIDE default also applies to the Qwen
(vLLM-kernel) builds, which have not been re-measured with it.
