# GLM FFN TP/EP code and archived-log analysis — 2026-10-22

## Status and scope

**Historical FFN-only scope below; subsequent CPU-only implementation adds
optional head-sharded attention and a separate up-tail switch.** See
[the variant runbook](PARALLEL-VARIANTS-2026-10-22.md) for current options and gates.
The earlier ceiling estimates below apply to attention_tp=0, not the new mode.

**Code written; not compiled, tested or executed. No TP/EP performance result.**
The owner stopped `162.243.244.52`. This work used local source edits and archived
log analysis only: no SSH, GPU work, provisioning, local tests or compilation.
Static inspection is not build validation. Regression test sources were added
but have not run. No independent full-model HF correctness certificate exists.

This implements **FFN tensor parallelism**, not full-layer attention TP:

| Mode | Parallel work | Rank-0-only work |
|---|---|---|
| `parallel=0` | None for one token; whole layers partitioned as PP | Embedding/feedback on 0; head on last stage |
| `parallel=1` | Dense and sparse FFN gate/up intermediate rows, matching down columns, across 2/4/8 ranks | All attention, states, normalization/mHC, router/top-k, combine, embedding/head |
| `parallel=2` | Routed experts by `expert_id % world`, across 2/4/8 ranks | All attention, dense FFNs, shared expert, router/top-k, ordered combine, embedding/head |

PP fits the roughly 330 GB whole-model checkpoint; it does not provide same-token
compute scaling at batch 1. The next layer waits for the preceding layer, and the
next generated token waits for the current logits. More PP stages reduce capacity
per card, not that dependency chain. TP/EP instead split work **inside one FFN**.
Serial attention is an intentional current limitation, not something hidden by the
TP label.

## Implemented path

1. `parallel_layout.py` derives compact common/rank weight blobs from stage tensor
   metadata, retaining the global slot ABI. It emits generated `.meta` files,
   `parallel-layout.json` and `ranges.tsv`, not additional checkpoint-sized files.
   TP cuts gate/up rows and down columns consistently, including scale matrices;
   FP8 cuts must respect 128-element block alignment. EP retains complete expert
   matrices on only their owning rank; the shared expert stays on rank 0.
2. `model_build.py` emits the argument ownership and exact loaded-byte contracts.
   The host loader loads outer weights normally and generated metadata from the
   build directory. It mmap-reads one original layer blob at a time, packs strided
   column slices in bounded 4 MiB host chunks, and uploads compact local payloads.
   It rejects out-of-bounds, overlapping or incomplete destination ranges. Padding
   and sentinel blobs are zeroed. The layout/ranges/metadata participate in the
   build identity, alongside helper/graph/generator source hashes.
3. `parallel_graph.py` creates route → rank-local dispatch → local up → local down
   → root combine. Every rank has a distinct packet, activation and partial-output
   tensor. Cross-device graph edges publish through SYSTEM events; local edges use
   device-local event storage. These are real local expert/weight shards, not
   replicated weights streamed from another GPU.
4. `parallel_phases.h`, through wrappers in `tiles.hip`, computes rank work and
   combines it. Dispatch performs peer input copies into local packets. Return
   uses peer-visible per-expert output buffers read by root after their SYSTEM
   completion events; it is **not** an RCCL all-reduce implementation. Communication
   completion must be included in eventual end-to-end timing.
5. `run.py` and `oracle.py` distinguish `etx-ffn-tp`, `etx-ep` and PP2/4/8. Standalone
   oracle comparisons accept an explicit `--candidate-backend`; the default remains
   PP2 for compatibility. Qualification verifies the existing build identity and
   all rank sources, uses the correct head owner, and strips ambient diagnostic
   environment flags. It does not invent a numerical tolerance or certificate.

### Numerical contract

EP preserves sorted selected-expert order, per-expert BF16 rounding, per-weighted
addition BF16 rounding, and shared-expert addition last. It must not first sum all
experts on each rank and then sum rank totals: BF16 addition is nonassociative.
Unowned output slots are zeroed; unowned descriptors and activation slots are not
read. Ranks with no selected routed expert still participate in graph completion.

TP produces raw FP32 down-projection partials, sums them in increasing rank order,
then applies per-expert BF16 rounding before the same ordered expert combination.
Splitting K changes the original dot-product summation order. **TP bit equality is
not assumed.** A combine-only regression source covers rank-order cancellation
and the error caused by rounding rank partials before summation; full sharded
matrix execution/numerics remain unverified.

### Communication and imbalance

Let hidden size be H, expert count E and selected count K. Each rank receives
`4 * (round4(E) + H + round4(2K))` bytes/request of router/input/selection packet.
Each rank currently materializes `4 * (K+1) * H` bytes/request of separate expert
down outputs, even for empty EP slots. TP root reads every rank's partials; EP root
reads each selected expert only from its owner, plus the local shared expert.
These are logical tensor volumes, **not measured peer-link traffic or time**.
The graph has no claim of communication/computation overlap.

Eight selected experts do not imply eight busy ranks: modulo ownership can
collide, and rank 0 also owns shared-expert work. Future EP evidence must include
selected-expert ownership histograms, per-rank active time and actual overlap.
The current down helper processes experts separately rather than the original
grouped multi-expert down loop, so even local kernel efficiency may differ.

## Reproducible analysis of archived logs

From the repository root (stdlib data analysis only):

```sh
python3 -I examples/glm53/analyze_logs.py \
  results/o55-events-20261022/le2run.log \
  results/o55-events-20261022/le4run.log \
  results/o55-events-20261022/le8run.log \
  results/o55-logs-20261022/*.log \
  --out results/o55-events-20261022/archived-log-analysis.json
```

The script requires exactly one full 45-layer completion marker and exactly the
ordered positions 0–1054, batch 1. It refuses to produce throughput statistics for
partial or duplicate runs. It retains log hashes, cold step, all-step/steady/early/
last-100/decode-32 distributions and resource warnings. The dense/sparse trace
attribution assumes this pinned model's first three layers are dense; it is not a
general-purpose schema inference tool. `log-analysis.json` retains the smaller
four-run v7/local-events subset.

### Instrumented candidate execution

Same MI325X box, gfx942, full 45 layers, batch 1, 1024-token prompt processed one
position at a time, 32 generated outputs. Trace and execution checks enabled.
**Wall timing = host launch submission through all streams retired**, excluding
reset, input upload, completion checks, logits D2H and file output. Thus these are
neither whole-request latency nor chunked-prefill throughput measurements.

| Run | All 1055 median ms | Decode 32 median ms | Decode mean ms | Decode p90 ms | Decode max ms | Cold position 0 ms |
|---|---:|---:|---:|---:|---:|---:|
| v7, shared events, PP2 | 71.264 | 78.453 | 78.110 | 81.371 | 81.817 | 115.576 |
| Local events, PP2 | 29.455 | 30.690 | 30.999 | 30.966 | 40.341 | 68.203 |
| Local events, PP4 | 30.547 | 31.653 | 31.686 | 32.139 | 32.357 | 144.409 |
| Local events, PP8 | 28.955 | 30.030 | 30.134 | 30.545 | 31.544 | 255.728 |

Every row completed 1055 steps. These are single runs, not interleaved repeats or
significance tests. The PP2 decode maximum is retained, not discarded as an
outlier. All-step medians mix prompt processing and generation; use the separately
identified decode window, positions 1023–1054, for generation comparisons.

The owner separately reported repeated-run PP2 **28.54 ms** versus PP8 **28.05 ms**.
Raw repeat logs, timing-window details and spread were not archived here. These
owner-reported values support no inference of linear PP scaling and must not be
mixed with the instrumented table or promoted to confidence intervals.

### Earlier archive: progression, not qualified speedups

| Archived log | Completion | All-step median ms | Decode-32 median ms |
|---|---|---:|---:|
| `old.log` | Partial: positions 0–5 only | Not reported | Not reported |
| `ph.log` | Complete | 200.261 | 237.996 |
| `tr4.log` | Complete | 199.870 | 236.360 |
| `tr1000.log` | Complete | 192.553 | 229.036 |
| `ni.log` | Complete | 179.024 | 211.207 |
| `v2.log` | Complete | 117.593 | 125.414 |
| `v2b.log` | Complete | 111.916 | 120.640 |
| `v3.log` | Complete | 113.702 | 119.341 |
| `v4.log` | Complete | 87.210 | 94.782 |
| `v4b.log` | Complete | 82.300 | 89.273 |
| `v5.log` | Complete | 75.575 | 83.475 |
| `v6.log` | Complete | 77.549 | 85.842 |
| `v7.log` | Complete | 71.264 | 78.453 |

Names identify archived artifacts, not isolated causal ablations. Different
variants/context traces changed conditions; the table does not assign each delta
to one optimization. The old six observed steps were approximately 50.66–51.29
seconds each, supporting the original diagnosis but **not a completed full-run
baseline**. Build logs and the archived short logs carry no full-run result, and
no HF reference job was restarted. Complete runs are still unqualified candidates.

### What the trace says about the FFN-only ceiling

PP2 local-events position 4 has host wall **28.353256 ms**. Summed grid spans:

| Work | Span sum ms |
|---|---:|
| Gate/up | 7.4225 |
| Down | 4.5374 |
| Attention prologue p1 | 5.3040 |
| Attention output projection | 2.7099 |
| KDA | 1.4838 |
| p2 | 1.4609 |
| Router | 0.9952 |

Up + down totals **11.9599 ms**, of which sparse-layer up/down accounts for
**11.2342 ms** (including the unisolated shared expert). The rest of the step will
not become parallel merely by introducing FFN TP/EP.

Using `T_estimate = T_host - F + F / degree`, keeping everything else fixed and
assuming perfect balance and zero communication:

| Degree | FFN-TP optimistic estimate ms | EP optimistic estimate ms |
|---|---:|---:|
| 2 | 22.373 | 22.736 |
| 4 | 19.383 | 19.928 |
| 8 | 17.888 | 18.523 |

**These are estimates from one early-context step, not measured TP/EP timings or
decode predictions.** EP's estimate even scales the shared-expert portion, which
actually remains serial, and is therefore additionally optimistic. Tile grouping,
occupancy, dispatch, peer reads, ordered reduction and imbalance can consume much
of the gain. Infinite FFN-TP scaling leaves about **16.393 ms** in this simple model.

An earlier report called 28.196 ms a global trace span and its 2.698 ms residual
pure inter-grid gaps. That interpretation is corrected: each rank's printed table
uses its **own earliest-taken origin**, so relative tables cannot be merged into a
globally aligned timeline. Summed grid spans total 25.4984 ms; the residual to host
wall is **2.854856 ms**, including host launch/retirement and scheduling, not just
kernel gaps. Raw timestamps needed for a global critical path are not locally
archived. Per-task mean wait columns must never be summed as elapsed overhead.

The earlier active-byte estimate was approximately 25 GB/token, implying about
6 ms at 4.24 TB/s; this is an **estimate**, not measured traffic. Dividing all
330 GB of checkpoint weights by bandwidth is not the selected-expert batch-1
traffic floor. Neither estimate establishes a communication or scaling result.

## Evidence limits and next gates

- Recorded remote comparisons found PP2/4/8 and v7 logits byte-identical for this
  prompt; digest and byte counts are in `split-events-summary.json`. Raw logits
  were not downloaded, so the comparison was not repeated locally.
- This does not establish independent HF equality for the full model. There is no
  same-kernel HIP-graph or vLLM baseline in this archive.
- ROCm SharedSignalPool resource-leak warnings remain visible in logs; cleanup is
  not claimed complete. Initial PP2 source identity omitted `phases.h`; later
  coverage must not be retroactively attributed to that receipt.
- No current TP/EP build has passed compilation or execution. Layout, graph,
  ownership, EP-equivalence and TP-combine regression sources are authored only.

If hardware work is explicitly reauthorized, the first gates are compact shard
payload/scale identity and capacity, VM-compiler builds for every requested degree,
EP output preservation, and measured TP numerical errors followed by 32/32 HF
greedy correctness. Only after correctness: compare PP and TP/EP with matched
instrumentation, launch-to-all-ranks-retired timing and rank overlap traces.
A provisional performance go/no-go is a repeatable **>2% median wall gain**, all
paired gains positive and median gain above twice the pair spread. Otherwise do
not promote that mode; report the loss, imbalance and communication cost. This
threshold authorizes no run. The stopped box must not be contacted.
