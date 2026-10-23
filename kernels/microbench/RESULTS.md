# First host-GPU timing round

Date: 2026-10-24. Card: **NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB**;
architecture **sm_89** (compute capability 8.9). These are not H100 numbers.
Host compiler: `/opt/cuda/bin/nvcc`, CUDA 13.4.92, `-O3 -std=c++17 -arch=sm_89`.
Driver: 615.71.09. No Docker was used.

Command, from the repository root:

```bash
bash kernels/microbench/run_bench.sh
```

The script compiles and runs each driver serially, with no overlapping benchmark
processes. Desktop graphics processes remained active on the card; clocks were
not locked. Each driver checks synthetic outputs against expected results before
20 warm-up iterations and 200 measured iterations. Each measured iteration has
its own CUDA start/stop events and stop-event synchronization. p50 is the mean of
the two middle sorted samples. Allocation, initialization, correctness copies,
and compilation are outside the timing window.

| Driver / timed launch sequence | p50 (µs) | min (µs) | max (µs) | Effective GB/s |
|---|---:|---:|---:|---:|
| argmax: partial → final | 6.336 | 5.120 | 7.296 | 48.122 |
| topk_row: top-16 | 57.344 | 56.320 | 59.296 | N/A |
| spec_round: splice_draft → splice_verify → spec_count → spec_lines → ones | 13.312 | 12.288 | 18.432 | N/A |
| markov_rows: embedding → partial argmax → final argmax | 55.296 | 55.232 | 58.240 | 185.935 |

All four output checks passed. Each row measures the complete listed launch
sequence, not a separate time for each entry point. Event spans include gaps
between launches in a sequence; no CUDA graph is used. This is a synthetic
microbenchmark, not an end-to-end model or ETX fusion measurement.

## Shapes and launches

- **argmax:** one bf16 logits row, vocabulary 151,936, 64 partial chunks.
  Partial grid `(1,64,1)`, block `(1024,1,1)`; final grid `(1,1,1)`,
  block `(64,1,1)`. Patterned logits with equal maxima at indices 17 and
  151,935 verify smallest-index tie breaking.
- **topk_row:** one bf16 logits row, vocabulary and row stride 151,936,
  k=16. Grid `(1,1,1)`, block `(1024,1,1)`. Logits are
  `bf16(((i*73)%1024)/32.0)`; all 16 IDs and values are checked against
  a CPU sort by descending value and ascending index.
- **spec_round:** 32 sequences, block length 7, draft stride 8, 64 line
  rows, `seqs_max=32`. The first four launches each use grid `(32,1,1)`,
  block `(64,1,1)`; ones uses grid `(1,1,1)`, block `(64,1,1)`.
  Anchors are 42, draft/verification tokens 17, mask 99, staged lines 123.
  Full acceptance (count 7), every splice output, advanced line entry, and
  constant-one output are checked. Verification tokens are preinitialized;
  this sequence measures glue entry points, not an actual draft/verify model.
- **markov_rows:** 32 sequences, vocabulary 151,936, logits row stride
  `7*151936`, ID/output stride 7, hidden size 4,096. Embedding grid
  `(32,1,1)`, block `(256,1,1)`; partial grid `(32,64,1)`, block
  `(1024,1,1)`; final grid `(32,1,1)`, block `(64,1,1)`.
  Active logits rows use the argmax pattern and tied peaks above. IDs select
  embedding row zero, filled with ones. Gather outputs, argmax results, and
  untouched output-stride padding are checked.

## Traffic interpretation and limits

Effective GB/s is **logical bytes divided by median event time**, using decimal
GB. Argmax counts 304,904 bytes: logits reads, partial writes and reads, and final
output. Markov counts 10,281,472 bytes: active logits reads, partial writes and
reads, embedding table reads/output writes, ID reads, and final outputs.
These are apparent scan/gather rates, not measured DRAM traffic or proof that the
sequences are bandwidth bound. Buffers are reused across iterations; cache reuse
is expected, particularly for the small logits row and shared embedding row.
Top-k includes substantial selection/reduction work and spec_round consists of
small glue launches, so a bandwidth rate is not assigned to them. No parameter
sweep, optimality claim, cross-card comparison, or performance threshold is
established by this first timing round.

## Raw timing output

```text
argmax p50_us=6.336 min_us=5.120 max_us=7.296 rows=1 n=151936 grid=(1,64)+(1) block=1024+64 effective_GBps=48.122 check=OK
topk_row p50_us=57.344 min_us=56.320 max_us=59.296 rows=1 n=151936 k=16 grid=(1) block=1024 effective_GBps=N/A check=OK
spec_round p50_us=13.312 min_us=12.288 max_us=18.432 seqs=32 block_len=7 rows=64 dstride=8 grid=(32)x4+(1) block=64x5 effective_GBps=N/A check=OK
markov_rows p50_us=55.296 min_us=55.232 max_us=58.240 rows=32 n=151936 hidden=4096 stride=7 grid=(32)+(32,64)+(32) block=256+1024+64 effective_GBps=185.935 check=OK
```
