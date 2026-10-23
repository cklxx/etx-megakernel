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

## Remaining sm_89 drivers

Target card: **NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB**, **sm_89**;
compiler: `/opt/cuda/bin/nvcc`, **CUDA 13.4.92**, `-O3 -std=c++17 -arch=sm_89`.
All twelve drivers compile with this compiler. `BENCH_COMPILE_ONLY=1 bash
kernels/microbench/run_bench.sh` compiles serially without executing GPU code.
This compile-only check does not establish runtime correctness or timings.

The eight added drivers use the same output-check, 20-warm-up, 200-sample CUDA
event protocol as the first round. Each file's row measures its complete launch
sequence. New timing numbers await authorization to run GPU benchmarks; none
have been estimated or copied from another worktree.

| Driver / timed launch sequence | p50 (µs) | min (µs) | max (µs) | Grid | Block | Effective GB/s |
|---|---:|---:|---:|---|---|---:|
| copy_rows: strided copy → last row → own rows | pending | pending | pending | (32)+(1)+(4) | 256×3 | pending |
| dflash_conv: grouped convolution | pending | pending | pending | (20) | 256 | N/A |
| dflash_select: 7-step candidate walk | pending | pending | pending | (4) | 256 | N/A |
| gdn_advance: mask row zero → shift conv history | pending | pending | pending | (32)+(4,1) | 256×2 | N/A |
| line_copy: gather → scatter | pending | pending | pending | (16,4)×2 | 256×2 | pending |
| peer_barrier: single-rank self-peer barrier | pending | pending | pending | (1) | 32 | N/A |
| peer_collective: single-rank local allgather copy | pending | pending | pending | (16) | 256 | pending |
| weight_prep: weight+1 → cast → RoPE → four fills → iota | pending | pending | pending | (17)×2+(32)+(17)×5 | 256×2+64+256×5 | N/A |

### Shapes, checks and interpretation

- **copy_rows:** 32 bf16 rows, width 4,096, input/output strides 4,112/4,104;
  four own rows start at row 30, testing clamping to the last row. Checks every
  output and untouched stride padding. Logical traffic is 606,480 bytes per
  sequence (row reads/writes plus four plan-index reads).
- **dflash_conv:** 16 tokens, 5,120 channels, 320 groups, side 1. Nonzero
  coefficients and delta check both taps and the reset at token 8 against the
  original input. The timed operation updates the same buffer repeatedly;
  initialization/restoration is outside timing. No bandwidth rate is assigned.
- **dflash_select:** four sequences of seven steps, rank 256, 16 candidates,
  eight rows per sequence; 16,384 dynamic shared-memory bytes. Alternating
  predecessor signs force choices 15,0,15,0,15,0,15, exercising dependence on
  the previously selected candidate. All token IDs are checked.
- **gdn_advance:** four sequences of eight rows, 257 bf16 columns, 4,096-byte
  history tokens, eight tokens per line. Acceptance counts 1–4 cover no shift,
  overlapping shifts by one/two tokens, and a null line. Checks all masked
  outputs and the entire state against a snapshot-based host reference. Timed
  shifts repeatedly mutate the same state; restoration is outside timing.
- **line_copy:** four sequences, indices 0,3,1,2, 65,536 bytes per sequence,
  offset 16, line stride 65,568. Separate source and destination state buffers
  check null-line handling, permutation, data and untouched guards. Logical
  traffic is 786,464 bytes (three active gathers and scatters plus index reads).
- **peer_barrier:** one rank pointing to its own flag, timeout 1 ms. Checks
  flags, epoch and error before timing and the final epoch/error afterward.
  This does not exercise missing-peer timeout or inter-device communication.
- **peer_collective:** one rank, 16 rows of 8,192 bytes, 16 CTAs, timeout 1 ms.
  Checks all output words, errors and epoch advancement. Logical local-copy
  traffic is 262,144 bytes. With no remote rank, send/receive loops are skipped;
  this is not a fabric-bandwidth or multi-rank protocol measurement.
- **weight_prep:** 4,099 elements (including a partial final block); RoPE is
  32×64 with base 10,000 and separate cosine/sine buffers. Checks weight+1,
  cast, all four fill types, stride-two iota including untouched padding, and
  RoPE against double-precision host trigonometry (absolute tolerance 0.004
  for bf16 output). All eight entry points are timed together.

### Blocked on sm_90 (Hopper ISA)

No drivers are added for `attn_prep`, `gdn_decode`, `gemma_rms_norm`,
`peer_allreduce`, `sigmoid_mul`, or `silu_mul`. These kernels require Hopper ISA
and are blocked on `sm_90`; the sm_89 card cannot run them.
