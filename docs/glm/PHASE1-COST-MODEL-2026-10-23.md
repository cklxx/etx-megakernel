# Phase 1 cost model: EP8, headTP+EP8, TP8 at batch 1 — 2026-10-23

**Every number in this file is an estimate [E] unless marked [M].** This is a model built from data we
already own; it is not a measurement, it does not rank the arms, and it does not replace the Phase 1
runs. Those remain: full-model HF greedy 32/32 per arm, then T10 paired timing of the arms against each
other ([measurement plan](PARALLEL-MEASUREMENT-PLAN-2026-10-22.md)). Its use is to pre-register what
the runs should show, so a surprise is recognised as one.

Reproduce: `python -m examples.glm53.cost_model` (weights-free; reads the pinned config, derives the three
layouts, builds the three graphs; about 7 s on a laptop).

## Inputs

- **Per-phase time [M, one step]:** PP2 build `le2`, MI325X, batch 1, decode position 4, host wall
  28.353 ms; grid spans summed over 45 layers ([FFN-PARALLEL](FFN-PARALLEL-2026-10-22.md) "What the trace
  says"; `results/o55-events-20261022/le2run.log`): up 7.4225, p1 5.3040, down 4.5374, o 2.7099,
  KDA/attention heads 1.4838, p2 1.4609, router 0.9952 ms. The rest, 4.440 ms, is hc/hcf/embedding/head
  grids, scheduling and host launch/retire ("residual"). The compiler that built `le2` is not recorded in
  the archive (`build_le2.log` has no toolchain line) [U]; the Phase 1 arms are built with ROCm 7.2.4, so
  any trace-to-arm difference may include a compiler difference.
- **Per-tile shape [M, one layer]:** `l10_up` body 141 µs of a 175 µs span; `l10_down` body 82 µs of
  114 µs (`le2run.log:578-579`). 512 tasks start within 5-13 µs of each other: each phase is one tile deep.
- **Per-phase bytes:** every tensor each rank loads, from `model_build.production_compile_spec()` and
  `parallel_layout.derive()` for the arm, payload plus FP8 128×128 block scales; slot-to-phase map from
  `examples/glm53/phases.h`.
- **Cross-device hops and peer bytes:** counted on each arm's planned graph (`model.build_from`): the
  longest chain of grid dependencies, counting each edge whose producer runs on another GPU; peer bytes =
  non-weight tensors a grid reads that another GPU wrote, per request.

## Assumptions (each is an estimate or a choice, not a measurement)

1. The position-4 PP2 step is representative; the decode window (positions 1023-1054) ran about 8 % slower
   in the same build [M], and KV/state reads grow with position. Totals below are for position 4.
2. A phase's span is one tile body plus 33 µs of scheduling per grid (from the l10 up/down spans), for every
   phase. Where the trace phase is shorter than its grid count × 33 µs, the whole phase is treated as overhead.
3. Routed experts are selected uniformly and independently: 8 of 288 per MoE layer, without replacement.
4. EP places expert e on rank e mod 8 and the shared expert and dense FFN on rank 0 (layout); the critical
   rank holds E[max] = **2.778** expert-equivalents of 9 per MoE layer (exact enumeration).
5. TP8 splits every FFN (dense, shared, routed) evenly; every rank reads 1/8 of each selected expert.
6. headTP shards KDA Q/K/V/β and MLA Q-B/KV-B/O by head; MLA Q-A/KV-A and the indexer stay replicated on
   every rank (layout); router, norms, hc and head stay on rank 0 in all arms.
7. **Bytes-proportional column:** a sharded phase's tile body scales with the critical rank's bytes. This
   is the optimistic bound; the trace contradicts it for one-tile-deep phases (next item).
8. **Tile-depth column:** a phase whose tiles keep their rows and K (up, p1, p2, heads) stays one tile deep,
   so only contention relief applies, 0-15 % of the body (dense 342-tile up grids ran 10-15 % faster than
   512-tile ones [M]). A phase whose tiles keep their rows but whose reduction K shrinks (down under EP/TP,
   o under head TP) scales its body with the critical rank's bytes.
9. Each added critical-chain hop costs 10-20 µs (one SYSTEM publication and wait). Proxies only: 5.46 µs
   intra-GPU barrier on this box, 10.6 µs remote flag on 2× MI300X [M, other conditions]; never measured on
   the 8× MI325X fabric.
10. Peer reads move at 50-350 GB/s effective (one link to all seven in parallel): unknown, bounded.
11. Phases do not overlap each other across ranks beyond what the graph allows; no extra contention from
    peer traffic; KV/state bytes, spills and the per-expert down helper's efficiency are not modelled.

## Bytes per token, PP2 basis [E from the spec]

| phase | PP2 bytes/token (GB) | PP2 trace ms |
|---|---:|---:|
| up | 6.645 | 7.4225 |
| p1 | 7.053 | 5.3040 |
| down | 3.323 | 4.5374 |
| o | 3.020 | 2.7099 |
| heads | 0.150 | 1.4838 |
| p2 | 0.784 | 1.4609 |
| router | 0.100 | 0.9952 |
| hc | 0.071 | not itemized |
| head | 1.269 | not itemized |
| total | 22.415 | 23.9137 |

Weights read per token: **22.4 GB** [E], plus KDA/MLA state (position-dependent, not in the table). The
traced phases move these bytes at 0.7-1.3 TB/s, against 4.24 TB/s streaming measured on this box [M]:
the phases are not bandwidth bound, which is why assumption 8 matters more than the byte split.

## EP8 (`parallel=2, devices=8`)

cross-device hops on the critical chain 84, peer bytes/request 48526464 (48.53 MB)

| phase | rank 0 GB (expected) | each of ranks 1-7 GB (expected) | critical rank GB | critical / PP2 bytes |
|---|---:|---:|---:|---:|
| up | 1.712 | 0.705 | 2.260 | 0.340 |
| p1 | 7.053 | 0.000 | 7.053 | 1.000 |
| down | 0.856 | 0.352 | 1.130 | 0.340 |
| o | 3.020 | 0.000 | 3.020 | 1.000 |
| heads | 0.150 | 0.000 | 0.150 | 1.000 |
| p2 | 0.784 | 0.000 | 0.784 | 1.000 |
| router | 0.100 | 0.000 | 0.100 | 1.000 |

| phase | PP2 trace ms | critical-rank bytes / PP2 | bytes-proportional ms [E] | tile-depth ms [E] |
|---|---:|---:|---:|---:|
| up | 7.42 | 0.340 | 3.50 | 6.53-7.42 |
| p1 | 5.30 | 1.000 | 5.30 | 5.30 |
| down | 4.54 | 0.340 | 2.52 | 2.52 |
| o | 2.71 | 1.000 | 2.71 | 2.71 |
| heads | 1.48 | 1.000 | 1.48 | 1.48 |
| p2 | 1.46 | 1.000 | 1.46 | 1.46 |
| router | 1.00 | 1.000 | 1.00 | 1.00 |
| phases | 23.91 | | 17.98 | 21.01-21.90 |
| residual (hc, hcf, embed, head, host) | 4.44 | 1.000 | 4.44 | 4.44 |
| communication: 82 added critical hops, 48.5 MB peer reads | 0.00 | | 0.96-2.61 | 0.96-2.61 |
| **total, position 4** | **28.35** | | **23.4-25.0** | **26.4-28.9** |

## headTP+EP8 (`parallel=2, attention_tp=1, devices=8`)

cross-device hops on the critical chain 174, peer bytes/request 58848384 (58.85 MB)

| phase | rank 0 GB (expected) | each of ranks 1-7 GB (expected) | critical rank GB | critical / PP2 bytes |
|---|---:|---:|---:|---:|
| up | 1.712 | 0.705 | 2.260 | 0.340 |
| p1 | 1.048 | 1.047 | 1.048 | 0.149 |
| down | 0.856 | 0.352 | 1.130 | 0.340 |
| o | 0.378 | 0.378 | 0.378 | 0.125 |
| heads | 0.019 | 0.019 | 0.019 | 0.125 |
| p2 | 0.219 | 0.219 | 0.219 | 0.279 |
| router | 0.100 | 0.000 | 0.100 | 1.000 |

| phase | PP2 trace ms | critical-rank bytes / PP2 | bytes-proportional ms [E] | tile-depth ms [E] |
|---|---:|---:|---:|---:|
| up | 7.42 | 0.340 | 3.50 | 6.53-7.42 |
| p1 | 5.30 | 0.149 | 2.05 | 4.73-5.30 |
| down | 4.54 | 0.340 | 2.52 | 2.52 |
| o | 2.71 | 0.125 | 1.64 | 1.64 |
| heads | 1.48 | 0.125 | 1.48 | 1.48 |
| p2 | 1.46 | 0.279 | 0.67 | 1.30-1.46 |
| router | 1.00 | 1.000 | 1.00 | 1.00 |
| phases | 23.91 | | 12.87 | 19.20-20.83 |
| residual (hc, hcf, embed, head, host) | 4.44 | 1.000 | 4.44 | 4.44 |
| communication: 172 added critical hops, 58.8 MB peer reads | 0.00 | | 1.89-4.62 | 1.89-4.62 |
| **total, position 4** | **28.35** | | **19.2-21.9** | **25.5-29.9** |

## TP8 (`parallel=1, devices=8`)

cross-device hops on the critical chain 90, peer bytes/request 51992640 (51.99 MB)

| phase | rank 0 GB (expected) | each of ranks 1-7 GB (expected) | critical rank GB | critical / PP2 bytes |
|---|---:|---:|---:|---:|
| up | 0.831 | 0.831 | 0.831 | 0.125 |
| p1 | 7.053 | 0.000 | 7.053 | 1.000 |
| down | 0.415 | 0.415 | 0.415 | 0.125 |
| o | 3.020 | 0.000 | 3.020 | 1.000 |
| heads | 0.150 | 0.000 | 0.150 | 1.000 |
| p2 | 0.784 | 0.000 | 0.784 | 1.000 |
| router | 0.100 | 0.000 | 0.100 | 1.000 |

| phase | PP2 trace ms | critical-rank bytes / PP2 | bytes-proportional ms [E] | tile-depth ms [E] |
|---|---:|---:|---:|---:|
| up | 7.42 | 0.125 | 2.23 | 6.53-7.42 |
| p1 | 5.30 | 1.000 | 5.30 | 5.30 |
| down | 4.54 | 0.125 | 1.87 | 1.87 |
| o | 2.71 | 1.000 | 2.71 | 2.71 |
| heads | 1.48 | 1.000 | 1.48 | 1.48 |
| p2 | 1.46 | 1.000 | 1.46 | 1.46 |
| router | 1.00 | 1.000 | 1.00 | 1.00 |
| phases | 23.91 | | 16.05 | 20.35-21.24 |
| residual (hc, hcf, embed, head, host) | 4.44 | 1.000 | 4.44 | 4.44 |
| communication: 88 added critical hops, 52.0 MB peer reads | 0.00 | | 1.03-2.80 | 1.03-2.80 |
| **total, position 4** | **28.35** | | **21.5-23.3** | **25.8-28.5** |

## Reading [E]

- **Tile-depth estimates (assumption 8), position 4:** EP8 **26.4-28.9 ms**, TP8 **25.8-28.5 ms**,
  headTP+EP8 **25.5-29.9 ms**, against 28.35 ms for the traced PP2 step. The three ranges overlap almost
  entirely: a tie under the T10 bar is a plausible Phase 1 outcome.
- **Bytes-proportional bounds (assumption 7):** 23.4-25.0, 21.5-23.3 and 19.2-21.9 ms. Reaching these
  requires the up phase (and, for headTP, p1/p2/heads) to shrink with bytes, which a one-tile-deep phase
  does not do with the current tile geometry.
- **Where the gain is:** down in every arm (−2.0 ms EP, −2.7 ms TP), o under head TP (−1.1 ms).
  **Where it goes:** 82-172 added cross-GPU hops per token (+0.8-3.4 ms) and 49-59 MB of peer reads
  (+0.1-1.2 ms).
- **Floor:** the phases no arm shards here (p1, o, heads, p2, router for EP/TP) plus the residual sum to
  **16.4 ms** [E], above the goal's "about 12 ms", which counts the phase bodies only (assumption: the
  residual stays in host wall).
- The legacy PP2 29.455 ms/token is a reference only; the 28.35 ms position-4 step above is the model's
  input basis, not a comparator.

## What a Phase 1 trace would confirm or falsify

- `up` body on an EP8 or TP8 rank: ≥ 120 µs per sparse layer confirms assumption 8; ≤ 60 µs falsifies it
  and moves the arm toward the bytes-proportional column.
- `down` body: EP8 critical rank 25-45 µs, TP8 10-20 µs (assumption 8 with the 82 µs PP2 body).
- Hop cost: (measured total − modelled phases − residual) / added hops; outside 5-30 µs means assumption 9
  is wrong.
