# DeepSeek D1-D5: go/no-go thresholds, written before the code (2026-10-13)

Work order: `/tmp/claude-dsv2-d1d5.md` (kailun, 2026-10-13); scope: `docs/SCOPE-300TOKS-2026-10-11.md` section 2.
Branch `dsv2-d1d5` off `dsv2-session`. CPU only until a box is opened. Every switch defaults to today's behaviour and
the default path stays byte-identical (checked per item). Pair rules: 5 interleaved pairs in one session, device basis
(embed -> argmax), every invocation 32/32 HF-greedy tokens and 27/27 layers. "KEEP bar" = 30 us/token
(`docs/BASELINE-STATE-2026-10-12.md`, the project's lever bar); "2% bar" = 70 us at ~3.5 ms.

Context measured 2026-10-13 (`examples/dsv2lite/results/2026-10-13-ds_inplace`): the ds_inplace arms ran the SAFE sc1
acquire (no flag); the 3.474 ms record of 2026-09-27 ran `-DETX_ACQUIRE_DOMAIN_RELAXED`. sc1 cost: +242 us in
2026-09-27b (3.987 vs 3.745), consistent with today's sc1 default 3.709 ms.

| item | done on CPU when | GPU threshold (fixed now) |
|---|---|---|
| D1 safe option in the sessions | no DeepSeek session passes the relaxed flag implicitly; relaxed only as a named, explicit arm; default build byte-identical | not a speed lever: GO = the safe build passes 32/32 + 27/27 in every invocation. The paired cost safe - relaxed is RECORDED, expected +235 +- 30 us [E] |
| D2 coherent fleet accessors (Qwen's ETX_VL_COH, ported) | the emitted megakernel has 0 per-task domain `buffer_inv sc1` on the covered events and keeps the producer writeback (counted in the generated code and the ISA); default byte-identical | GO if (a) coherent vs sc1: every pair faster and median >= 70 us; (b) coherent vs relaxed: median difference <= max(2 x relaxed-pair spread, 20 us); (c) 32/32 + 27/27 everywhere |
| D3 four K-chunks (352 rows) | the 4-chunk plan builds and verifies (events, chunk waits), default byte-identical | GO only if every pair faster than one chunk and median >= 30 us and > 2 x spread; otherwise DROP (prior: 512-row chunks 4.020 ms, fleet STATUS.md:33) |
| D4 shared experts during routing + placement check | (a) shared-expert gate_up on the two shared XCDs carries no routing; (b) every routed slice's gate_up and down are on one XCD; (c) the build REFUSES an invalid placement, and a test in the suite proves the refusal fires; default byte-identical | GO if the trace shows the shared XCDs' gate_up ready before the layer's routing completes, and every pair faster with median >= 30 us and > 2 x spread (ceiling ~0.18 ms, scope section 4) |
| D5 compiler prefetch (p7 + kernel hook) | the hook emits a prefetch body for fleet's tiles behind a switch; default byte-identical | DROP if the median is inside the spread or any pair is slower; KEEP only if every pair faster, median >= 30 us and > 2 x spread (prior: --prefetch-next 4% slower, idle prefetch no gain) |

## Scope change (2026-10-13, owner via kailun, `/tmp/claude-dsv2-priority.md`)

D1, D2 and D4 only, today. **D3 and D5 are parked** (not started): their prior evidence is negative or neutral and the
owner did not ask for them. Their thresholds above stay as written for whenever they are picked up.

## D2 on CPU (done; the GPU pairs wait for a box)

Switch: `ETX_DSV2_COH=1` (plan) + lever `coherent_loads` (tiles); session arm `coh`. Default build unchanged: plan
`09011c20`, ROCm 7.2.4 default ISA `a105cb16` (re-verified with the shim edit, which is `#ifdef FLEET_COH_LOADS` only).

Memory model. Consumer: every payload another task wrote is read with relaxed system-scope atomics through
address_space(1) (`global_load ... sc0 sc1`, past L1 and L2; the lowering Qwen's ETX_VL_COH builds use), so the tile
needs no acquire invalidate. Producer: stores stay plain and the release writes the XCD's L2 back (fleet measured that
write-back necessary, `fleet_runtime.h:247-255`): the new compiler mode `coherence="coherent_loads"` is a coherent
consumer but a PLAIN producer, so DOMAIN releases feeding it become `ETX_RELEASE_DEVICE` (buffer_wbl2) and DEVICE events
keep the last-arriver flush (no `_nf`). One hand-off has no ETX edge (attention waits for q_absorbed inside its body):
the shim's q_absorb wrapper writes back before bumping that counter. Not covered, and why that is safe: KV-cache rows
come from earlier tokens = earlier launches (host.hip launches once per token); dense_down reads h from global inside
gemv_rows, so it stays plain and keeps its acquire; q_absorb without PUB_WAIT is unreachable from the shim.

| ROCm 7.2.4, gfx942 | default | coh |
|---|---|---|
| `buffer_inv sc1` in etx_megakernel_d0 | 276 | 3 (dense_dn_0's relay-branch pair + the relay worker's, runtime-off) |
| `buffer_wbl2` in etx_megakernel_d0 | 114 | 249 |
| `global_load ... sc0 sc1` | 0 | 485 |
| VGPR / AGPR / SGPR / scratch / LDS / waves | 256 / 48 / 108 / 0 / 26,668 / 1 | 256 / 36 / 106 / 0 / 26,668 / 1 |
| plan: grids with no acquire | 0 of 218 waiting grids | 217 of 218 (all but dense_dn_0) |

(`fleet_decode_step`, fleet's own kernel compiled into the TU but never launched, keeps its 3 in both.)
Proof: `.venv/bin/python -m pytest -q tests/test_dsv2_lever_coherent_loads.py tests/test_coherence.py` (15 passed) and
`.venv/bin/python -m examples.dsv2lite.verify_lever coherent_loads` (record `results/levers/coherent_loads.json`).
`coh_check.py` refuses (exit 2, before the compiler) a coherent plan with unpatched tiles and patched tiles with a plain
plan; a test proves both refusals fire.

Unknown until a box: correctness under traffic (32/32 + 27/27), and the price. The per-task invalidate is gone, but
every task now ends in a whole-L2 write-back (249 sites, ~one per producer task instead of one per domain per DEVICE
event), and payload reads are MALL round trips. If (a) passes and (b) fails, the next step is coherent STORES too (the
full Qwen form, `_nf` arrives, no write-back), which the work order's "keep the writeback" rules out today.

## D4 on CPU (done; the GPU pairs and trace wait for a box)

Switch: `ETX_DSV2_D4=1` (plan) + `-DETX_DSV2_D4=1` (tiles); `ETX_DSV2_D4_ROWS` (default 32); session arm `d4`, pairs vs
default plus `d4trace` (criterion (a)). Default unchanged: plan `09011c20`, ROCm 7.2.4 default ISA `a105cb16`.
Implemented in ETX's port (graph.py `build_moe_layer_d4`, fleet_shim.hip `fleet_d4_*`); the pinned fleet export is
not edited, so fleet's own launcher (`fleet_launch.hip:543`, `taskgraph.py:420`) is untouched.

Placement per MoE layer (`examples/dsv2lite/d4_placement.py`):
- routed XCD k (routing slot k): router, then gate_up / down of rows [32, 1408) of slot k; gate_up's worker 0 publishes
  slot k's (id, weight) and bumps the owning shared XCD's counter (written back first);
- shared XCD 6 + x: NO router. Its gate_up folds the o_proj partials and norms by itself and waits on o_proj only
  (starts during routing), runs shared half x, then -- in-body wait on the counter, as fleet's q_c publication -- rows
  [0, 32) of slots 3x..3x+2 from the same staged x; one down for the half and the three slices (fp32 accumulator,
  one bf16 store; each slice's routing weight folded into its h).
Build-time refusal (graph.build(), so `etx compile` exits non-zero before writing a plan): a split slice (gate_up and
down on different XCDs), rows uncovered or covered twice, a row boundary not a multiple of 32 (an h cache line written
by two XCDs), a shared XCD whose first stage needs routing, and a realised grid whose domain_map puts a stage on another
XCD than the table (evaluated per task coordinate). D4 + D2 together is refused (the d4 bodies are not audited for
coherent loads). Tests: `tests/test_dsv2_d4.py` (17), including the refusals through graph.build() and `etx compile`.

| ROCm 7.2.4 | default | d4 |
|---|---|---|
| VGPR / AGPR / SGPR / scratch / LDS / waves | 256 / 48 / 108 / 0 / 26,668 / 1 | 256 / 54 / 108 / 0 / 26,672 / 1 |

Expected [E], from fleet's per-XCD stamps (gate_up 27.5 us + 5.5 prologue for 1408 rows, down 28.4 + 1.6; router 9 =
7.4 fold/norm + 1.6 logits): the shared XCDs drop the router logits and the gate_up routing, about 7.1 us of slack per
layer; each routed row moved costs them three. With 32 rows the routed XCDs end 1.3 us earlier and the shared ones still
before them: about 1.3 us/layer, 33 us/token, at the 30 us KEEP bar. 64 rows overshoots (the shared XCDs become the
stragglers, a loss). The estimate is crude, so the GPU decides. Numerics change (a routed expert's output is split over
two bf16 partials; the shared XCD's slices are summed in fp32 before one rounding), so 32/32 + 27/27 must be re-proven
on the GPU.

## GPU results 2026-10-13 (box 165.245.139.183, ROCm 7.14; runs `$ETX_CACHE/results_2026-10-13-d1d5/run{1,2}`)

Every invocation 32/32 + 27/27. Device medians: default 3.668-3.672, fleet 3.542, all 3.564 (-108, GO), relaxed 3.439
(-232, D1 cost as expected), coh 3.677 (+9: **D2 NO-GO**, (a) and (b) fail; 237 us behind relaxed), d4 3.936 (+271:
**D4 NO-GO**; criterion (a) holds, shared gate_up ready 11.5 us before routing completes in 26/26 layers).

## Next two arms, thresholds written before the code (2026-10-13, same box)

| arm | what | rule (fixed now) |
|---|---|---|
| allrelaxed | all five levers + `-DETX_ACQUIRE_DOMAIN_RELAXED` (UNSAFE acquire: a measurement of the prize, never a candidate) | no verdict; its gain over `all` is the most a safe acquire as cheap as sc0 could add to the best arm. If `all - allrelaxed` < 150 us, the acquire route alone cannot close the 231 us to 3.333 ms |
| d4r0 | D4's placement with 0 slice rows (shared XCDs: own fold + norm, no router, no slices) | splits D4's +271 us: d4r0 slower than default by >= 200 us -> the placement itself is the loss; by <= 50 us -> the slices are |

Results (run 3, 05:08-05:11): **allrelaxed -326 us vs default (spread 8)**: the unsafe acquire is worth ~218 us on top
of `all`, above the 150 us bar, so the acquire route can close most of the gap to 3.333 ms if a SAFE acquire costs
what sc0 costs. **d4r0 -62 us vs default (spread 8, every pair faster)**: D4's placement without slices is a gain;
the slices are the +271 us (the shared down, 35.7 us per task against 17.5 for a routed down, is the straggler).

| arm | what | rule (fixed now, before the code) |
|---|---|---|
| alld4r0 | all five levers + D4 placement, 0 slice rows; three-way default / all / alld4r0 | KEEP if every pair faster than `all`, median >= 30 us and > 2 x spread; 32/32 + 27/27 everywhere |

## The deciding microbenchmark for a safe cheap acquire (rule written before the run)

`bench/sync/sync_bench.hip loadscope`: the ping-pong with NO acquire invalidate and the payload read with plain (xn,
control), sc0 (xl0) or sc1 (xl1) loads on one XCD, and sc1 (cl1) or sc0 sc1 (cl01) loads across XCDs behind a
`buffer_wbl2 sc1` release. A mode is SAFE only with 0 stale words at 64 B, 8 KB and 64 KB over 2000 rounds, and only
if `xn` goes stale in the same run (the check detects staleness). If xl1 is safe: next lever "D2b" = sc1 payload loads,
no per-task invalidate, no per-task write-back (DEVICE events keep their flush). If only cl01 is safe: that is D2's
form, measured at no gain, and the safe-acquire route is closed.

Results, runs 4-5 (05:12-05:16, `$ETX_CACHE/results_2026-10-13-d1d5/run4`, `run5`):
- **alld4r0 3.506 ms vs all 3.567 ms: -61 us, spread 8, every pair faster: KEEP.** Best safe arm (285 tok/s).
  Against default (3.666): -160 us.
- **loadscope**: control xn stale at 64 B / 8 KB (detection proven; 64 KB clean by L1 eviction); xl0 (sc0 loads)
  stale like xcd; **xl1 (sc1 loads, no invalidate, same XCD) 0 stale at all sizes**, one-way 1.047 / 1.351 /
  4.721 us (xcd1, the safe buffer_inv sc1: 1.055 / 1.409 / 4.165); cl1 and cl01 (cross XCD behind buffer_wbl2) 0 stale.
  By the rule: **xl1 is safe -> D2b** (sc1 payload loads, no per-task invalidate, no per-task write-back on
  same-XCD events).

## D4 slices without shared-XCD stragglers: fixed GPU gate (2026-10-13, before code)

New arm `alld4s`: all five levers + D4 with **32 slice rows** per routed expert. Run five interleaved three-way
rounds **default / alld4r0 / alld4s**, alternating direction between rounds. KEEP only if **every pair is faster
than alld4r0**, the median device gain is **>= 30 us/token and > 2 x spread**, and **32/32 tokens + 27/27 layers**
pass in every invocation. Otherwise DROP (an incomplete run is NOT-COMPLETE). Compare with default as well, but
`alld4r0`, the measured best safe arm, is the deciding baseline.

The deciding per-layer target is <= **8 us** extra work on each shared XCD with slices on: shared down task body
<= **25 us**, shared gate_up <= **36.7 us** (the measured 33.7 us + 3 us), and shared down last completion no later
than routed down. These are GPU targets, not CPU timing claims. The measured 32-row build shortened routed completion
by **7.3 us/layer** (72.9 -> 65.6 us after o_proj), but shared down completed at 77.1 us with a 35.7 us mean body.
The recoverable routed improvement is a ceiling of about **190 us/token [E]** (26 MoE layers x 7.3 us), not a prediction.
For the 3 slices, gate/up plus down reads 1,179,648 weight bytes per shared XCD/layer (bf16, hidden 2048, rows 32);
the two shared XCDs' combined 2,359,296 bytes have a **0.55 us [E]** whole-device bandwidth floor at 4.3 TB/s.
Placement, latency and dependency constraints make that optimistic floor unattainable in general.

CPU-only work: reduce the slices' serial staging and routing overhead, retain the placement checks and rows=0 path,
compile with ROCm 7.2.4, record resources and source/ISA round-trip counts. No GPU is provisioned by this change.
If the gate passes, keep slices; if it fails, retain `alld4r0`. The task-body targets and shared/routed completion
ordering need a GPU trace in addition to the paired verdict.

### D4s implementation and CPU evidence

The optimized path is restricted to `rows=32, spx=3`; other slice sizes retain their old implementation. The graph,
placement table and routing publication are unchanged. Shared gate_up still finishes its shared half before it
waits for routing. Wave 0 broadcasts the wait result, then lanes 0..2 each load one `(id, weight)` pair coherently
into the existing `Scratch.ids/ws`; one workgroup barrier publishes all six words. The 96 slice rows are flattened
across the shared XCD's 37 workers and four waves, using Fleet's `wave_dot<2,4>` and unchanged BF16/SiLU epilogue.
The shared down reuses that routing cache: the compiled queues place shgu immediately before sdown on the same
worker, with no intervening task. D4 already refuses multiple requests and uncached routing. A lowered-plan test
now checks this LDS lifetime, in addition to the existing placement refusals.

Shared down stages all 96 weighted h values in the otherwise-unused `Scratch.row` before its shared-half GEMV;
`stage_vector`'s existing barrier publishes them. It does not overwrite `vec` for each slice. After the shared-half
GEMV, one barrier makes `part` visible to the new row owners. Four lanes now compute each K=32 row, with separate
slice reductions and exactly `((shared + slice0) + slice1) + slice2` before BF16 rounding. This uses all 64 lanes for
16 independent output rows instead of four useful lanes in each 32-lane half-wave. Scaling stays on h, before the
dot. Adding +0 before the shortened shuffle tree preserves the original inactive lanes' signed-zero normalization.
No weight-prefetch overlap with the shared GEMV is claimed or required by this implementation.

Source-level count (per shared task, default 32 rows; a group below is a dependency phase, **not a measured memory
round trip**, and the unchanged shared-half stream is excluded): old sdown had three serial
`metadata -> h staging -> matrix stream` chains, **9 groups**, **6 per-slice barriers**, and **1,536 coherent scalar
metadata lane-loads** (3 slices x 2 words x 256 lanes). New sdown has **0 metadata global reads**, **1 batched h group**
before the shared stream, **1 matrix group** in source after it, and **1 added barrier** after the shared stream.
The three slice matrices have independent addresses, but the compiler may split their issue; the ISA counts below
are the actual static evidence. Old shgu had three serial metadata/GEMV chains; new shgu has one six-word broadcast
phase and one wave-distributed slice batch. Only **6 metadata scalar words per shgu task** are read globally.

Prior evidence was checked in Fleet's pinned `docs/STATUS.md` / `results/` and ETX design section 15: broad prefetch
regressed (4.69488 -> 4.77216 ms in that older Fleet session), and one versus three waves/SIMD measured 38.8 us for
both in the occupancy probe. This change is confined to the short slices; those old results are not timing evidence
for D4s. The original D4 trace was re-read using `ledger.load_etx`: median across layers 3..25, shared gate_up mean
body **33.660 us**, shared down ready **39.030 us**, last done **77.110 us**, mean body **35.734 us**; routed down last
done **65.570 us**, mean body **17.514 us**. The original default trace's down last done is **72.870 us**. These are
old GPU observations, not measurements of the new implementation.

Final ROCm 7.2.4 gfx942 static evidence (`examples/dsv2lite/results/levers/d4s.json`; device assembly + host device
pass + host syntax pass all exit 0, offline container):

| build | VGPR | AGPR | SGPR | scratch B/lane | LDS B | waves/SIMD |
|---|---:|---:|---:|---:|---:|---:|
| default | 256 | 48 | 108 | 0 | 26,668 | 1 |
| alld4r0 before and after | 256 | 90 | 108 | 0 | 26,668 | 1 |
| old all + D4, 32 rows | 256 | 89 | 108 | 0 | 26,672 | 1 |
| optimized D4, 32 rows, no levers | 256 | 65 | 108 | 0 | 26,672 | 1 |
| alld4s | 256 | 104 | 108 | 0 | 26,672 | 1 |

Default plan SHA256 is `09011c207b1e92febf29a2c2efc2d944ef9ad7ac8a0803633579c8444d821f8a`; default ISA is
`a105cb16a414ee0f9f697c9de13205419d09674f5c69f6ee804cf36942377724`. **alld4r0's entire ISA is byte-identical**
before/after, SHA256 `f26a7ff3bdf99aa5d0f7ae1c7a50450bb5bfb801b089a6b3217455f3fb396bf5`; its plan is identical too.

In `build/d4s-final-slices/<all-five-levers>/mk.s`, metadata uses **two vector coherent-load instructions**, each
followed by a wait, with three active lanes (lines 112465 and 112471), then the workgroup barrier at 112478. An initial
thread-0-only implementation compiled to six serialized loads and was replaced before the final build. In shared
down the expert-base pointer fetch is at 108655; the three slice matrix-load sites are 108660, 108665 and 108668,
with a full wait at 108666: **two matrix issue batches**, not one latency round trip. The new all-lever build uses
15 more AGPRs than the old 32-row build; scratch, LDS and reported occupancy are unchanged. These counts do not
predict the GPU body time.

Proof commands (from this worktree, pinned export `~/code/etx-cache/fleet-3fb67a6`):

```sh
bash examples/dsv2lite/rocm_build_check.sh ~/code/etx-cache/fleet-3fb67a6 default build/d4s-final-default
ETX_DSV2_D4=1 ETX_DSV2_D4_ROWS=0 ETX_CFLAGS=-DETX_DSV2_D4=1 bash examples/dsv2lite/rocm_build_check.sh ~/code/etx-cache/fleet-3fb67a6 gemv_no_terminal_all+norm_tree_all+argmax_tree+topk_rank+attn_prologue_live build/d4s-final-r0
ETX_DSV2_D4=1 ETX_CFLAGS=-DETX_DSV2_D4=1 bash examples/dsv2lite/rocm_build_check.sh ~/code/etx-cache/fleet-3fb67a6 default build/d4s-final-d4
ETX_DSV2_D4=1 ETX_CFLAGS=-DETX_DSV2_D4=1 bash examples/dsv2lite/rocm_build_check.sh ~/code/etx-cache/fleet-3fb67a6 gemv_no_terminal_all+norm_tree_all+argmax_tree+topk_rank+attn_prologue_live build/d4s-final-slices
.venv/bin/python -m pytest -q tests/test_dsv2_*.py tests/test_coherence.py
.venv/bin/python -m etx.tools.gpu_jobs examples/sessions/ds_d4s.py --gpus 0 --dry-run
```

Required suite: **133 passed in 35.53 s, exit 0**. Tests include placement refusal, lowered same-worker queue
adjacency, exact row cover, 4,096 three-slice reduction cases with signed zero/cancellation, BF16 epilogue, KEEP
threshold boundaries, three-way order, correctness failure in any arm, interruption/truncation, and refusal to reuse
a completed report. Dry run: **7 jobs, exit 0**, all generated job commands pass `bash -n`; three builds and preflight
are hard dependencies of the sole paired GPU job. Full logs, per-invocation deadlines and silence probes use the
existing session runner. `ds_d4s.py` is prepared, not run, with a 1,800-second cap and no provisioning/download jobs.

Unknown: GPU correctness (32/32 + 27/27), the <=25 us shared-down / <=36.7 us shared-gate_up targets, shared-versus-routed
completion order, and the paired >=30 us gain against alld4r0. No new performance result, KEEP, or latency estimate
is claimed. GPU hours **0**, GPU cost **$0**. The initial metadata serialization was caught locally; a CPU test's
nonassociativity witness was corrected from 2^24 to 2^25 before the passing full suite. A trace lookup initially
used run2 for the default trace; the reused default trace is in run1. No GPU session failed or was provisioned.
