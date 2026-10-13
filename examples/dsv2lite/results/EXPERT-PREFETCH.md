# Can prediction improve expert-weight prefetch? (DeepSeek-V2-Lite, ETX port, 2026-10-13)

Work order: `/tmp/claude-expert-prefetch.md` (kailun). CPU only: no GPU ran for this document and no code was written.
Labels: **[M]** measured, with its source; **[E]** arithmetic on measured numbers; **[U]** unknown (contributes nothing).
Traces: `examples/dsv2lite/results/2026-10-13-ds_inplace/trace_{default,all}.trace.bin` plus their STAMPS2 files
(`$ETX_CACHE/results_2026-10-13-ds_inplace/trace_*.stamps.bin`, hashes in that session's `SHA256SUMS`), plan
`09011c20` (`PLAN.sha256`). Ledger: `python -m examples.dsv2lite.ledger --etx <plan.json> <trace> <stamps>`.

## Verdict

**NO-GO. Not worth a session, and no code.** A perfect predictor's best mechanism is worth at most **97.8 us/token**
(about 272 us under an unphysical "L2 hits are free" bound). In this expert layout a prefetch shortens a layer only
when **all six** routed experts were predicted: one XCD carries one expert, and the slowest XCD ends the phase. To
reach the 30 us bar, a 6-expert predictor would need to name t+1's exact set in **>= 30.7 %** of (token, layer) pairs.

The only routing data we have says that is far away. Expert popularity across contexts is near-uniform: per-expert
recall is 10.4 % against 9.4 % random. Same-sequence persistence (token t to t+1) was never captured, so it stays
**[U]**. The prior measurements on this GPU are all negative: `--prefetch-next` 4 % slower, idle-worker prefetch no
gain, cached working set 18 % slower.

**Mechanism.** Prediction could only help by **hiding the router-to-body latency** (and filling the HBM that idles
during the router window). Smoothing the demand peak has no room, because the expert stream is the peak.

## Step 1 — the predictor's ceiling from existing data

**What exists.** No per-token routing IDs were ever archived. `routing_stats.py` has a capture mode (`--dump ids.npz`)
that was never run: no `.npz` exists under `~/code/etx`, `~/code/etx-cache` or `~/code/fleet-mi300x`, and no doc
records its output. The one DeepSeek routing record is `routing_overlap.py`'s run in 2026-09-27b
(`$ETX_CACHE/data-2026-09-27/results_2026-09-27b/routing.json`, log `results/2026-09-27b/routing2.log`). Its setup is
4 sequences x 32 decode steps x 26 MoE layers, top-6 of 64. It records **distinct experts across concurrent
sequences at the same step**, which is cross-context overlap, not token-to-token overlap.

| # | Quantity | Value | Basis |
|---|---|---|---|
| 1 | Persistence: P(set at t+1 == set at t), same sequence and layer | **[U]**. Never captured. Uniform routing gives 1 / C(64,6) = 1.3e-8 | [E] |
| 2 | Recall of the best-k predictor, k = 6 / 8 / 12 (same sequence) | **[U]** for persistence predictors | — |
| 2' | Recall of a popularity predictor (what another context selected at the same step and layer) | **10.4 %** overall (0.622 of 6 experts shared, against 0.5625 uniform); best layer 11: 16.5 % (0.99 of 6); worst layer 1: 5.2 % (0.31) | [M] `routing.json` `per_layer`, overlap = 12 - distinct(2 seq) |
| 3 | Random baseline, recall 6/64 / 8/64 / 12/64 | 9.4 % / 12.5 % / 18.8 %. P(all six inside a random k-set): 1.3e-8 / 3.7e-7 / 1.2e-5 | [E] |
| 4a | Decay with layer depth (cross-context overlap of 6, layers 1-26) | 0.31 0.57 0.66 0.38 0.60 0.59 0.57 0.90 0.87 0.73 0.99 0.71 0.75 0.67 0.80 0.75 0.45 0.66 0.47 0.47 0.63 0.48 0.48 0.51 0.49 0.68. Skew rises in layers 8-16 (0.67-0.99) and falls back to near uniform (0.45-0.51) after layer 17. Concurrent 4-sequence distinct counts reach 19.28 (layer 8) against 20.83 uniform | [M] `routing.json` |
| 4b | Decay with token distance | **[U]**. Needs per-token IDs | — |

**5. Cheap signals, and when the code can read them** (pinned fleet `src/kernels/fleet_kernel.hip`, ETX
`examples/dsv2lite/`):

- **Router partial logits.** `gemv_rows(w.layer_router…)` at `fleet_kernel.hip:433` splits the 64 logits over an XCD's
  37 workers, about 2 each. One worker's partial is 2 of 64 experts, so it says nothing about the top-6. The full 64
  are readable only after the router task's event (`graph.py` `E_norm_L`).
- **Normed input.** `x_norm` is published at `fleet_kernel.hip:428-430`, after `rms_norm_lds` (`:425`). That is
  before the logits, but the router GEMV costs 0.06 us of bytes per layer, so the logits themselves are not what is
  slow.
  - Exact-ish signal, not a prediction: top-k order is invariant to the positive RMS scale. So the IDs could come from
    the gamma-scaled fold before the norm's reduction, saving the norm's latency only (about 2.5 us per layer).
  - It is not bit-exact, because of the bf16 rounding of the normed vector. Recorded, not pursued.
- **The previous layer's routing.** It is computed in gate_up's prologue (`fleet_kernel.hip:830`) into LDS `s.ids` of
  each worker. It is published to global only with `FLAG_TOPK_PUBLISH` (`:443`, off in the ETX graph) or by D4's
  `fleet_d4_routed_gate_up` (`fleet_shim.hip`, D4 build only). It is available about one layer (about 139 us) early.
  Its predictive power is the cross-layer correlation, which has not been measured on this model.
- **The previous token's routing at the same layer.** It is not persisted: `s.ids` is LDS, overwritten every layer,
  and each token is a separate launch (`host.hip:197`). Reading it needs a new global write.
- **The shared expert's output.** It is computed in the same layer's down phase, concurrently with the routed
  experts, so it is available only after routing. The previous layer's shared output is already inside the residual
  stream the router reads. Not an earlier signal.

## Step 2 — the account

**Bytes.**

| | Value | Basis |
|---|---|---|
| Routed expert | **17,301,504 B** (8,650,752 elems); gate_up 11,534,336 + down 5,767,168 | [M] `2026-10-13-ds_inplace/prep_pack.log:34` (the pinned pack); split by fleet's layout `expert.h:52-58` |
| Shared expert (2 halves) | 34,603,008 B, static: no prediction needed | [M] same stride; `expert.h:3-5` |
| Per MoE layer | routed 6 x 17.30 = 103.8 MB; all experts 138.4 MB | [E] |
| Per token (26 MoE layers) | routed 2.699 GB; all experts 3.599 GB | [E] |

**Time.** Per MoE layer, mean of layers 2-25 [M], from the ledger of the stamped traces:

| trace | token | router | gate_up staging (routing) + prologue | gate_up floor + stream | down floor + stream | tail (gate_up / down) |
|---|---|---|---|---|---|---|
| default (sc1) | 3959.8 us | 14.73 | 4.96 + 0.95 = **5.91** | 21.46 + 6.60 | 10.73 + 7.54 | 1.54 / 1.97 |
| all levers (topk_rank in) | 3890.7 us | 14.30 | 2.90 + 0.86 = **3.76** | 21.46 + 7.38 | 10.73 + 7.50 | 1.84 / 1.91 |

**Expert-weight stall share of the non-byte time.**
- Non-byte time = token - floor = **2756.3 us** (all levers) / **2825.4 us** (default) [M].
- The part where a routed XCD has no expert weight in flight and waits on its routing is the gate_up staging +
  prologue: 3.76 x 26 = **97.8 us = 3.5 %** of non-byte time (all levers); 5.91 x 26 = 153.7 us = 5.4 % (default) [E].
- The rest of the expert phases' non-byte time is not a weight stall a prefetch can hide: stream excess
  (7.38 + 7.50 us/layer, time above the floor while streaming), handoff/ramp (events) and tail (XCD skew).

**The two ceilings, with a perfect predictor.**
- **(i) Hide the router-to-body latency.** Issue the predicted expert's first loads at gate_up start instead of after
  routing: **<= 3.76 us/layer = 97.8 us/token** [E].
- **(ii) Fill the idle HBM during the router window.** The window is router + gate_up pre-stream = 19.3 us/layer (all
  levers) [M]. Each routed XCD can hold at most 4 MB of prefetch in its private L2, out of the 11.5 MB of gate_up it
  streams (making weights cacheable ran 18 % slower: fleet `docs/STATUS.md:81`).
  - Upper bound, if those 4 MB then cost nothing: 4.19 MB / (11.53 MB / 28.84 us) = **10.5 us/layer = 272 us/token**
    [E].
  - That bound is not physical. Under load a load round trip is 0.73 / 0.71 / 0.69 us at L2 / MALL / HBM
    (`docs/MODEL-vllm-kernels.md:321` [M]). Cache-resident streams measured at most +9 % to +17 % faster, and slower
    at expert sizes (`docs/XCD-PREFETCH-2026-10-11.md` section 2 [M]).
  - At +17 %, (ii) is about 40 us/token [E].
  - Whether the non-temporal weight loads (`gemv.h:54-56`, `-DFLEET_NT_WEIGHTS=1` at `build.sh:52`) can hit a
    prefetched line at all is **[U]**.

**Why "all six".** The expert phase is one unit per XCD (`expert.h:1-6`), and the layer continues when the slowest
XCD's down arrives. A hit on five XCDs and a miss on one leaves the phase where it was, give or take the 1.8 us tail.
So the expected gain is P(all six predicted) x ceiling, minus the misses' cost:
- a predictor of k > 6 candidates splits the per-XCD budget, so the ceiling scales by 6/k;
- dynamic row rebalancing after routing, which would turn partial hits into gain, does not exist.

**Cost of a wrong prefetch.**
- In bytes: 24 MB of extra HBM reads per layer (6 XCDs x 4 MB), 5.7 us of bandwidth inside the 19.3 us window [E].
- In time: those reads land next to the router's latency chain (fold, norm, top-k). Filling idle bandwidth beside a
  latency chain was measured to double its round trip, 1.52 -> 3.08 us (`docs/XCD-PREFETCH-2026-10-11.md`, overlap
  bench [M]).
- Fleet's direct ancestor of this idea, holding prefetched loads across the event wait, measured **4 % slower**,
  about 150 us/token (fleet `docs/STATUS.md:30` [M]).

## Threshold (written before any code)

A prediction arm is worth a GPU session only if a per-token routing capture of this model and prompt shows the
following, averaged over the 26 MoE layers:

| predictor size k | ceiling (i) x 6/k | required P(t+1's six experts all inside the predicted set) |
|---|---|---|
| 6 | 97.8 us | **>= 30.7 %** |
| 8 | 73.4 us | >= 40.9 % |
| 12 | 48.9 us | >= 61.4 % |

Even against the unphysical ceiling (ii), k = 6 would need >= 11.0 %. The measured cross-context skew implies
popularity predictors sit near the random 1e-8 to 1e-5. So the bar can only be met by same-sequence persistence, at
least 30 % exact-set repeats, which is [U].

## What would change the verdict

One number: the same-sequence persistence of the exact top-6 set. It costs no new code. `routing_stats.py --model
<dsv2-lite> --steps 32 --batch 4 --dump ids.npz` runs in about 2 GPU minutes on any box that has already downloaded
the model (the ds_inplace `prep_dl` job does). `routing_stats.py --ids ids.npz` then gives persistence, recall and the
two decay curves on the laptop. Worth adding to the next DeepSeek session only as a free extra, not as a reason for
one. If it reports >= 30.7 %, step 3 (one default-off switch, the k = 6 persistence set, an arm paired against
default with reject on "any pair slower") is the design to write. Below that, the idea stays closed.
