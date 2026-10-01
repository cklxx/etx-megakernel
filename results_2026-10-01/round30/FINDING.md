# Round 30: multi-request decode on the current build

Three interleaved rounds, wall ms/step; ETX token gate is every request matching 32/32 (all did).
`--check-dsl` refuses a multi-request plan, so the token match is the gate that applies above B=1.

| B | tasks / events | spill | vLLM /step | vLLM /token | ETX /step | ETX /token | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 60362 / 436   |  8 | 4.870 | 4.870 | 4.867 | 4.867 | **1.00x** |
| 2 | 120724 / 872  | 21 | 5.555 | 2.778 | 6.860 | 3.430 | 1.23x |
| 4 | 241448 / 1744 | 39 | 6.951 | 1.738 | 12.492 | 3.123 | 1.80x |

Old build (round 14): 1.08x / 1.29x / 1.90x. Every batch improved; the shape of the curve did not.

`ETX_VL_BATCH=1` at B=2: 7.114 (7.102/7.120/7.119) against 6.860 replicated, **+0.254 ms, still loses**
(was +0.29 on the old build, so the vectorised coherent paths and the spill drop moved it by 0.04).

## What the second request costs (trace, layer 0, B=2)

Bodies roughly double -- qkv 7.54 + 6.15 = 13.69 against ~9.5 for one request -- so each request streams its
own weights, as the plan's exact doubling of tasks and events says it must. But the waits anti-correlate:

| phase | r0 wait / body | r1 wait / body |
|---|---|---|
| qkv_0   | 18.51 / 7.54 | 3.83 / 6.15 |
| prep_0  | 1.25 / 4.42  | 11.11 / 4.53 |
| reduce_0| 6.05 / 5.94  | 15.18 / 5.71 |

Each request runs in the other's stalls. Layer period 126.6 us for two tokens = 63.3 per token against
~96.6 at B=1, so interleaving is worth 1.5x -- latency hiding, not bandwidth sharing. The step grows 41%
for a second token where vLLM's grows 14%, and that difference is the duplicated weight stream.

## A regression found and fixed on the way

The build at HEAD 22d46f5 measured 5.494 wall at B=1 against round 29's 4.865, with a **byte-identical plan**
(same md5). Cause: round 28 made the top-k fold decision derived in `model.py`, but `build.sh` still gated the
tile's `-DETX_VL_NO_TOPK_FOLD` on the env var, defaulting to "fold". The plan gave the top-k its own phase while
the expert tiles recomputed it anyway -- 448 redundant softmaxes per layer, round 7's 0.676 ms, measured here as
0.63. Tokens still matched, so nothing failed. Fixed by routing the decision through the feature manifest
(`plan.graph.codegen` -> `ETX_EMIT_FOLD_TOPK` in `plan_data.h`), which is the mechanism round 20 built for
exactly this and which rounds 14, 19 and now 30 have each been bitten by. Commit `f439088`; after it, 4.879
against round 29's 4.878 on the same box in the same sequence.
