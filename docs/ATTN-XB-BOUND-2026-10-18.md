# ETX_DSV2_ATTN_XB: bound and design (Step 1, no tile code)

Date: 2026-10-18. Step 1 base: 17996ae (x4xbl). Step 1b base: 0c641ec (PR #25, x4xbor = x4xbl +
ETX_DSV2_ROUTER_FIRST=1 + ETX_DSV2_ROUTER_SPREAD=1). B = 4 only. Script: `tools/attn_xb_bound.py` (Step 1b: x4xbor env).

## Design

- Grids: per request, `qabs (16, 2)`, `attn (16, 8)`, `merge (16, 8)`. Today, attn and merge are `(16, 16)`.
- Workers: request r uses the 16 workers `off + (r // 2) * 16 ..` of its slot (`off = (r % 2) * CUS`).
  - attn and merge: `hc -> (h % 2) * 8 + c + off + (r // 2) * 16`.
  - qabs: `hs -> (h % 2) * 2 + s + off + (r // 2) * 16` (the first 4 workers of the block, before attn in the queue).
  - Thus the 2 requests of a slot run qabs, attn and merge in one wave. Today, they run one after the other on workers 0..15.
- Events: `E_attn` wait_count 8 (was 16). `E_merge` wait_count `(16 / 8) * 8 = 16` (was 32).
- Tiles: new shim entries with `n_split = 8` (attention, attention_kvs, merge_uv). fleet bodies take `n_chunks` at run
  time: `attention_chunk` uses `per_chunk = ceil(seq / n_chunks)`, the merge loops over `n_chunks` partials. The KV
  cache writer stays `head 0, chunk 0`. The `n_split` value goes in as a const, so no new kernel-argument base.
- Equality: the merge order changes (8 partials, not 16), so the result is not bitwise equal to the 16-chunk path.
  The numpy model compares the 8-chunk path to its own reference bitwise, and to the 16-chunk path with a stated
  fp32 tolerance (Step 2).
- Switch off: no change to any existing grid, so plans and kernels stay byte-identical to 17996ae (Step 2 test).

## Per-slot timing model per layer (attention phase)

| arm | tasks per slot worker | time |
| --- | --- | --- |
| x4xbl | 2 x (qabs 4.0 + attn 16.7 + merge 7.6) | 56.6 us |
| attn_xb optimistic | qabs 4.0 + attn 20.3 + merge 7.6 | 31.9 us |
| attn_xb pessimistic | qabs 4.0 + attn 24.0 + merge 9.0 | 37.0 us |
| attn_xb stress | qabs 6.0 + attn 24.0 + merge 11.0 | 41.0 us |

attn 20.3 / 24.0 us at 8 chunks is fix's range (/workspace/bound_next_1p85.md). merge at 8 chunks reads 16 W_UV rows
(not 8) and merges 8 partials (not 16): 7.6 us optimistic, 9.0 us pessimistic.

## 27-layer protocol simulator, Step 1b (against x4xbor, the A of the KEEP rule)

3 seeds, kv_a shared, COH=l1, COUPLE=none, QKV_XB + OPROJ_XB + LMHEAD_XB + ROUTER_FIRST + ROUTER_SPREAD on.

```
{"tag": "base", "attn_us": 16.7, "merge_us": 7.6, "qabs_us": 4.0, "tasks": 173496, "cv0": 6186.4, "seeds": [6862.7, 6871.1, 6867.1], "mean": 6866.9}
{"tag": "opt", "attn_us": 20.3, "merge_us": 7.6, "qabs_us": 4.0, "tasks": 145848, "cv0": 5502.1, "seeds": [6145.8, 6145.9, 6151.1], "mean": 6147.6}
{"tag": "pess", "attn_us": 24.0, "merge_us": 9.0, "qabs_us": 4.0, "tasks": 145848, "cv0": 5639.8, "seeds": [6310.1, 6307.9, 6313.9], "mean": 6310.6}
{"tag": "pess2", "attn_us": 24.0, "merge_us": 11.0, "qabs_us": 6.0, "tasks": 145848, "cv0": 5747.8, "seeds": [6418.1, 6415.9, 6421.9], "mean": 6418.6}
```

- base reproduces x4xbor (6866.9 us = 7148.7 - 281.8).
- Gain: optimistic -719.3 us, pessimistic -556.3 us, stress -448.3 us. All are above the 0.3 ms stop line.

## 27-layer protocol simulator, Step 1 (against x4xbl, 17996ae; record only)

```
{"tag": "base", "attn_us": 16.7, "merge_us": 7.6, "qabs_us": 4.0, "tasks": 189304, "cv0": 6411.1, "seeds": [7143.1, 7150.8, 7152.2], "mean": 7148.7}
{"tag": "opt", "attn_us": 20.3, "merge_us": 7.6, "qabs_us": 4.0, "tasks": 161656, "cv0": 5726.8, "seeds": [6428.9, 6431.1, 6440.0], "mean": 6433.3}
{"tag": "pess", "attn_us": 24.0, "merge_us": 9.0, "qabs_us": 4.0, "tasks": 161656, "cv0": 5864.5, "seeds": [6591.9, 6594.5, 6603.8], "mean": 6596.7}
{"tag": "pess2", "attn_us": 24.0, "merge_us": 11.0, "qabs_us": 6.0, "tasks": 161656, "cv0": 5972.5, "seeds": [6699.5, 6702.5, 6711.8], "mean": 6704.6}
```

Gain against x4xbl: -715 / -552 / -444 us.

## KEEP threshold (written before any GPU measurement)

KEEP only if every token gate passes and `tools/ab_same_box.py` (6 rounds, all 6 orders, `--identity-cmd`,
A = x4xbor, B = x4xbora = x4xbor + ETX_DSV2_ATTN_XB=1 + -DETX_DSV2_ATTN_XB=1) gives verdict WIN (6/6 rounds faster,
p <= 0.05, median gain > max |A/A control|) and a median step saving (A - B, from ab.json) >= 0.587 ms
(pessimistic -556.3 us x 1.055). Otherwise DROP.

## Step 2 (built): ETX_DSV2_ATTN_XB=1, arm x4xbora

- The real x4xbora graph (graph.py, durations attn 20.3 / merge 7.6 us) in the same simulator: seeds
  [6145.8, 6145.9, 6151.1], mean 6147.6 us. This is the optimistic case of the bound (the bound script and the graph
  agree).
- Resources, etx_megakernel_d0 (VGPR + AGPR, scratch B/lane, SGPR spill, VGPR spill, WG/CU, LDS B):
  - ROCm 7.14 (gate): x4xbor 256 + 0, 80, 1796, 17, 2, 26668; x4xbora 256 + 0, 80, 1794, 17, 2, 26668.
  - ROCm 7.2.4 (record): x4xbor 248 + 6, 64, 1756, 14, 2, 26668; x4xbora 248 + 6, 64, 1754, 14, 2, 26668.
- Equality: not bitwise to x4xbor. tests/test_dsv2_attn_xb.py states the fp32 tolerance against the 16-chunk path.
