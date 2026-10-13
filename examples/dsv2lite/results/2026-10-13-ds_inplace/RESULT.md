# ds_inplace session, 2026-10-13: result

Generated from this directory's JSON (pairs_*.json, stages_*.json, report.json). One MI300X VF (gfx942), ROCm 7.14
(hipcc 7.14.60850), DigitalOcean box, batch 1, context 1024, 32 tokens, repeat 2; ETX dsv2-session @7f3dd89, fleet
3fb67a6. Every invocation passed 32/32 HF-greedy tokens, 27/27 layers and a 32/32 warm-up. Device = embed -> argmax
median on the device clock; 5 interleaved rounds per comparison, order reversed on odd rounds.

## Absolute times (medians of 5)

| arm | device ms (min-max) | wall ms |
|---|---|---:|
| default (baseline rounds) | 3.709 (3.704-3.716) | 3.743 |
| fleet (baseline rounds) | 3.568 (3.561-3.570) | 3.580 |
| all (all rounds) | 3.616 (3.614-3.621) | 3.652 |
| default (all rounds) | 3.712 (3.702-3.717) | 3.746 |
| fleet (all rounds) | 3.568 (3.564-3.570) | 3.582 |

## Paired device gains vs default (us/token; positive = faster than default)

| arm | median | spread | every pair faster | rule |
|---|---:|---:|---|---|
| baseline | 139 | 20 | True | attribution |
| all | 92 | 18 | True | GO |
| Q | -3 | 7 | False | attribution |
| Qall | -3 | 10 | False | attribution |
| N | 30 | 16 | True | attribution |
| Nall | 37 | 7 | True | attribution |
| argmax | 2 | 21 | False | attribution |
| topk | 66 | 10 | True | attribution |
| prep | 27 | 14 | True | attribution |

all vs fleet: median -49 us (negative = slower than fleet), spread 4.

## Per stage: one stamped token each (STAMPS2 builds; attribution, not a verdict)

| stage | default us | all us | measured saving | expected saving |
|---|---:|---:|---:|---:|
| prologue | 140.5 | 138.4 | 2.1 | 14.2 |
| in_norm | 190.8 | 173.2 | 17.6 | 40.0 |
| qkv | 448.8 | 437.1 | 11.7 | 120.1 |
| prep | 143.6 | 139.8 | 3.7 | 63.5 |
| attn | 345.9 | 349.6 | -3.7 | 0.0 |
| reduce | 315.9 | 314.7 | 1.2 | 12.5 |
| o_proj | 279.4 | 277.0 | 2.4 | 28.9 |
| router | 368.2 | 357.9 | 10.3 | 52.9 |
| topk | 148.1 | 94.2 | 53.9 | 85.4 |
| moe_up | 767.4 | 798.5 | -31.1 | 57.7 |
| moe_down | 552.9 | 558.0 | -5.1 | 39.7 |
| lm_head | 135.0 | 129.4 | 5.7 | 4.4 |
| argmax | 6.0 | 5.4 | 0.6 | 0.5 |
| tail | 117.4 | 117.6 | -0.2 | 13.6 |
| **sum** | | | **69.2** | **533.4** |

## Against the expectation (examples/dsv2lite/results/stages/stage_table.md)

- expected 533.4 us arithmetic / 531.1 us simulated; measured 92 us paired (17% of the simulated plan).
- gemv_no_terminal(_all): expected 120 us (qkv) / ~310 us (all sites) from the byte account; measured -3 us each.
  The removed terminal loads re-read a row the workgroup had just loaded: L2 hits, not HBM bytes. The BYTES basis
  does not apply to them; their rows of the stage table were the flagged upper bounds.
- topk_rank 66 us (expected ~89), norm_tree_all 37 us (~84), attn_prologue_live 27 us (~66): 40-78% realised.
- single levers sum to 129 us against 92 us combined: not additive.
- ETX default measured 3.709 ms here, not the 3.473 ms record of 2026-09-27 (different box, ROCm 7.14, fleet 3fb67a6): absolute ms are not compared across sessions.

Raw STAMPS2 stamp files (2 x 112 MB) are kept outside git in ~/code/etx-cache/results_2026-10-13-ds_inplace/;
SHA256SUMS here covers every file of the session, including them.
