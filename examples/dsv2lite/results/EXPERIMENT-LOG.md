# DeepSeek experiment log — four columns

Every experiment run on DeepSeek-V2-Lite / DeepSeek-Coder-V2-Lite, in both repositories.
`ETX` = the port under `examples/dsv2lite` (the target). `fleet` = `fleet-mi300x` (the reference kernel).
GPU column: the session that measured it. All GPU numbers are paired medians against the same-session default.

| Experiment | Why we think it is a good experiment | Run on GPU? | Result |
|---|---|---|---|
| gemv_no_terminal | Drop the terminal dummy loads at the end of the GEMV K-loop. The byte account had counted them as issued HBM bytes, worth about 120 us on qkv | yes — session 2026-10-13, arm `Q` | −3 us. Dead. The removed loads re-read a row the workgroup had just loaded: L2 hits, not HBM bytes |
| gemv_no_terminal_all | Same change at every GEMV site, not only the last one, for about 310 us on the byte account | yes — arm `Qall` | −3 us alone, and −14 us when kept in the combination. Dead, and removed |
| topk_rank | One rank pass over the logits instead of repeated scans. The stage measures 128 us; the rank re-derivation said about 89 us | yes — arm `topk` | +66 us, every pair faster. 74% of the expectation. Kept |
| norm_tree_all | Do the norm reduction once for every consumer instead of once per consumer. in_norm measures 168 us against 128 us expected | yes — arm `Nall` | +37 us, spread 7. Kept |
| norm_tree | The same reduction sharing without the all-sites variant | yes — arm `N` | +30 us. Kept as attribution only |
| norm_gamma_early | Hoist the gamma into the earlier pass so the later pass disappears | locally only (CPU gates + ROCm resources) | PASS locally, 0 scratch, AGPR 72. Never measured alone |
| attn_prologue_live | Compute the attention prologue live rather than materialising it. prep measures 129 us against 65 us expected | yes — arm `prep` | +27 us, every pair faster. 41% of the expectation. Kept |
| argmax_tree | Tree-shaped argmax instead of a linear scan. The stage only measures 4.7 us | yes — arm `argmax` | +2 us with spread 21: inside the noise. Dead |
| attn_cache_double_buffer | Double-buffer the attention cache to overlap its fill with use | locally only, three variants | REJECTED by the local gate every time: +160 B/lane scratch, +60 AGPR. Never went to GPU |
| stamps (STAMPS2) | Instrument each stage with one stamp per cache line, so every stage has a measured value rather than an inferred one | yes — trace builds in the same session | No speed verdict, by design. Produced the per-stage table the expectations are judged against |
| combined 5-lever arm (`all`) | The five passing levers together: topk_rank + norm_tree_all + attn_prologue_live + argmax_tree + gemv_no_terminal_all | yes — arm `all` | +92 us, all five pairs faster: GO. Then re-measured without the dead lever |
| `all` minus gemv_no_terminal_all | The same combination with the lever whose byte basis was wrong removed | yes — follow-up, 5 rounds | **+108 us**, spread 7, holds. The best ETX result of the day |
| Q-all inside the combination | Check whether a lever that is neutral alone still steals resources from the others | yes — follow-up, 5 rounds | Costs 14 us in the combination. Dropped |
| ROCm 7.2.3 against 7.14 | The box ran 7.14 while the local resource records were built on 7.2.4. Same generated sources, two compilers | yes — follow-up, 5 rounds | **7.14 is 62 us slower** (3.708 against 3.646 ms), every pair holds. The 7.2.3 build uses 48 AGPR |
| fleet F: per-XCD local expert fold | Do the expert fold where the data already sits instead of crossing XCDs, per the fleet trace | yes — fleet session 2026-10-12 | REJECTED: the QKV stage goes 367.1 → 374.6 us (+7.5) |
| fleet K: attention variant | Repay the gather and publication waits at the attention body-ready point | yes — fleet session 2026-10-12 | REJECTED: attention goes 210.55 → 219.62 us (+9.1) |
| fleet: six switches, all levers | Port the port's own deficits back into the reference kernel, one switch each, default OFF | locally only | 22/22 gates, 24/24 default artifacts byte-identical. GPU never measured; the repository was misdirected |
| fleet: cached working set | Keep the hot weight set resident instead of streaming it, on the theory that reuse beats a fresh read | yes — earlier fleet session | 18% slower. Closed, no further work |

## The two accounting lines

- expected for the ETX combination: 531–533 us. Delivered: 92 us, then 108 us after dropping the dead lever. 17–20% realised.
- ceiling: 5.20 GB/token at the measured 4.2 TB/s = 1238 us = 808 tok/s. Best measured on the ETX box: fleet 3.568 ms = 280 tok/s.
