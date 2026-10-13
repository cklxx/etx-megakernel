# DeepSeek experiments — the full list

Scope: every experiment run on DeepSeek-V2-Lite / DeepSeek-Coder-V2-Lite, in both repositories.
ETX port = `examples/dsv2lite` (the target). fleet = `fleet-mi300x` (the reference kernel).
Sources: `results/levers/*.json`, `results/PRE-GPU.md`, `results/2026-10-13-ds_inplace/RESULT.md`,
`fleet-mi300x/docs/STATUS.md`, `fleet-mi300x/results/phase-table-2026-10-12.md`.

## A. ETX port — 13 local lever arms (CPU gates + ROCm 7.2.4 device resources)

| arm | verdict | VGPR | AGPR | scratch B/lane | waves/SIMD |
|---|---|---:|---:|---:|---:|
| default | PASS | 256 | 48 | 0 | 1 |
| gemv_no_terminal | PASS | 256 | 48 | 0 | 1 |
| gemv_no_terminal_all | PASS | 256 | 30 | 0 | 1 |
| norm_gamma_early | PASS | 256 | 72 | 0 | 1 |
| norm_tree | PASS | 256 | 72 | 0 | 1 |
| norm_tree_all | PASS | 256 | 88 | 0 | 1 |
| argmax_tree | PASS | 256 | 49 | 0 | 1 |
| topk_rank | PASS | 256 | 50 | 0 | 1 |
| attn_prologue_live | PASS | 256 | 62 | 0 | 1 |
| stamps (trace build) | PASS | 256 | 58 | 0 | 1 |
| combined 5-arm | PASS | 256 | 86 | 0 | 1 |
| combined 5-arm + stamps | PASS | 256 | 99 | 0 | 1 |
| **attn_cache_double_buffer** | **REJECTED** | 256 | 108 | **160** | 1 |

## B. ETX port — on-GPU session 2026-10-13 (MI300X VF, ROCm 7.14, 5 interleaved rounds)

Absolute medians and the paired gain against the same-session default:

| arm | what it changes | device ms | paired gain vs default | verdict |
|---|---|---:|---:|---|
| default | — | 3.709 | — | — |
| fleet | the reference kernel | 3.568 | 139 µs | reference |
| all | 5 levers combined | 3.616 | **92 µs** | **GO** |
| Q | gemv_no_terminal | — | −3 µs | dead |
| Qall | gemv_no_terminal_all | — | −3 µs | dead |
| N | norm_tree | — | 30 µs | attribution |
| Nall | norm_tree_all | — | 37 µs | attribution |
| topk | topk_rank | — | 66 µs | attribution |
| prep | attn_prologue_live | — | 27 µs | attribution |
| argmax | argmax_tree | — | 2 µs (spread 21) | dead |

- all vs fleet: **−49 µs** (the ETX port with every passing lever is still slower than fleet).
- Single levers sum to 129 µs, the combination measures 92 µs: not additive.
- Every invocation passed 32/32 HF-greedy tokens, 27/27 layers, 32/32 warm-up.

## C. ETX port — per stage, measured against the expected saving (stamped builds)

| stage | default µs | all µs | measured saving | expected saving |
|---|---:|---:|---:|---:|
| prologue | 140.5 | 138.4 | 2.1 | 14.2 |
| in_norm | 190.8 | 173.2 | 17.6 | 40.0 |
| qkv | 448.8 | 437.1 | 11.7 | 120.1 |
| prep | 143.6 | 139.8 | 3.7 | 63.5 |
| attn | 345.9 | 349.6 | −3.7 | 0.0 |
| reduce | 315.9 | 314.7 | 1.2 | 12.5 |
| o_proj | 279.4 | 277.0 | 2.4 | 28.9 |
| router | 368.2 | 357.9 | 10.3 | 52.9 |
| topk | 148.1 | 94.2 | 53.9 | 85.4 |
| moe_up | 767.4 | 798.5 | −31.1 | 57.7 |
| moe_down | 552.9 | 558.0 | −5.1 | 39.7 |
| lm_head | 135.0 | 129.4 | 5.7 | 4.4 |
| argmax | 6.0 | 5.4 | 0.6 | 0.5 |
| tail | 117.4 | 117.6 | −0.2 | 13.6 |
| **sum** | | | **69.2** | **533.4** |

Realised 92 µs of an expected 531–533 µs: **17%**.

## D. Rejected experiments and why

| experiment | result | reason it died |
|---|---|---|
| attn_cache_double_buffer (3 variants) | REJECTED | +160 B/lane scratch, +60 AGPR; local gate fails |
| gemv_no_terminal(_all) | −3 µs | the removed terminal loads re-read a row already in L2; the byte-account basis never applied |
| argmax_tree | 2 µs, spread 21 | inside the noise |
| ETX default 3.473 ms record (2026-09-27) | not reproduced | different box and ROCm; 3.709 ms here; absolute ms are not comparable across sessions |

## E. fleet-side DeepSeek experiments (2026-10-12, the reference kernel)

| experiment | result | verdict |
|---|---|---|
| F: per-XCD local expert fold | QKV stage 367.1 → 374.6 µs | REJECTED (+7.5) |
| K: attention variant | attention stage 210.55 → 219.62 µs | REJECTED (+9.1) |
| 6 switches, all levers | 22/22 gates, 24/24 default artifacts byte-identical | local pass, GPU unmeasured |
| cached working set | 18% slower | closed |
| per-stage phase table | step 3576.8 µs; LM_HEAD 130.2 µs the largest single task | instrumentation |

## F. The two numbers that frame all of it

- byte ceiling: 5.20 GB/token at the measured 4.2 TB/s = **1238 µs = 808 tok/s**.
- best measured today: fleet 3.568 ms = 280 tok/s on the ETX box, default 3.709 ms = 270 tok/s.
