| quantity | B1 | B2 | B4 | B8 (plan) |
|---|---:|---:|---:|---:|
| megakernel dispatches per step | 1 | 1 | 1 | 1 |
| grids per step | 271 | 438 | 876 | 1752 |
| tasks per step | 53022 | 109892 | 219784 | 439568 |
| resident worker slots N | 296 | 592 | 592 | 592 |
| requests per slot | 1 | 1 | 2 | 4 |
| tasks per slot (mean / max) | 179.1 / 186 | 185.6 / 219 | 371.3 / 438 | 742.5 / 876 |
| summed body W (ms) | 737.3 | 1934.0 | 3852.4 | — |
| body per task (us) | 13.90 | 17.60 | 17.53 | — |
| slot busy fraction u = W/(N span) | 0.8049 | 0.7529 | 0.7506 | — |
| traced span (ms) | 3.0944 | 4.3392 | 8.6702 | 17.34 [E] |
| span per generated token (ms) | 3.0944 | 2.1696 | 2.1675 | 2.17 [E] |
| critical path: bodies (ms) | 2.7157 | 3.7398 | 7.3408 | — |
| critical path: dependency gaps (ms) | 0.3787 | 0.5322 | 0.9631 | — |
| critical path: queue gaps (ms) | 0.0000 | 0.0672 | 0.3662 | — |
| host step of the traced run minus span (ms) | 0.1598 | 0.2346 | 0.3410 | — |
| replicated bytes per step [E] (GB) | 4.934 | 9.869 | 19.738 | — |
| bytes / span [E] (TB/s) | 1.595 | 2.274 | 2.277 | — |

| step | total (ms) | tasks | body per task | slots | busy fraction |
|---|---:|---:|---:|---:|---:|
| B1 -> B2 | +1.2448 | x2.0726 = +3.3190 | x1.2657 = +1.7038 | x0.5000 = -4.0586 | x1.0691 = +0.2806 |
| B2 -> B4 | +4.3310 | x2.0000 = +4.3392 | x0.9960 = -0.0348 | x1.0000 = +0.0000 | x1.0031 = +0.0265 |

| family | body/task B1 (us) | B2 | B4 | B2/B1 | B4/B2 |
|---|---:|---:|---:|---:|---:|
| gate_up | 28.79 | 36.63 | 35.77 | 1.272 | 0.976 |
| down | 15.38 | 22.32 | 21.75 | 1.451 | 0.974 |
| qkv | 13.02 | 16.67 | 16.44 | 1.280 | 0.986 |
| attn | 15.06 | 18.50 | 19.26 | 1.228 | 1.041 |
| router | 8.14 | 10.25 | 10.17 | 1.258 | 0.992 |
| oproj | 6.54 | 8.68 | 8.85 | 1.328 | 1.020 |
| merge | 4.73 | 6.19 | 6.41 | 1.309 | 1.036 |
| qabs | 3.67 | 4.36 | 4.65 | 1.188 | 1.066 |
| lm_head | 121.88 | 124.77 | 129.59 | 1.024 | 1.039 |
