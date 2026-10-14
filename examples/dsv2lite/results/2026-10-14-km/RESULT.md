# DeepSeek-V2-Lite KM session (K1 + M1 together), 2026-10-14

Same MI300X VF as the K1 session (droplet 608939000), GPU 0; bf16, batch 1, context 1024, 32 tokens; five rounds of
`alld4r0l1`, `alld4r0l1mw`, `alld4r0l1kvsmw`, device time deciding. Rule bf92fd4 (docs/DSV2-D1-D5.md "KM"), committed
before the code 87b0a2f. Record 20261014T081217Z-0ef82b, runner 08:12:17-08:14:27 UTC, every job rc=0; the fetch
verified 57 files against the SEAL. Build directories omitted (hashed in MANIFEST).

Correctness: all 15 timed runs 32/32 tokens and 27/27 layers; preflight matched the pinned inputs.

| comparison | device gains, us | median | spread | outcome |
|---|---|---|---|---|
| `alld4r0l1kvsmw` vs `alld4r0l1mw` (deciding) | 48, 52, 50, 58, 54 | +52 | 10 | **KEEP**: the new DeepSeek kept arm is `alld4r0l1kvsmw` |
| `alld4r0l1kvsmw` vs `alld4r0l1` (attribution only) | 120, 125, 122, 120, 119 | +120 | 6 | - |

Device time per arm (5 rounds): `alld4r0l1` 3.188-3.195 ms, `alld4r0l1mw` 3.120-3.126 ms, `alld4r0l1kvsmw`
3.068-3.075 ms. Wall gains agree (+122 / +54). The two levers measured separately on the K1 session summed to +107 us
(+41, +66); together they give +120 us in this session.
