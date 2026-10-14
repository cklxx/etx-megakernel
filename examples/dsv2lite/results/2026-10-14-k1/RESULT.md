# DeepSeek-V2-Lite K1 session, 2026-10-14

One MI300X VF (droplet 608939000), GPU 0, HIP 7.14.60850; bf16, batch 1, context 1024, 32 tokens; five interleaved
pairs per study, device time deciding. Code 0fc7e06 (staged archive 84e83652...); session record
20261014T072651Z-6792a0, runner 07:26:51-07:34:45 UTC (7 min 53 s), every job rc=0. The fetch verified 120 files
against the SEAL. Omitted from the repository and kept locally: the six build directories `b_*` and
`trace_alld4r0l1kvs.stamps.bin` (112 MB; sha256 9f59f1dd... in MANIFEST).

Correctness: all 40 timed runs 32/32 tokens and 27/27 layers; preflight matched the pinned inputs.

| study | rule (commit) | device gains, us | median | spread | outcome |
|---|---|---|---|---|---|
| K1 `alld4r0l1kvs` | KEEP: every pair faster, median >= 30, > 2 x spread (2ec2d29) | 40, 40, 41, 48, 44 | +41 | 8 | **KEEP** |
| M1 `alld4r0l1mw` | same (3d27ff8) | 76, 65, 55, 74, 66 | +66 | 21 | **KEEP** |
| `alld4r0l1noQ` | REMOVE if every pair faster and median > 2 x spread (2ec2d29) | -29, -29, -23, -32, -22 | -29 | 10 | **STAYS** |
| `alld4r0l1noA` | same | -13, -6, -21, -6, -14 | -13 | 15 | **STAYS** |

Kept arm `alld4r0l1` device time in this session: 3.185-3.193 ms. Wall gains agree (K1 +41, M1 +68, noQ -30, noA -14).

Trace (one stamped K1 token, attribution only): span 3380.9 us; qkv 329.8, attn 354.0 us. The kept arm's stamped
token from 2026-10-13 (another session): qkv 434.2, attn 308.7 us.

Not measured: K1 and M1 together.
