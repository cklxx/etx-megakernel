# Convergent baseline, 2026-10-12

One page: the number we measure from now on, what backs it, what is banked, what is closed, and what is open.

## The baseline

| quantity | value | label | evidence in the tree |
|---|---|---|---|
| step time, host wall | **4.078 ms** | [M] | `docs/GAINS-LEDGER.md`, section "2026-10-12, after the baseline refresh" |
| step time, device | **3.943 ms** | [M] | same |
| throughput | **254 tok/s** | [M] | same |
| target | 3.3333 ms = 300 tok/s | rule | `AGENTS.md` section 1 |
| gap, device basis | **0.61 ms** | [M] | 3.943 - 3.3333 |

The build is `examples/vllm_llm/kept_build.py`: router narrowing in `KEPT_ENV` (`ETX_VL_ROUTER_NARROW=1`),
normalisation hoist in `BASE_FLAGS` (`-DETX_DSL_NORM_HOIST`). Landed in commit `45e1d36`. Every later arm is
measured from this baseline, and staging a comparison after a refresh must rebuild the baseline arm too.

**Evidence status.** The numbers above are recorded in the ledger, but the raw run records died with the box:
the pairing report and the per-step `gen32` logs sat in `/root/sess/` and `/root/moe_run/` on droplet
`608307900`, destroyed 2026-10-12 17:21 CST. No `docs/` or `results_*` file carries them. Treat 4.078 / 3.943
as a recorded measurement, not as a re-derivable one, and copy the session directory into
`examples/<project>/results/<date>/` before the next destroy.

## Timeline and cost

| what | when |
|---|---|
| box created (droplet `608307900`, `atl1`, MI300X VF) | 2026-10-12 13:48 CST (05:48:43Z) |
| box destroyed | 2026-10-12 17:21 CST |
| lifetime | 3 h 33 min |
| spend at $1.99/h | **about $7.1** [M] |

## Banked

| lever | gain | basis |
|---|---|---|
| router narrowing | 0.0626 ms | five interleaved pairs, every round faster |
| normalisation hoist | 0.0556 ms | same |
| **total banked** | **0.118 ms** | 3.943 -> 3.943 - 0.118 |

## Closed

| route | number | why closed |
|---|---|---|
| MoE rows per task 24 / 16 / 16 with 304 workers | -0.140 / -0.356 / -0.413 ms | five interleaved pairs each, all negative, 32/32 held |
| single-wave row ownership (`ETX_DSL_ROW_WAVE`) | barriers 168 -> 170, LDS ops 1101 -> 1119; constant segment count 170 / 1119 | static only: the tile adds sync work instead of removing it |
| no-touch and isolate-poller | 0.0069 / -0.0002 ms | below the 0.030 ms KEEP bar |

## Open

| item | ceiling | basis | cost to measure | decision it moves |
|---|---|---|---|---|
| device-side reset | **0.1134 ms** | [E] upper bound; historical readings 0.073-0.075 ms | about 30 min of a fresh box | whether the reset path is worth a lever at all |
| the gate `G` | **0.088 ms** threshold | [M] atoms; raw figure 0.356727 ms | about 15 GPU minutes | whether the 300 tok/s route stays open |

Neither is a banked gain: the reset row is an `[E]` ceiling, and this project's predictions have landed at
0.4% against 1.5% predicted (about 0.27x), so the realized reset gain may be near the 0.030 ms bar. `G` produces
a verdict, not milliseconds.

## Two proposals, and where each lands

1. **Fix RPT=32 and remove only the partial matrix, the gather and one barrier, then run an isolated A/B.**
   This experiment is **already written**: the `ETX_DSL_ROW_WAVE` tile in `etx/ops/gemv.py` does exactly that at a
   fixed grid (192 workgroups, 8 experts, RPT 32), `#ifdef`-guarded and default off. What it has **not** had is a
   GPU pair - it was closed on static evidence (the assembly counts above). The static verdict can be wrong: a
   barrier count over the whole megakernel is not the critical path. One pair, about 15 GPU minutes, settles it.
2. **Increase RPT (48, 64) to test whether 32 is a local optimum in task granularity.**
   New, and cheap to build: the grid is `TOPK * (MI / RPT)`, so RPT 48 gives 128 workgroups and RPT 64 gives 96,
   against 192 today. Build-time gate `MI % 32 == 0` holds at `MI = 768`. What it does **not** answer on its own is
   *why*: fewer workgroups means less HBM concurrency and more weight resident per task (384 KiB and 512 KiB
   against 256 KiB), so a loss there is as ambiguous as the 24/16 arms. Use it as a task-granularity probe, not as
   a fixed-cost measurement.
