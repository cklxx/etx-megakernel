# CPU verification and next-session registration (2026-10-12)

Condition: one MI300X VF, Qwen3-30B-A3B, bf16, batch 1, context 1024,
32 free-running tokens. [M] is retained measurement or a CPU observation with
its source, [E] is an estimate, [U] is unverified and contributes zero.
No GPU, provisioning or GPU Docker execution is part of this change.
The six mechanisms in LEVERS-MORE add zero ledger ceiling. KEEP remains
0.030 ms, every adjacent pair faster, median greater than twice max-minus-min
spread; atom bar 0.625 us/layer, S1 0.088 ms, S2 0.324 ms, cap 8400 s.

## 1. Reset: settled by CPU source reading

`etx/runtime/include/etx/host.h:hour_profile_run` records reset_start on the
stream, starts CPU clock a, calls reset_step, stops CPU clock b, then records
reset_stop. The host-submit mean b-a is **38.63 us (W) / 38.00 us (J)** [M],
32 HOUR-STAGE records (steps 1023..1054), in the two retained report.json files
under `results_qwen300_hour_20261011/session-{wklfw6bc,jc7gb969}`.
The stream event mean is **74.98 us (W) / 73.69 us (J)** [M]. It brackets the
reset commands: ctrl clear, event initial counts, domain slot clears, remaining
counts, per-domain arrival counters, mirror if enabled, local/global queue
initialization; execution-counter clear only when enabled. It includes stream
starvation between host enqueues and event overhead, not just DMA/memset
service. The CPU-submit and event intervals overlap: **do not add them**.

`launch()` records ev_start only after reset_stop and the residency query;
ev_stop follows the megakernel. Thus the 74.98 us is outside kernel_device_us
and outside the token_time prologue→argmax device span. It is inside the
host step w0→wms (`examples/vllm_llm/host.hip`, HOUR-STAGE producer).
Both kept seal and raw/hour_build.json explicitly have ETX_DEV_RESET=0;
the latter is the stamps build, not the kept timing build. Profile numbers
come from stages (ETX_HOUR_PROFILE), whose extra events perturb submission.
They are not a measured ordinary-wall saving.

The old **0.008 ms** is a historical *net wall gain* of ETX_DEV_RESET, not
existing reset device service: `docs/RESULTS-2026-10-01.md:107` explicitly
says -0.008 wall, 40.8 us fewer enqueues, 14 us returned inside the kernel.
It cannot price today's event interval. A1's 0.049 is the old estimated
host-submission ceiling. Account the whole reset path once in A1, replacing
that price only with a fresh ordinary-wall pair; do not add 0.075+0.039 to it.
Host H excludes pending reset service already inside wait-minus-kernel
(`second_stop`), and excludes kernel-outside-token time as a separately
recoverable round trip. This conservative host attribution is not a device
service subtraction proof. Cross-step residency would overlap A1; its old
0.005 estimate is not a bound on reset avoidance. Its actual gain is [U].

### Registered for the session: RESET.keep

One new build directory `reset`, kept recipe with **only ETX_DEV_RESET=1**;
all existing directories and flags stay as registered. `prepare` builds and
seals it before the measurement clock. `reset_pair.check_pair` verifies
compiler, import/DSL/original hashes, overrides and schedule; the generated
reset tail/header are the intentional differences. Import certificates now
record ETX_DEV_RESET rather than silently omitting this plan-affecting flag.
Fresh host --check 48 at 0/1040 and CHECK-DSL/32/32 precede three alternating
adjacent kept/reset pairs (31 post-first-token wall steps each), using
`reset_pair.run` → `candidate_pairs`; no profile/stamps in timing.

Measurement: ordinary wall gain and token-device delta on the same pairs.
Budget: **120 s / 2 min / $0.0663 [E]**, reserved inside B; correctness is
included. A cold load or slow checks may exhaust this, yielding NOT-COMPLETE,
never a shortened population. Expected [E] direction on reset ON: fewer
host enqueues, higher kernel-event time from the reset tail (outside the
token timers), lower wall if reset removal dominates. Reverse the flag and
those signs reverse. Token-device time should be roughly unchanged [E];
resource/scheduling effects are [U]. 74.98 us of wall recovery is not promised.
KEEP only at the unchanged bar; reject returns zero reset credit. Falling
back to host resets or changed math/schedule falsifies this experiment.
A passing singleton is measured again in D before claiming any final target.
