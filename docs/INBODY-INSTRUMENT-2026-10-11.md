# In-body instrumentation — 2026-10-11

Starting commit: `4a39f92`; worktree/branch: `etx-inbody`, `wf/inbody`.
CPU only: no GPU, provisioning, downloads or network spend. Local Docker builds
used the already installed `etx-rocm22` image with `--pull=never --network=none`.
The supplied target remains 4.180 -> 3.333 ms (approximately 0.847 ms).
**This change supplies runnable observations, not an exclusive-removability certificate.**

## Record and isolation

Ordinary emission and ABI are unchanged. `ETX_INBODY_BUILD=1` in `build.sh`
selects an independent diagnostic translation unit and defines `ETX_INBODY=1`
for BOTH host and device. The diagnostic executable needs `ETX_INBODY_RAW=FILE`
to allocate records. Use the ordinary executable for stamps OFF; a diagnostic
executable with null recording pointers is not the byte-identical OFF control.
Keep KEY32 and every other mathematical/configuration flag identical across arms.

The original `trace_time[task][4]` remains `{T, B, E, worker}`:
T = entry to `etx_run_task`, after dequeue/broadcast; B = after waits/acquire and
barrier, before body; E = after all-wave DOMAIN release and barrier, before
phase-3 releases/arrivals. T is not the time a task became runnable; E is not
necessarily global payload publication. Legacy `ETX_TRACE_RAW` remains nine
little-endian i64: type, four coordinates, and those four words.

The diagnostic file has six little-endian u64 header words:
`{0x3142444f424e4954, 1, task_count, event_words, waves_per_task, 10}`.
Arrays follow, without filtering: legacy `[task][4]`, poll `[task][1]`,
event `[word][before, after, final_producer_task]`, wave `[task][wave][Cw, Rw]`.
Each timer tick is 10 ns, not one ALU clock cycle. Counts/lengths are checked.

P (poll-success) is sampled immediately after the original wait returns,
before acquire. Multiple input waits retain their final success observation.
It is an issue timestamp after the successful read, not the exact load cycle.
Abort/backoff, scoped polls, releases, arrivals and dynamic pushes are retained.

Arrival records bracket the ORIGINAL arrival call; only its zero-returning
producer writes the event row. The bracket includes any last-arriver flushing
inside that call. Diagnostic stores happen after zero becomes observable;
consumers NEVER wait for those stores. The reader resolves a task's final
readiness bracket `[Alo,Ahi] = [max(input.before), max(input.after)]` using the
plan's exact event coordinates. It returns the six-word task view
`{T,B,E,worker,Ahi,P}`, its arrival bracket and final producer task IDs.
Root tasks use T as an explicitly synthetic readiness origin, with no input poll.
There is no hardware instruction here that timestamps the atomic's exact cycle.

Cw is sampled immediately after the tile returns, before its retained DOMAIN
release; Rw after that release, before the retained barrier. Every wave's lane
zero writes its own row. No barrier or data-protocol fence is added. Stamps
add diagnostic stores to the ON release path, so ON perturbation includes them.
Cw is tile return: internal compute/store/reuse barriers can precede it.
It must not be called the last arithmetic operation. DOMAIN drainage does not
replace a later DEVICE write-back. Root/abort/missing-wave cases cannot get
fabricated completion credit.

## Per phase, per layer, with explicit measurement limits

`python -m etx.tools.inbody PLAN RAW --bytes PER_TASK_BYTES.json --output REPORT.json`
returns every raw word, every task, arrival owners, tied last-return/drain waves,
the existing timeline critical path, and phase/layer aggregates. Optional
`--legacy-output FILE` exports the original nine-i64 format for existing tools.
Optional `--hops FILE` takes independently supplied per-type hop estimates.
Missing byte rules or hops stay unknown; impossible estimates are not clipped
into a positive result. Raw data is never replaced by phase means.

Let k=0.01 us/tick, C=max(Cw), R=max(Rw), and D be the selected critical
predecessor's E (T at the window origin for the first item). Observations are:
- Readiness-to-observation: `[max(0,P-Ahi), P-Alo] * k`.
  This includes scheduling, poll phase and event visibility, not just data waiting.
- Acquire/setup/broadcast: `(B-P)*k`; tile-return interval: `(C-B)*k`.
- Working after success: `(C-P)*k`; all-wave drain tail: `(R-C)*k`;
  barrier tail: `(E-R)*k`. Wave drain durations overlap; do not sum them.
- Legacy body: L=`(E-B)*k`; legacy critical gap: G=`max(0,B-D)*k`.
  Producer publication is within G where dependencies dominate, not inside L.

The four-class algebra per critical item is EXACT as an accounting identity,
NOT four exclusive hardware measurements:
1. Bytes floor estimate F=`logical_task_bytes / (4.3 * 10^6)` us.
2. Hop H=independently measured exposed load latency, otherwise UNKNOWN.
   An ISA sum or a sparse pipeline issue interval alone is only an estimate.
3. Machinery residual M=`L-F-H`, only with valid F,H and `0 <= F+H <= L`.
4. Dependency/queue gap G as above, with predecessor kind and consumer owner.
   Therefore `F+H+M+G = L+G`. No saving follows merely from this identity.

Phase/layer rows sum the selected critical items' L/G and observation tails;
they also report the all-grid bytes floor separately. Summing each selected
task's full-bandwidth floor does NOT reproduce the global 6.1904 GB floor:
concurrent tasks share bandwidth. Neither floor identifies a wall-clock interval
occupied exclusively by bytes. Residual machinery therefore remains modeled.
The reader deliberately credits no exclusive removable time from these stamps.

Worked **synthetic CPU-test example**, layer 0, `moe_down`, task 1:
`{T,B,E,worker,Ahi,P}={201,240,400,0,220,230}`, Alo=215;
C=360, R=390, predecessor E=200. Waiting is 0.10--0.15 us;
acquire 0.10 us; tile return 1.20 us; drain tail 0.30 us; barrier 0.10 us.
With an explicitly supplied 430,000-byte task and a hypothetical H=0.20 us:
F=0.10, M=1.30, G=0.40 us; all four sum to 2.00 us.
Only the timestamp intervals are observed here; F/H/M are estimates and
**no item is certified removable**. See `docs/evidence/inbody-worked-example.json`.

## Runnable pack, replay and raw reader

The full model keeps the existing `prep.py` vl.bin/vl.manifest format and an
independent HF golden. `bf16_300/pack.py --src LOCAL_HF` optionally invokes that
packer; no checkpoint is downloaded. A CHECK host built with
`ETX_BF16_CAPTURE_BUILD=1` exports original 160-byte argument blocks and mutable
inputs immediately before the original constituent, and independent output
bytes immediately after it, inside the existing sequential `host --check`.
Weight slices refer to the full model's vl.bin and are hashed, not recopied.
`pack.py` binds inputs/reference to original host/plan/check-log identities.
Use a separate snapshot directory and retain both raw blocks and references.

Capture covers one selected expert of MoE up/down at one layer/position,
including its XCD replica's original input/routing pointers. Mode-zero `_fold`
MoE aliases have the same bodies and are supported; top-k recomputation modes,
norm-fold pointers, batching and extended layouts are refused. The snapshot
is a replay slice from a full-model pack, not a replacement for the full model.

`bf16_300.launch_build --out DIR --pair PAIR --task T --wave W --row R`
builds unchanged original and stamped modules plus `launcher.hip` with the VM's
HIP compiler. `--local` instead emits gfx942 IR/ISA and checks host syntax.
Use unique directories per pair/tuple; T is the local tile block, not global
task ID (reader emits tile_block). Completion stores are diagnostic-only.

Invocation (inside the existing session environment):
```sh
python -m examples.vllm_llm.bf16_300.launch --selector candidate --pair body \
  --pack SNAP/moe_up/pack.json --build PROBE --arguments SNAP/moe_up/arguments.bin \
  --reference SNAP/moe_up/reference.bin --layer 0 --task 0 --wave 0 --row 0 \
  --iterations 5 --deadline 60 --output OUT/sample
```

Every invocation alternates original/stamped pairs, restoring mutable buffers
and output holes, and checks the selected expert's entire output byte-for-byte
against the supplied independent reference. The selector chooses the primary
reported arm; records always identify the stamped companion. The original
control has no fabricated timestamps. Full logs, config, `.bin`, timing JSONL
and JSON retain compiler/resources/module identity and all task/wave evidence.
Each iteration's raw file contains one 32-byte `QQIIII` sample followed by
`count*16` records `{begin,end,task,wave,xcd,row}`. Truncation/duplicates/missing
waves and wrong selectors fail. Last-wave evidence covers this replay slice,
not the full model's critical wave. Reported perturbation is whole standalone
grid device span, not a per-task or full-model wall-time measurement.
The launcher does not implement the standalone prep/attention fusion prototype,
dispatcher probes, full-model candidate integration, or HF token certification.

## Measurement, acceptance and budget

No GPU perturbation (OFF or ON, counters 0 or 1), byte check or 32/32 token result
was measured here. CPU OFF identity is code evidence, not a measured 0.000 ms.
The enabled builds change allocation/spills; sparse evidence remains unqualified
under `bf16_300/attribution.py`'s equal-resource gate. Do not subtract away spills.

`examples/sessions/inbody.py` supplies ONE pinned gpu_jobs job with a 900-second
hard stop, full logs, 15-second progress and stuck-state probes. Generate its
explicit configuration with `--template FILE`; prepare packs, probe modules and
same-math KEY32-on original/OFF/ON CHECK hosts beforehand. Full-model hosts also
need `-DETX_INBODY_REPORT` to report every repetition outside the timed step.
`python -m etx.tools.gpu_jobs examples/sessions/inbody.py --dry-run` is CPU only.

Order: fresh imported-launch checks at positions 0/1040 for each build; five
complete 32/32 repetitions per arm at counters OFF and ON; lossless full-step
records; sparse replay pairs. OFF median movement must be within the larger
original/OFF run spread. Report ON-minus-OFF independently for both counter
settings. These are noninterleaved arm distributions, not a fusion verdict.
Missing/failed repetitions or a deadline produce NOT-COMPLETE, never a shorter
run's number. Heavy cold model loading may prevent completion in fifteen minutes.

Pre-registered SUCCESS requires >=90% of same-condition machinery+gap+hop
attributed to named, nonoverlapping, exclusive atomic items AND the largest
single removable item with a validated price. If met, propose only that item
for a separate correctness-gated experiment; do not promise the entire target.
Otherwise retain raw data, report NOT-ESTABLISHED and specify the missing
observability/control. This reader/job currently fail closed: issue stamps and
modeled F/H/M cannot meet that exclusive-removability test. Measurement completion
is distinct from attribution success; the job cannot manufacture a success.

Measurement budget: 0.25 GPU-hours, estimated $0.50 at the repository's historical
$1.99/GPU-hour rate; setup/pack/build time extra, not a current price quote.
No money was spent here. Fetch results and delete a future VM immediately after
completion/deadline using the established session path.

Brief corrections: 847/2972.6 = **28.5%**, not 35%. The supplied pools come from
the OFF trace and cannot be transferred unchanged to the KEY32-on 4.180 ms step.
Two stamps cannot isolate load-hop latency, exact atomic transition time, last
arithmetic compute, or prove removal. Those remain deciding unknowns; the new
reader makes them explicit rather than relabeling residual time as a saving.

## CPU evidence

See `docs/evidence/inbody-2026-10-11.json` and retained full logs under
`results_inbody_cpu_20261011/`. Toolchain/device libraries: Homebrew LLVM 23 and
ROCm 7.2.4 clang 22 in the existing x86 container, gfx942, with
`ETX_DEVLIBS=/Users/ckl/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode`.
Default two-task emission equals the starting commit; OFF optimized IR/ISA
match after only nonexecutable source-ID/location normalization, and OFF ELF
`.text` bytes match. Full Qwen code and exact resource/spill reports are retained.
All twelve sparse pairs compile; launcher and diagnostic/capture host syntax
checks pass. Final full-suite/matrix counts and artifact hashes are in the evidence.
Earlier failed syntax/path checks and resource-disqualified builds are retained;
CPU-only failures cost zero GPU-hours. Protected project documents were untouched.
