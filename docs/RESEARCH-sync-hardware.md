# Research: hardware levers for making a megakernel's synchronization cheaper than a kernel boundary on MI300X

Status: research, no measurements yet · Date: 2026-09-26 · Sources: LLVM AMDGPUUsage "Memory Model GFX942" (main, 2026), AMD Instinct MI300 CDNA3 ISA Reference Guide (561 pages), ROCm 7.2.4 device libraries (`ockl.bc`), ETX measurements of 2026-09-22/26

## 0. The problem in one line

With vLLM's own kernels, ETX fusion is 7-16% slower than separate launches of the same kernels.
Each token crosses 310-472 device-wide synchronizations. Each ETX device event costs about 1.2 us more than a kernel boundary.
Fusion wins only if synchronization costs less than a boundary or overlaps other work.

## 1. What a device-scope event costs today, and why

On gfx942, each XCD has its own 4 MB L2.
Ordinary device memory does not provide coherence between these L2 caches.
An agent-scope release/acquire pair therefore uses these GFX942 code sequences from AMDGPUUsage:

| Side | Sequence | What the hardware does |
|---|---|---|
| producer release | `s_waitcnt vmcnt(0)`, `buffer_wbl2 sc1` | writes back every dirty line of the producer XCD's L2 |
| arrive | atomic add, agent scope | performed at the L2 that owns the line<br> bypasses L1 |
| consumer wait | atomic/`sc1` load poll, `s_sleep` backoff | one round trip per poll, 64-8128-clock sleep granularity |
| consumer acquire | `s_waitcnt vmcnt(0)`, `buffer_inv sc1` | invalidates the non-local lines of the consumer XCD's L2 |

The 2026-09-22 calibration measured a one-way cross-XCD counter cost of about 0.74-0.9 us.
A whole-GPU counter barrier with 304 workgroups cost 8.6 us.
A kernel boundary performs similar writeback/invalidation through the command processor once per dispatch, instead of once per workgroup that waits.

## 2. The levers

### 2.1 Coherence by memory type: keep the communicated data out of L2 (highest expected payoff)

AMDGPUUsage: "PCIe access from the GPU to the CPU can be kept coherent by using the MTYPE UC (uncached) which bypasses the L2", and local-memory lines under MTYPE RW are invalidated by probes when another L2 writes them. The memory type is a page attribute, set at allocation (`hipExtMallocWithFlags(..., hipDeviceMallocUncached)`, or fine-grained allocation), so **the imported kernels do not change**.

- Put the small tensors that cross XCDs in uncached memory: activation vectors (KBs per layer), the ETX event words, and possibly the attention partials.
- Keep weights in ordinary cached memory.
  No step writes them, so they need no writeback or invalidate.
- The device-scope release then needs only `s_waitcnt vmcnt(0)` (an uncached store is complete once acknowledged), and the acquire only the L1 invalidate (`buffer_inv sc0`). Both whole-L2 operations disappear from every event.
- It also makes the relay's hierarchical acquire safe: one poller per XCD, with only an L1 invalidate at each consumer.
  This acquire was unsafe on 2026-09-24 because L2 could retain stale lines.
  Uncached data cannot leave such lines in L2.
- Cost: accesses to those buffers go to HBM (about 350 ns instead of about 100 ns for L2). They are vectors of a few KB read once per phase, so the latency matters and the bandwidth does not. The KV cache is the open question: one row is written per token and 1024 rows are read, so it probably stays cached, with the new row made visible by the cache-write kernel's event.
- In ETX this is a machine-model change: a DEVICE-scope lowering row for "coherent memory" (release `s_waitcnt vmcnt(0)`, acquire `buffer_inv sc0`), chosen when every tensor an event protects is allocated uncached.

**Prior measurements that lower the expectation (fleet-mi300x, same GPU type, found 2026-09-27; this section was written without them).** `fleet-mi300x/results/microbench_summary.txt` and `docs/STATUS.md`:

| fleet measurement | result |
|---|---|
| (a) cross-XCD event round trip, counters in device memory vs uncached | 1.40 vs 1.44 us: uncached counters are not faster |
| (d'') payload in uncached memory, **no fences at all** | 12.3 M of 16.4 M words stale (75%) |
| (d''') ordinary memory, every payload access an agent-scope atomic (L2 bypass), no fences | 0 stale |
| `--coherent-acts` (L2-bypassing activations, consumer L2 invalidate dropped, producer writeback kept) in the full decode | correct, 1.5-2% faster before the prologue loads were vectorised, within noise after |
| (e) cross-XCD event round trip while 302 workgroups stream HBM | 6.16 us (1.40 idle): under load the cost is memory contention, not the fence |

**Measured 2026-09-27:** `buffer_inv sc0` does not invalidate the vector L1 (outside threadgroup-split mode), so the protocol above read stale data too (`sync_bench` mode `uc`); with `buffer_inv sc1` (`uc1`) it is clean and the whole-GPU barrier costs 5.47 us against 7.62 us cached, still 3.4 kernel boundaries (1.6 us). Design doc 15.11. (d'') did not invalidate the consumer's L1 either, so it does not refute the protocol above (`bench/sync/sync_bench.hip` mode `uc` does invalidate L1 and counts stale words). But (a), `--coherent-acts` and (e) say the same thing: in a loaded persistent kernel the fences are a small part of an event's cost. Expect little from 2.1; the gate in `vm_next.sh` measures it before anything depends on it.

### 2.2 Stay inside one XCD (built, not measured)

XCD-local events need only `s_waitcnt vmcnt(0)` and `buffer_inv sc0`, because the L2 is shared. `ETX_VL_XCD` replicates the one-block launches per XCD so that the GEMVs wait on XCD-local events: device-scope events for Qwen3-8B go from 472 to 254. `ETX_VL_CHAIN` removes three more events per layer by running q/k norm, RoPE and the cache write as one task.

### 2.3 Barrier shape

- **Fan-in:** 304 atomics on one word serialise at the L2 atomic unit. ETX already combines per XCD (38 local arrivals, then one device arrival per XCD, "last-arriver flush").
- **Fan-out:** 304 pollers require 304 round trips per poll interval.
  ETX's P5 relay uses one poller per XCD and an XCD-local flag for other workers, which reduces the count to 8.
  The full acquire was neutral, and the cheap acquire was unsafe.
  With 2.1, the proposal becomes both safe and cheap.

### 2.4 Waiting: sleep, wake-up, priority

- `S_SLEEP n` sleeps 64 × n clocks (up to 8128). ETX uses `s_sleep 8` (512 clocks, about 0.2 us at 2.1 GHz), which is a floor on reaction time; a shorter sleep for the first polls, then backing off, trades memory traffic for latency.
- `S_WAKEUP` wakes the sleeping waves of the same workgroup early. Useful when one wave polls and the others sleep.
- `S_SETPRIO` (0-3) can lower the priority of polling waves so they do not take issue slots from compute on the same CU, and raise it for the wave that must react to the release.

### 2.5 Hardware global barrier (GWS): exists, unproven

The CDNA3 ISA has a Global Wave Sync unit with 64 resources.
It provides `DS_GWS_INIT`, `DS_GWS_BARRIER`, and the `DS_GWS_SEMA_V/P/BR/RELEASE_ALL` semaphores.
Hardware queues the waves, so they generate no poll traffic. But:

- ROCm 7.2.4's own grid sync (`__ockl_grid_sync`) takes a **software** barrier (atomics plus `s_sleep`) on gfx942 and gfx950 and uses GWS only for other ISAs.
- GWS has to be allocated to the queue by the kernel driver.

AMD does not rely on GWS on this chip.
It warrants one microbenchmark, but no design dependency.
GWS orders waves, not memory. Data still needs 2.1 or the L2 operations.

### 2.6 Overlapping the wait

- Our own tiles can issue their weight loads before waiting, since weights do not depend on the event (registers or LDS, not L2: L2 warming measured -3% on 2026-09-25 because it competes with the previous phase's streaming tail).
- Imported kernels permit no internal reorder, so overlap must come from the graph.
  Independent launches, such as q-norm and k-norm, can run side by side instead of in stream order.

### 2.7 Levers that do not help here

- Direct-to-LDS loads on gfx942 are dword-only and unswizzled.
- There is no TMA / `cp.async` equivalent on CDNA3.
- AGPRs are a partition of the register file, not extra capacity.
- The 256 MB MALL cannot hold 16 GB of weights. It is useful only for the small per-step tensors, which 2.1 routes around it anyway.

(See also the memory note on gfx942 latency-hiding limits.)

## 3. Proposed measurements (about 30 minutes of GPU time, as microbenchmarks in `bench/calib`)

1. The event round trip between XCDs: today's fences vs. uncached data with `s_waitcnt` + `buffer_inv sc0`.
2. The cost of `buffer_wbl2 sc1` and `buffer_inv sc1` as a function of how much of the L2 is dirty or cached (0 to 4 MB).
3. The kernel-boundary gap in a HIP graph for tiny kernels, which is the baseline to beat.
4. GWS: whether a cooperative launch gets GWS, and the latency of `DS_GWS_BARRIER` / semaphore release across XCDs.
5. Then `examples/vllm_llm` with the activation buffers uncached, combined with `ETX_VL_XCD` and the relay.

## 4. Recommendation

Try 2.1 first.
It alone removes whole-L2 work per event without changes to imported kernels and makes the relay usable.
It combines with the existing 2.2 and 2.3.
Measure the remaining time before GWS and priority adjustments.
