# CUDA calibration bench ports (WP-19)

Status: **compiled, never run**. This means the six benchmarks have been
compiled, and only their CPU `--help` paths have been executed. No GPU
correctness, timing, residency, or forward-progress claim is made.

These are syntax-level ports of `bench/sync/sync_bench.hip`,
`bench/hw/hw_bench.hip`, and the four corresponding `bench/calib/*.hip`
sources. No generated AMD lowering header or HIP runtime is required.

## Build and CPU acceptance

```bash
# CUDA is an optional command prefix supplied by the caller (empty locally).
$CUDA bash bench/calib/cuda/build.sh
for name in sync_bench hw_bench atomic_pingpong queue_contention step_overhead p2p_sync; do
  build/calib_cuda/"$name" --help
done
```

The build script compiles all six binaries and never executes them. Set `NVCC`,
`CUDA_ARCH` (default `sm_90`), and `OUT` to override the compiler,
architecture, and output directory. Default compiler: `nvcc` on `PATH`.
Default output: `build/calib_cuda/`. CUDA 13.4 was used for local acceptance.
`--help` and `-h` return before any CUDA runtime call or GPU discovery.

## Correspondence and limits

The output format strings, column ordering, units, and `RESULT` prefixes are
retained to allow side-by-side tables. The warp reduction labels change to
32 lanes and eight warps for a 256-thread workgroup. Legacy labels such as
`wbl2+agent-poll`, `HIP graph`, `CUs`, and `LDS` remain table identifiers;
they do **not** mean those AMD instructions or resources exist on NVIDIA.

| Source operation | CUDA spelling / meaning |
| --- | --- |
| HIP allocation, copy, events, streams, cooperative launch, graphs | Corresponding CUDA runtime calls |
| HIP bf16 | `__nv_bfloat16` / `__bfloat162float` |
| 100 MHz wall timer | PTX `%globaltimer` in nanoseconds divided by ten; existing ns/us conversions retained |
| GPU cycle counter | `clock64()`; distinct from the global wall timer |
| Agent atomic load / store / add / subtract | Device-scope `atomicAdd(p, 0)` / `atomicExch` / `atomicAdd` / `atomicSub` |
| Device release | `__threadfence()` |
| XCD ID | One device-wide L2 domain, ID zero; SM ID separately uses PTX `%smid` |
| Wave64 shuffle | Full-mask warp32 shuffle, offsets 1 through 16; block reduction combines eight warp sums |
| AMD sleep | `__nanosleep()` (requires sm_70 or later) |
| Scalar/vector dependent global loads | PTX `ld.global.cg.u64`; NVIDIA has no AMD scalar memory pipeline |
| Coherent 32-bit loads/stores | PTX `ld.global.cg.u32` / `st.global.cg.u32` |

A fence is **not** a cache invalidate. The retained syntax for AMD invalidation,
writeback, uncached allocations, and cache-bit variants is not a semantic CUDA
implementation. Those experiment entry points are rejected before GPU discovery
rather than emitting misleading timing rows. Source comments below the port
header describe the original AMD experiment, including its historical results.

| Binary | CUDA experiment entry points |
| --- | --- |
| `sync_bench` | `boundary`: captured CUDA graph of tiny kernels. Ping-pong/cache modes, barrier/relay, `loadscope`, `fresh`, `gws`, and `all` are rejected (exit 2). GWS has no CUDA equivalent. |
| `hw_bench` | `launch`, `wg`, `clock`. All other modes, including `all`, are rejected (exit 2): their fixed eight-XCD placement, scalar/vector distinction, or cache invalidation semantics require a separate experiment design. Retained kernels are syntax-compiled only. |
| `atomic_pingpong` | DEVICE protocol only, same device-wide L2 domain. AMD INVL1/SC0 protocols and cross-XCD pairs are omitted. Discovery cannot promise placement on distinct SMs. |
| `queue_contention` | CAS versus tagged ticket pop, same ring layout, task counts, grid sizes, and throughput columns. |
| `step_overhead` | A, B0/B1/B2, C, D with original 608-worker / 256-thread shapes. All workers use one domain's slot counter. Cooperative residency must be qualified on the target GPU; runtime errors terminate instead of yielding timing rows. |

The non-`--help` paths require future GPU qualification under a deadline.
In particular, fixed launch sizes and polling protocols have not been checked
for residency or forward progress. The port does not establish equivalent
cache protocols, correctness, performance, or an ETX fusion result.

`p2p_sync` ports the consumer-device-memory case of `bench/calib/p2p_sync.hip`.
It retains the condition/location/payload/dev0/dev1 columns and us units. Each
initiator is measured separately, with request, reply and optional 4 KB payload
on its consumer; one-way is estimated as half the local-clock round trip, not
a timestamp difference across GPUs. CUDA system-scope atomic references use
release stores and acquire loads, with payload validation before acknowledgement.
Both peer directions must support access and native atomics. Host-pinned and
non-consumer flag placements from HIP are omitted. Peer capability alone does
not establish NVLink transport; confirm topology before interpreting GPU results.
The loaded case streams initialized 1 GB buffers on both devices for five
seconds; signalling polls also have a five-second total deadline and fail
without timing rows on timeout or payload mismatch. Load overlap and residency
still need GPU qualification. Status: **compiled, never run**.
