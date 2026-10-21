# GLM parallel execution: capacity, communication and scaling

This is a design, not GPU qualification. It supersedes the cost-first ordering and performance extrapolations in `docs/glm/GLM-MULTIGPU-2026-10-21.md`. No TP or EP implementation or MI325X performance result is claimed.

## Evidence and accounting contract

**[E]** means an analytical estimate, configuration-derived quantity, proposed degree or schedule, not a measurement. **[U]** means unknown pending measurement or implementation. All numbers in the comparison tables and formulas below are **[E]**, including batch sizes, degrees, counts, capacities and calculated times; this label applies to every numeric cell. Source line numbers and document section numbers are locators, not measurements.

Measured historical inputs must retain their conditions:

- The recorded **8.61 us** counter barrier is an **intra-GPU** whole-grid barrier with **304** resident workgroups; **608** workgroups cost **17.30 us**. It is not a remote collective startup (`docs/core/ETX_Technical_Design.md:1261`; `bench/calib/phase_switch.hip:13-23,45-61`).
- The recorded **10.6 us** one-way remote flag observation was on **two MI300X** GPUs, not MI325X (`docs/core/ETX_Technical_Design.md:1287-1294`; `bench/calib/p2p_atomic.hip:10-19`). It is not an all-reduce measurement.
- The recorded peer-store rate printed as **285 GB/s** actually uses **GiB/s** arithmetic: the source allocates **256 MiB**, divides MiB by seconds and by **1024**, and times **workgroup 0**, not completion of all workgroups (`bench/calib/p2p_atomic.hip:22-26,45-50`). Conversion gives **306.02 GB/s [E]**. Use **285 GiB/s** only as a *historical timing proxy*, not a validated completed-transfer or collective rate. Original calibration stdout has not been located [U].
- Actual MI325X topology, completed-transfer rate, collective startup/tails and GLM stage times are **[U]**. Therefore the latency tables are sensitivity calculations, not predictions.

Comparison contract [E]: one decode step generates one token for each of global batch `B = 1, 2, 4, 8`; no prefill or speculative tokens. Count logical transmitted fabric payload once, not send plus receive. Headers, flags, routing metadata, count exchange, padding, retries and topology-dependent forwarding are extra [U]. `MiB = 2^20 B`; `GiB = 2^30 B`.

For BF16 hidden width `H = 4096`, ordinary activation `A(B) = 8192 B` bytes; four-stream mHC residual `R(B) = 32768 B` bytes. Weight budget `W = 305.71 GiB` is the requested rounded checkpoint inventory, not a claim that all tensors are FP8. Routed expert matrix inventory `X = 283.5 GiB`; remaining `N = W-X = 22.21 GiB`. A nominal **256 GB** card is conservatively `C = 256e9 / 2^30 = 238.418579 GiB`, not 256 GiB. Actual allocatable HBM must be queried [U].

Native cache at `maxlen = 1072` is `K = 1.595962524 GiB/request` [E], including recurrent KDA state and expanded FP32 MLA state. Capacity tables subtract cache but **not workspace, allocator/runtime reserve, staging copies, communication buffers or imbalance** [U]. They are optimistic budgets, not fit certificates. Whole-layer PP is especially not evenly divisible; exact staged rank bytes must replace `W/p`.

## Scheme comparison: pure schemes

### Fabric payload per global decode step [E]

| Scheme | GPUs | B1 MiB | B2 MiB | B4 MiB | B8 MiB | Communication phases on one request path |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PP2 | 2 | 0.031250 | 0.062500 | 0.125000 | 0.250000 | 1 forward seam |
| PP4 | 4 | 0.093750 | 0.187500 | 0.375000 | 0.750000 | 3 forward seams |
| PP8 | 8 | 0.218750 | 0.437500 | 0.875000 | 1.750000 | 7 forward seams |
| TP2 | 2 | 1.406250 | 2.812500 | 5.625000 | 11.250000 | 90 all-reduces; 180 dependent ring rounds |
| TP4 | 4 | 4.218750 | 8.437500 | 16.875000 | 33.750000 | 90 all-reduces; 540 ring rounds |
| TP8 | 8 | 9.843750 | 19.687500 | 39.375000 | 78.750000 | 90 all-reduces; 1260 ring rounds |
| EP2 | 2 | 2.625000 | 5.250000 | 10.500000 | 21.000000 | 42 dispatch + 42 combine all-to-all phases |
| EP4 | 4 | 3.937500 | 7.875000 | 15.750000 | 31.500000 | 42 dispatch + 42 combine all-to-all phases |
| EP8 | 8 | 4.593750 | 9.187500 | 18.375000 | 36.750000 | 42 dispatch + 42 combine all-to-all phases |
| SP2 reference | 2 | 1.406250 | 2.812500 | 5.625000 | 11.250000 | 90 reduce-scatter/all-gather pairs; 180 phases |
| SP4 reference | 4 | 4.218750 | 8.437500 | 16.875000 | 33.750000 | 90 pairs; 180 phases |
| SP8 reference | 8 | 9.843750 | 19.687500 | 39.375000 | 78.750000 | 90 pairs; 180 phases |

**PP:** `V_PP = (p-1) R(B)`. Add `4 B` bytes [E] of sampled token-ID feedback to the first stage if sampling is on the last stage; control publication/acknowledgement bytes are additional [U]. Layers and their entire expert sets remain local to their assigned stage: experts are not partitioned *within* a layer. No step above assumes a same-request next token can start before sampling the preceding token.

**TP reference:** column/row sharding with two hidden-width reductions per layer: `45 × 2 = 90` all-reduces. Ring aggregate bytes per all-reduce `2(t-1)A`, per-rank injection `2(t-1)A/t`; hence `V_TP = 180(t-1)A`. This assumes attention-head/KDA-head sharding, shard-compatible mHC, local combination of selected expert partials before the FFN reduction, and replicated hidden state at boundaries. Router, indexer, embeddings/head and sampling need a proven layout; any extra exchanges are [U] and excluded. FP32 reductions double the listed BF16 payload [E]; numerical acceptance may require them. This is a reference lower communication contract, not an implementation fact.

**EP reference:** token-owner attention/router/shared-expert work; uniformly placed routed experts; dispatch one hidden vector per selected expert and return one hidden result per expert. With top-8 and 42 MoE layers, `V_EP = 42 × 2 × 8 × A × (1-1/e) = 672 A(1-1/e)`. These are *expected* bytes over routing, not deterministic loads. All-to-all means sparse variable-sized exchange, not a dense equal-size transpose. Duplicate experts on one destination could share an input and locally combine outputs, reducing payload, but that is a different contract. A hotspot rank, not the mean, controls completion. Eight selected experts do not imply all eight ranks get work.

**SP distinction:** the table is a TP-associated activation-layout reference: replace each all-reduce with a reduce-scatter/all-gather pair, same ring payload, not additional traffic on top of that all-reduce. SP alone leaves weights replicated and **does not fit** this model on one card. New-token-row SP has only `B` rows: B1 cannot occupy multiple ranks. Context/KV parallelism is a different scheme: MLA needs query distribution and online-softmax/output reductions [U]; KDA historical positions cannot be split as independent attention rows because the recurrence carries state. Do not substitute prefill sequence-length scaling for decode B1 scaling.

### Synchronisation and historical-rate sensitivity [E]

Each entry below is `local-barrier sensitivity + payload proxy`, in microseconds. Multiply the payload coefficient by `B = 1,2,4,8` to obtain each batch. `8.61 × phase_count` is explicitly a **what-if substitution**, not measured remote startup. PP uses serial seam bytes; TP/SP uses balanced per-rank injection; EP conservatively serializes aggregate bytes because balance/overlap is unproven. These different schedules are stated, not conflated with a common bisection bandwidth.

| Scheme | 8.61-us phase subtotal | Payload at 285-GiB/s proxy: B1 / B2 / B4 / B8 us |
| --- | ---: | --- |
| PP2 | 8.61 | 0.107 / 0.214 / 0.428 / 0.857 |
| PP4 | 25.83 | 0.321 / 0.642 / 1.285 / 2.570 |
| PP8 | 60.27 | 0.750 / 1.499 / 2.998 / 5.996 |
| TP2 | 774.90 | 2.409 / 4.819 / 9.637 / 19.274 |
| TP4 | 774.90 | 3.614 / 7.228 / 14.456 / 28.911 |
| TP8 | 774.90 | 4.216 / 8.432 / 16.865 / 33.730 |
| EP2 | 723.24 | 8.995 / 17.989 / 35.979 / 71.957 |
| EP4 | 723.24 | 13.492 / 26.984 / 53.968 / 107.936 |
| EP8 | 723.24 | 15.741 / 31.481 / 62.963 / 125.925 |
| SP2 | 1549.80 | 2.409 / 4.819 / 9.637 / 19.274 |
| SP4 | 1549.80 | 3.614 / 7.228 / 14.456 / 28.911 |
| SP8 | 1549.80 | 4.216 / 8.432 / 16.865 / 33.730 |

A real ring startup model is `T_TP,comm = 180(t-1) alpha_round + V_TP/(t beta_endpoint)` [E]. Substituting 8.61 us **per round**, instead of per collective, gives startup subtotals **1549.8 / 4649.4 / 10848.6 us** for TP2/4/8 [E]. SP has the same number of ring rounds. Neither substitution measures `alpha_round` [U]. For pairwise EP exchange, up to `84(e-1)` rounds [E] may replace 84 phase startups; algorithm and concurrent endpoint limits remain [U]. The historical 10.6-us flag result cannot resolve this ambiguity. PP token-feedback startup must also be added [U].

### Per-rank capacity [E]

`KV/state budget = C - weights`; final columns subtract the native cache at the listed global batch. PP/TP assume perfectly divided state `KB/g`. EP uses token-owned attention/cache and busiest owner `K ceil(B/e)`; if attention/cache is replicated instead, subtract **KB**, not this smaller amount. SP rows conservatively retain full cache.

| Scheme | Weight GiB/rank | KV/state budget GiB | Free after cache B1 | B2 | B4 | B8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PP2 / TP2 | 152.855 | 85.564 | 84.766 | 83.968 | 82.372 | 79.180 |
| PP4 / TP4 | 76.428 | 161.991 | 161.592 | 161.193 | 160.395 | 158.799 |
| PP8 / TP8 | 38.214 | 200.205 | 200.005 | 199.806 | 199.407 | 198.609 |
| EP2 | 163.960 | 74.459 | 72.863 | 72.863 | 71.267 | 68.075 |
| EP4 | 93.085 | 145.334 | 143.738 | 143.738 | 143.738 | 142.142 |
| EP8 | 57.648 | 180.771 | 179.175 | 179.175 | 179.175 | 179.175 |
| SP2 / SP4 / SP8 alone | 305.710 | -67.291 | -68.887 | -70.483 | -73.675 | -80.059 |

`W_PP = W_TP = W/g`; `W_EP = N + X/e`. Replicated residuals, router weights, scales, uneven layers and unsharded output weights mean actual allocations exceed ideal TP/PP shares [U]. EP capacity asymptotes to its replicated non-expert floor; pure TP/PP have no such ideal floor.

## Practical combinations [E]

Degrees are explicit: `p` pipeline stages, `t` tensor ranks, `e` expert groups, `d` independent replicas; GPU count `G=p t e d`. B always means **global** batch. Empty DP replicas transmit nothing.

**TP+EP orthogonal grid:** each expert is tensor-sharded across `t` ranks; complete expert sets are split over `e` groups. Each token owner is itself a TP group. Dispatch replicated hidden inputs along each tensor lane; return lane-local expert partials to that lane's owner. Sum all selected and shared-expert partials locally *before* one FFN all-reduce at the token owner. Attention has the other per-layer all-reduce. This gives `V = 180(t-1)A + t·672 A(1-1/e)` and **90 all-reduces + 84 exchange phases** on a request path. The whole model's token-owned attention work is not duplicated across EP groups. This avoids assuming 8 separate serial expert all-reduces per layer. Routing decisions and reduction order must be proven numerically [U]. Overlapping TP/EP groups are a different design, not this formula.

**PP+TP:** each stage has a `t`-rank TP group. Boundary residual is replicated on those ranks; each rank sends its full residual to the corresponding rank of the next stage, in parallel. Thus `V = 180(t-1)A + (p-1)tR`; one path has **90 all-reduces + (p-1) seam phases**, not a fabricated broadcast. Cross-stage links must support the simultaneous lane sends [U]. A sharded boundary can lower seam bytes but adds layout requirements.

**DP around a fitting group:** no model-activation communication between replicas, only within each fitting TP/PP group. Linear aggregate payload uses global B; busiest group uses `b = ceil(B/d)` for latency/cache. Small B leaves replicas idle. Pure single-GPU DP is infeasible for this inventory.

| Combination | GPUs | Fabric MiB B1 | B2 | B4 | B8 | Request-path phases |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TP2+EP2 | 4 | 6.65625 | 13.3125 | 26.625 | 53.250 | 90 AR + 84 A2A |
| TP2+EP4 | 8 | 9.28125 | 18.5625 | 37.125 | 74.250 | 90 AR + 84 A2A |
| TP4+EP2 | 8 | 14.71875 | 29.4375 | 58.875 | 117.750 | 90 AR + 84 A2A |
| PP2+TP2 | 4 | 1.46875 | 2.9375 | 5.875 | 11.750 | 90 AR + 1 seam |
| PP2+TP4 | 8 | 4.34375 | 8.6875 | 17.375 | 34.750 | 90 AR + 1 seam |
| PP4+TP2 | 8 | 1.59375 | 3.1875 | 6.375 | 12.750 | 90 AR + 3 seams |
| DP2×TP2 | 4 | 1.40625 | 2.8125 | 5.625 | 11.250 | 90 AR per active replica |
| DP4×TP2 | 8 | 1.40625 | 2.8125 | 5.625 | 11.250 | 90 AR per active replica |
| DP2×TP4 | 8 | 4.21875 | 8.4375 | 16.875 | 33.750 | 90 AR per active replica |
| DP4×PP2 | 8 | 0.03125 | 0.0625 | 0.125 | 0.250 | 1 seam per active replica |

PP combinations also need token feedback, excluded as in the pure table. Sync counts describe one batch microstep; splitting B into microbatches repeats phase startups for each microbatch, even where aggregate payload is unchanged.

Hybrid latency sensitivity is fully specified by [E]:

- TP+EP: `1498.14 + [180(t-1)8192/t + t·672·8192(1-1/e)] B / beta` us, with `beta = 285·2^30/10^6` bytes/us. EP term is the same conservative aggregate-serialization proxy as above. Replace 90 TP phase startups with `180(t-1)` ring startups for the ring sensitivity.
- PP+TP: `774.90 + 8.61(p-1) + [180(t-1)8192/t + (p-1)t32768] B / beta` us. Seam payload conservatively serializes lanes; ideal independent links reduce that term by `t` [E]. Compute across stages remains serial for a request.
- DP×TP: `774.90 + [180(t-1)8192/t] ceil(B/d)/beta` us on the busiest replica; DP×PP2: `8.61 + 32768 ceil(B/d)/beta`, plus feedback [U]. These formulas provide B1/2/4/8 without mistaking global payload for the replica critical path.

| Combination | Weight GiB/rank | KV/state budget GiB | Free after cache B1 | B2 | B4 | B8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP2+EP2 | 81.980 | 156.439 | 155.641 | 155.641 | 154.843 | 153.247 |
| TP2+EP4 | 46.543 | 191.876 | 191.078 | 191.078 | 191.078 | 190.280 |
| TP4+EP2 | 40.990 | 197.429 | 197.030 | 197.030 | 196.631 | 195.833 |
| PP2+TP2 | 76.428 | 161.991 | 161.592 | 161.193 | 160.395 | 158.799 |
| PP2+TP4 / PP4+TP2 | 38.214 | 200.205 | 200.005 | 199.806 | 199.407 | 198.609 |
| DP2×TP2 | 152.855 | 85.564 | 84.766 | 84.766 | 83.968 | 82.372 |
| DP4×TP2 / DP4×PP2 | 152.855 | 85.564 | 84.766 | 84.766 | 84.766 | 83.968 |
| DP2×TP4 | 76.428 | 161.991 | 161.592 | 161.592 | 161.193 | 160.395 |

Hybrid weights `(N+X/e)/(pt)`; cache `K ceil(ceil(B/d)/e)/(pt)` [E]. These require divisible head/state layout and balanced layers, not merely changing a degree flag.

## Critical path and scalability ordering

No ordering below uses implementation cost. There is no single scalar ranking across capacity, latency and throughput.

| Objective | Scalability ordering and limiting term [E] |
| --- | --- |
| Model capacity | Fully sharded TP/PP and their hybrids first (`W/G` ideal); EP/TP+EP next because non-expert replication remains; SP/DP alone cannot fit this inventory. PP layer granularity eventually limits capacity balancing. |
| B1 latency | TP and TP+EP can shorten work *within* a layer; EP can shorten only routed expert work and is load-sensitive. PP stages only divide storage: one request still traverses all stages. SP new-token rows and idle DP replicas do not shorten that request. |
| Increasing service throughput | DP over the smallest efficient fitting group scales without inter-replica collectives; TP/EP groups improve shared weight reuse as batch grows; PP needs enough independent in-flight requests to fill stages. SP is a layout complement, not an independent capacity solution. |
| Larger-than-one-card, eight-GPU extensibility | PP+TP offers whole-model capacity and intra-layer concurrency; TP+EP targets routed-weight capacity and expert work; replicas are useful only after a fitting group is established. Choose topology and workload, not a universal eight-way degree. |

For PP [E], `T_request = sum_i C_i + sum_seams L_i + sampling + feedback`; it is **not** `C_model/p`. For equal stage compute `c` and `m` independent microbatches, ideal flushed pipeline time is `(m+p-1)c` before communication: utilization `m/(m+p-1)`. At sustained decode, bounded in-flight independent requests give utilization at most `min(1,B/p)` before imbalance/feedback. More stages add serial seams and fill/drain depth; same-request autoregressive dependence prevents imaginary token overlap.

For TP [E], `T = C_shardable/t + C_replicated + C_imbalance + T_collectives`; more ranks reduce shard compute but increase ring depth and may worsen small GEMV efficiency. For EP [E], `T = C_nonexpert + max_rank(C_routed) + T_dispatch/combine`; the non-expert term and skew cap speedup. In TP+EP, apply those terms to the orthogonal grid, not an unqualified division of all model time by `te`. PP+TP retains the sum over stages with TP compute/collectives inside each. DP step latency is the slowest active replica; throughput is total completed tokens per host-wall second under the same latency objective.

**Batch-1 choice:** the design candidate for minimum latency is **TP across a measured efficient group**, trying TP2/4/8 and comparing EP/TP+EP if expert work dominates. PP is the low-traffic capacity fallback, not the latency winner merely because it has one seam. A measured winner is **[U]**. TP wins over PP only when its compute savings exceed its extra communication and layout costs; current data cannot choose the TP degree.

**Batch-8 choice on eight GPUs:** the first throughput candidate is **DP4×TP2**, two requests per fitting group [E], compared against TP8, EP8, TP2+EP4 and PP2+TP4 at the **same global B8**. It wins only if four groups' aggregate completion rate exceeds the larger group's weight-reuse benefit while meeting the same latency bound. DP4×PP2 is a capacity-valid low-communication comparator, not evidence that PP compute accelerates. Measured winner **[U]**; declaring one from a local barrier and a peer-store proxy would fabricate evidence.

### Code requirements, after the scalability comparison

TP needs sharded tensor descriptors, head/state ownership, deterministic reductions and in-kernel collectives. EP needs expert placement, bounded variable routing buffers, dispatch/combine, shared-expert ownership and skew instrumentation. PP extension needs arbitrary stage ownership, explicit residual layout, token feedback and cross-request scheduling. DP needs request admission/cache isolation and end-to-end wall-time accounting. SP needs a compatible TP/context layout and numerical tests, not a new launcher per operation.

**Reject any scheme that forces per-op host dispatch.** Communication must be a dependency-aware operation inside the resident megakernel or a proven persistent communication service with bounded progress. A per-op HIP/RCCL launch chain may be a comparison baseline, but it is not the ETX execution design.

## GLM configuration and reproducible capacity basis

The following configuration-derived numbers are all **[E]**, not GPU measurements. The pinned source is `zai-org/GLM-5.3-Flash`, revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` (`docs/evidence/qwen-batch-glm53/glm-model-api.json:1`). This is the release checkpoint, not the synthetic module gate. Source locators are repo-root-relative.

| Configuration [E] | Value [E] | Source |
| --- | --- | --- |
| Hidden width / mHC streams | 4096 / 4; mHC enabled | `docs/evidence/qwen-batch-glm53/glm-config.json:14-26,182` |
| Layers / FFN split | 45; 3 dense followed by 42 MoE | `docs/evidence/qwen-batch-glm53/glm-config.json:184-240` |
| Dense / expert intermediate | 12288 / 2048 | `docs/evidence/qwen-batch-glm53/glm-config.json:77,232` |
| Routed / selected / shared experts | 288 / 8 / 1 | `docs/evidence/qwen-batch-glm53/glm-config.json:235-239` |
| Routing | FP32 router, sigmoid, normalized selected weights, scale 2.5, one group | `docs/evidence/qwen-batch-glm53/glm-config.json:233-256` |
| KDA attention | 34 layers, 64 heads × 128 dimensions; convolution width 4 | `docs/evidence/qwen-batch-glm53/glm-config.json:126-166` |
| MLA attention | 11 layers at zero-based 3,7,11,15,19,23,27,31,35,39,43 | `docs/evidence/qwen-batch-glm53/glm-config.json:167-179` |
| MLA projections | 64 heads, Q LoRA 1536, KV LoRA 512, QK/V dimensions 256; RoPE dimension 0 | `docs/evidence/qwen-batch-glm53/glm-config.json:78,238-258` |
| Sparse indexer | 32 heads × 128, pooling 4, top-k 2048, tail always selected | `docs/evidence/qwen-batch-glm53/glm-config.json:21-26` |
| Vocabulary / embedding tying | 154880 / untied | `docs/evidence/qwen-batch-glm53/glm-config.json:254-259` |
| Model maximum positions | 1048576; not a native allocation guarantee | `docs/evidence/qwen-batch-glm53/glm-config.json:181` |
| Native default allocation length | 1072; runtime runner uses the same value | `examples/glm53/stage.py:41`; `examples/glm53/run_all.py:46` |

### What the requested weight total includes

The saved API tensor inventory (`docs/evidence/qwen-batch-glm53/glm-model-api.json:1`) gives [E]:

| Precision | Parameters [E] | Bytes [E] |
| --- | ---: | ---: |
| FP8 E4M3 | 314396639232 | 314396639232 |
| BF16 | 6926096640 | 13852193280 |
| FP32 | 295518 | 1182072 |
| Total | 321323031390 | 328250014584 |

Total is **305.70665 GiB [E]**, rounded to the requested **305.71 GiB [E]**. Calling all of it FP8 would be inaccurate. This inventory is not the exact staged text-only HBM footprint: native staging excludes vision/MTP, writes aligned matrix data plus explicit FP32 block scales and descriptors, and stages outer embedding/norm/head tensors (`examples/glm53/stage.py:1-5,97-123,193-200`). Only a validated stage manifest plus actual allocator telemetry can certify per-rank residency [U]. Do not equate download size, active bytes per token and resident allocation.

Routed matrices alone [E]: `42 layers × 288 experts × 3 matrices × 4096 × 2048 × 1 byte = 283.5 GiB`. Each expert has **25165824 matrix bytes [E]**, plus **6144 scale bytes [E]** for FP32 scales on `128×128` blocks. The rounded residual `22.21 GiB [E]` used above is an accounting bucket, not a proof that all excluded metadata and scale bytes are covered. Replace it with exact staged ownership before deployment. Shapes and scale ABI: `examples/glm53/stage.py:84-89,109-119,264-266`.

The runner's **1.72 [E]** weight-copy factor is a separate analytical traffic-budget line (`examples/glm53/run_all.py:52,415-438`). It is **not** a measured HBM residency multiplier and must not silently inflate or certify the capacity table.

### Native state is not compressed MLA latent cache

From `examples/glm53/stage.py:41-63`, the state element counts are [E]:

- `STATE_KDA = 3 × (64×128) × 4 + 64×128×128 = 1146880` FP32 values per KDA layer/request, or **4587520 bytes**.
- `STATE_MLA = maxlen × (2×64×256 + 2×128)` FP32 values per MLA layer/request, or **132096 × maxlen bytes**.
- Full native state `K(L) = [34×4587520 + 11×132096×L] / 2^30 GiB/request`.
- At `L=1072`, B1/B2/B4/B8 occupy **1.595963 / 3.191925 / 6.383850 / 12.767700 GiB [E]** before sharding. KDA state is recurrent and does not scale with context length; MLA state does. The tables do not substitute a shorter context's cache.

KV LoRA rank **512 [E]** does not mean the native cache has only that many values per token: it stores expanded K/V plus indexer state. A compressed-cache or context-parallel implementation requires a new numerical/memory contract and measurement [U]. Likewise a TP degree does not automatically shard all cache or scratch arrays. mHC residual ownership, recurrent KDA state, MLA indexer state and logits must be explicit in a rank layout before using the ideal divided-cache rows.

Current whole-layer partitioning is `examples/glm53/prepare.py:59-72`; staged validation insists on exactly two ranks with unique complete layer coverage (`examples/glm53/stage.py:224-226`). Actual staged bytes, layer balance, free-memory reserve and workspace must be checked per rank; passing the ideal arithmetic is not an allocation acceptance test.

## ETX evolution: from the current seam to an eight-GPU service

Stage numbers are ordering labels, not measured maturity scores. Proposed degrees, batch sizes and acceptance counts in this section are **[E]**; uncollected performance results are **[U]**. Capability claims below distinguish existing code from qualification. This is not authorization to provision GPUs.

### Existing execution, not a TP4 path

- **Whole-layer PP2 exists in code:** `examples/glm53/prepare.py:59-72` divides the routed layers between two ranks and assigns every routed expert of an owned layer to that layer's rank. This is commonly called replicated-expert PP because there is no expert-axis sharding; it does **not** allocate a second copy of every layer on both stages.
- **Complete-layer tiles, one workgroup per request/layer:** `examples/glm53/model.py:17-50` uses a `('B',)` device-call grid. This does not establish full-device GEMV efficiency. A peak bandwidth divided by a workgroup count is not an isolated-workgroup measurement.
- **Embedding, head and feedback are explicitly owned:** `examples/glm53/model.py:34-36,52-57`. The residual is a full mHC tensor (`model.py:32,43`), and state/scratch is per layer/request (`model.py:40-45`).
- **Exactly two visible GPUs, P2P plus native integer atomics:** `examples/glm53/model_host.hip:19,42-55`. Memory-fit telemetry exists (`model_host.hip:44-46,64-67`). There is no current TP4 launch mode.
- **One megakernel per rank per model step:** `examples/glm53/model_host.hip:64-80` resets both ranks before launch and waits for both streams. It is not a single process-lifetime kernel: host step/reset and token transfer still occur. Its reported step span excludes later result-copy work; serving wall time must include that work.
- **System-scope cross-device dependencies exist in the engine:** `etx/passes/plan.py:44-63,83-98,140-147`; host event allocation/launch support is `etx/runtime/include/etx/host.h:1-5,67-77`. These abstractions do not by themselves implement a collective or prove its progress under saturation.

### Capability ladder

| Stage | Capability and ownership | Code area (existing anchor → proposed extension) | Measurement that proves completion | What it unlocks |
| --- | --- | --- | --- | --- |
| S0 — qualify the current seam | Existing PP2, complete layers, unsharded expert sets, peer-visible residual and token feedback, one megakernel/rank/step | `examples/glm53/model.py:17-57`; `examples/glm53/model_host.hip:19-80`; `examples/glm53/prepare.py:59-72` | On the actual MI325X pair: pinned slice then full-model numerical checks, 32/32 greedy tokens per request at B1/2/4/8, allocated/free bytes per rank, both-rank completion and full host-wall step distributions. Existing CPU tests are not this evidence [U]. | A trustworthy capacity/correctness baseline and measured PP seams; no TP/EP claim. |
| S1 — expose within-rank parallelism and exact capacity | Replace a complete-layer single-workgroup bottleneck with head/tile decomposition while retaining the same numerical boundaries and persistent scheduler; make ownership/workspace explicit | `examples/glm53/model.py:30-50`; `examples/glm53/tiles.hip:13`; `examples/glm53/stage.py:97-123`; `etx/passes/plan.py:83-98` | Same-weight, same-numerics A/B against S0; per-phase span and HBM traffic, resident workgroups, spills/LDS, state/weight/workspace high-water marks, correctness under all batches. Reject a gain inferred from peak TB/s divided by workers. | A meaningful shardable compute baseline and a capacity planner for arbitrary rank layouts. |
| **S2 — production-shaped collective gate: WAIT FOR GPU MEASUREMENT** | Establish in-kernel all-reduce and sparse dispatch/combine progress and real endpoint/topology costs before choosing TP/EP implementation | `bench/calib/phase_switch.hip:13-23`; `bench/calib/p2p_atomic.hip:10-50` are historical probes, not the required benchmark. Extend peer/system-event contracts in `etx/passes/plan.py:140-147` with measured communication kernels. | At minimum a 4-GPU collective run, plus pair controls; B1/2/4/8 hidden/residual payloads, BF16 and required FP32 reductions, all workgroups/ranks complete before stopping the timer, concurrent readers/writers, sparse top-8 routes including hotspot/skew, median and tail, timeout and event-generation tests. Follow with 8-GPU topology qualification before claiming 8-way scaling. | Measured startup, completed payload bandwidth, skew and residency costs; chooses TP versus EP versus smaller groups. The old 2-GPU seam cannot pass this gate. |
| S3 — general pipeline capacity | Generalize whole-layer ownership and host ranks to PP2/4/8; preserve explicit feedback and schedule independent requests across stages | `examples/glm53/prepare.py:59-72`; `examples/glm53/stage.py:224-226`; `examples/glm53/model.py:34,54-57`; `examples/glm53/model_host.hip:19-29,64-80` | Per-rank staged bytes and allocation reserve; numerical equivalence across cuts; B1 confirms serial latency, B8 traces confirm cross-request overlap without same-request token causality violations; stage balance and bubble measurements. | Models exceeding a pair's capacity, low-traffic multi-stage execution, and a PP+TP outer structure. CPU layout work can proceed alongside S2; GPU performance claims cannot. |
| S4 — tensor and expert execution | Add a proven TP path for attention/dense/expert matrices and a token-owner EP path with bounded variable buffers, then the orthogonal TP+EP contract above | `examples/glm53/stage.py:71-94,244-269` → shard descriptors; `examples/glm53/model.py:38-50` → explicit collective dependencies; `etx/passes/plan.py:44-98,140-147` → rank groups/progress validation; native GLM tiles → shard operations | Must consume S2 measurements first. End-to-end correctness, deterministic routing/combine, per-rank bytes and expert counts, critical-path trace, deadlock/overflow refusal, same-kernel unfused comparison and best real serving-system comparison on identical inputs. | Within-layer latency scaling and routed-weight capacity without per-op dispatch. TP/EP speedup is not inferred from divided weights. |
| S5 — topology-aware hybrids and replicas | PP+TP for capacity plus within-layer speed; TP+EP where routed work dominates; DP replicas of the smallest efficient fitting group; request-local caches and admission | `examples/glm53/model.py:17-57` and generated rank ownership → multi-axis placement; `examples/glm53/model_host.hip:64-86` → multi-group orchestration and complete host-wall reporting | Fixed global B1/2/4/8, identical prompt/context/generated length, throughput plus request latency/tails, per-link contention, cache isolation and imbalance. Compare DP4×TP2, TP8, EP8, TP2+EP4 and PP2+TP4, not different global batches. | Efficient use of all eight GPUs for serving; replicas can be admitted independently rather than forcing every token through all GPUs. |
| S6 — oversized-model eight-GPU service | Deploy a model that cannot fit one card using the selected PP/TP/EP ownership; retain persistent computation, correct recurrent state, bounded queues and failure reporting | Existing stage validation (`examples/glm53/stage.py:205-231`) and host residency checks (`examples/glm53/model_host.hip:64-67`) → all-rank fit/progress certificates; model adapters must supply their own tensor semantics | Exact pinned model identity, capacity with workspace/reserve on every rank, long-running cache reuse, full reference correctness, no hidden host offload unless explicitly compared, same-kernel fusion and strongest-system baselines, sustained wall throughput/latency under the target workload. | An eight-GPU service rather than an eight-GPU launch demonstration. GLM already exceeds one card under this budget; larger models require their own ownership and numerical validation. |

The ladder is not an implementation-cost ranking. S1 removes a compute ambiguity; **S2 is the mandatory GPU-measurement stop before committing to S4's communication-dependent architecture**. S3's general capacity work is separable, but its speedups still require measurement. No number from S0 qualifies S2, and no synthetic module gate qualifies the full release checkpoint.

### Go/no-go rule and persistent execution invariants

For a candidate degree [E], measure `Delta_compute = T_local_baseline - T_sharded_compute` and `Cost = T_communication + T_layout + T_imbalance + added_host_cost`, on production-shaped data under concurrent compute. Proceed for latency only if `Delta_compute > Cost` with margin exceeding observed paired spread; otherwise retain the smaller fitting group or PP capacity path. Throughput choices must pass the same latency objective and compare global completed tokens per host-wall second. Values and margins are [U] until the measurement; no numeric win threshold is invented here.

The engine must preserve:

1. **Dependency semantics:** publish payload before the system-scope completion signal; consumers acquire before reading. Generation-tag/reset ordering prevents stale events. Progress must remain possible when compute occupies the device; no mutually waiting nonresident workgroup may be required to release a resident one.
2. **A bounded communication ABI:** explicit rank/group IDs, layout, dtype, reduction order, capacity/count bounds and scratch ownership. Sparse EP metadata is part of the real traffic report, even though excluded from the analytical vector tables.
3. **No per-op host dispatch:** graph/codegen schedules compute and communication inside each resident megakernel, or a bounded persistent service. A collective library requiring a host launch for every layer is a baseline, not an accepted ETX execution path. Reserving service workgroups must be included in measured compute cost.
4. **Autoregressive causality and isolation:** a request's next token waits for its own sample; only independent requests fill PP bubbles. KV/KDA state, routing buffers and return tokens cannot alias across requests/replicas.
5. **Qualification with matching identities:** pinned tensors, compiled options and rank layout tied to numerical results and timing. Tests under `tests/test_glm53_model.py`, `tests/test_glm53_preparation.py`, `tests/test_glm53_slice.py` and `tests/test_glm53_seam.py` provide local contract checks; they do not replace GPU evidence.

### Corrections to the earlier multi-GPU note

The earlier note is retained as a dated design, not silently rewritten as measured history. Its workgroup-divided bandwidth estimate does not measure single-workgroup throughput; its EP shortcut omits dispatch/combine, non-expert serial work and skew; its seam probe does not establish TP/EP collectives. Pipeline overlap of token `t+1` with token `t` is legal only for different independent requests, never for successive tokens of the same autoregressive request. Use the explicit contracts and evidence labels in this document for new decisions.
