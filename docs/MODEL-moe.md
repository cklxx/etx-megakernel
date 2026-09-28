# Qwen3-30B-A3B with vLLM's kernels fused by ETX: prediction, gate, and what we do when it is wrong

Status: pre-registration, written 2026-09-27 before any measurement it describes. Measured values carry their source;
everything else is a prediction. The go/no-go thresholds and the alternative explanations are fixed here.

## 1. What vLLM runs (measured, rocprofv3 kernel trace, batch 1, 1024-token context)

`examples/vllm_llm/layer_trace.py results_2026-09-27/vlnext/moe/prof` (1107 decode layers with the same 17-kernel
sequence; a HIP-graph trace has no gaps, so a duration is everything a launch costs). Traced layer 105.9 us; the untraced
token (4.898 ms) scales it to 97.5 us, and 0.22 ms per token are outside the 48 layers.

| # | kernel (vLLM) | grid | traced us | role |
|---|---|---|---|---|
| 0 | Inductor `fused_add_rms_norm` (+ moe_forward entry) | 1 x 512 | 4.18 | pre-MoE norm |
| 1 | `wvSplitK` YTILE 1, 128 x 2048 | 304 x 1024 | 4.12 | router |
| 2 | `topkGating<8,128,4,16,64>` softmax + top-8 | 1 x 256 | 8.03 | routing |
| 3 | torch `fill` of `num_tokens_post_padded` | 1 x 256 | 4.52 | **constant at batch 1** (naive block assignment) |
| 4 | Triton `fused_moe_kernel` gate_up, 8 experts x 1536 x 2048 | 192 x 256 | 17.81 | 50 MB, 3.4 TB/s body |
| 5 | `act_and_mul` silu | 8 x 96 | 4.54 | |
| 6 | Triton `fused_moe_kernel` down, 8 x 2048 x 768, x routing weight | 256 x 256 | 9.87 | 25 MB |
| 7 | `moe_sum` | 1 x 256 | 4.44 | |
| 8 | `copyBuffer` (MoE output into the custom op's out tensor) | 1 x 512 | 4.60 | **no model work** |
| 9 | Inductor `fused_add_rms_norm` | 1 x 512 | 4.23 | input norm |
| 10 | `wvSplitK` qkv 5120 x 2048 | 304 x 1024 | 7.34 | 21 MB |
| 11-12 | Inductor q/k norm, rope | 2 x 256 each | 4.40, 4.39 | |
| 13 | `reshape_and_cache` | 1 x 64 | 4.97 | |
| 14-15 | paged attention ll4mi mfma16 + reduce | 20 x 256, 32 x 128 | 7.54, 4.09 | |
| 16 | `wvSplitK` o_proj 2048 x 4096 | 304 x 1024 | 6.86 | 17 MB |

14 of the 17 launches are under 8 us (70 us of the 106). Bandwidth floor of the token: about 6.2 GB / 4.3 TB/s = 1.45 ms,
against 4.898 ms.

## 2. The model

vLLM layer = sum over the 17 launches of (F + b_k), with F = 1.6 us (kernel boundary inside a HIP graph, measured
2026-09-27) and b_k the body. The ETX graph runs the same kernels as tasks:

- the fill and the copy leave the critical path (the fill's value is a constant; the copy becomes an alias): 15 tasks;
- moe_sum -> norm and q/k-norm -> rope -> cache run chained in one workgroup (as `ETX_VL_CHAIN` does for the dense
  models): 12 hand-offs;
- the one-workgroup kernels between GEMMs (norms, top-k, silu, moe_sum) run once per XCD on per-XCD copies and feed the
  next GEMM's tasks of that XCD (the `ETX_VL_XCD` pattern), so about half of the hand-offs are XCD-local;
- ETX layer = sum of b_k over the 15 kept + 12 h + 15 p, where **h** is the hand-off (event release -> consumer body
  start) and **p** what the imported form adds per task (section 3, P-alternatives).

`examples/vllm_llm/moe_predict.py` (ms/token, vLLM 4.898):

| h \ p (us) | 0 | 0.5 | 1.0 | 1.35 |
|---|---|---|---|---|
| 1.0 | 3.92 | 4.28 | 4.64 | 4.89 |
| 1.2 | 4.04 | 4.40 | 4.76 | 5.01 |
| 1.6 | 4.27 | 4.63 | 4.99 | 5.24 |
| 2.0 | 4.50 | 4.86 | 5.22 | 5.47 |
| 2.5 | 4.78 | 5.14 | 5.50 | 5.76 |

Break-even: 12 h + 15 p = 32.4 us per layer (p = 0.5 needs h < 2.07; h = 1.2 needs p < 1.20).

What h and p are today, from the dense Qwen measurements (ETX_EV_STRIDE 1, design doc 15.10-15.11), with N hand-offs
per token and ETX fused - vLLM = N (h - F + p):

| build | N | fused - vLLM (ms) | h + p (us) | p from unfused - vLLM | h (us) |
|---|---|---|---|---|---|
| 1.5B base | 312 | 0.631 | 3.62 | 0.81 | 2.8 |
| 1.5B xcd + one block per slot | 284 | 0.311 | 2.70 | 0.48 | 2.2 |
| 8B base | 472 | 1.049 | 3.82 | 1.35 | 2.5 |

The hand-off microbenchmark says h should be 1.0 (XCD-local, padded counters) to 1.1-1.4 us (device-wide) (hw_bench
2026-09-27). **At today's h and p the MoE build loses (about 5.1-5.8 ms); it wins only if both come down.**

## 3. Why h and p may be what they are: alternatives, decided now

**H-alternatives (h = 2.2-2.8 us measured vs 1.0-1.4 in the microbenchmark).**

- **H1, false sharing of event words (predicted main cause).** The Qwen graph's events are single words at consecutive
  offsets (plan.json: `E_n` at word n). With stride 1, 32 events share a 128 B line: while the 304 producers of `E_n`
  decrement it, the ~303 workers already waiting for the next grids poll `E_n+1`, `E_n+2` on the same line. This is the
  DeepSeek C7 finding (3.3 -> 1.0 us) in another form, and the microbenchmark (one event, 296 -> 296) does not
  reproduce it. `ETX_EV_STRIDE` 32 (now the gfx942 default) removes it. Signature: stride 32 lowers h to ~1.0-1.4 us in the
  trace; the fused builds gain about N x (1.0-1.5 us): 0.3-0.5 ms (1.5B), 0.4-0.7 ms (8B).
- **H2, the device acquire invalidates L2**, so the consumer's first reads (activations, and its own argument block)
  come from the Infinity Cache or HBM. vLLM pays the same at a kernel start for data, but not for arguments (P1).
  Signature: body start -> first data load in the trace is long; the XCD-local hand-offs (acquire without L2 invalidate)
  are fast.
- **H3, ramp:** after the flip the 304 waiters wake up spread (s_sleep backoff); DeepSeek measured sleep 0/1/8/32 without
  effect, so predicted small. Signature: consumer start spread (first -> last) above 0.5 us.
- **H4, release cost:** each producer's `buffer_wbl2 sc1` walks its XCD's L2; 38 producers per XCD serialise (5.3 -> 1.4
  us with 64 KB dirty, padded). Signature: last producer body end -> event flip longer than 0.5 us.

**P-alternatives (p = 0.5-1.35 us per task: the imported code launched unfused is slower than vLLM's own launch).**

- **P1, one more dependent round trip for arguments:** an imported task reads `ctx.args[0]` (global), then its argument
  block (global), then its data; vLLM's kernel reads its kernarg segment, then data. After a device acquire both loads
  miss L2 (H2). Predicted 0.3-0.8 us. Fix to try: load the argument block before the event wait (it is written by the host
  before launch, so it does not depend on the event) into the scalar cache, which `buffer_inv` does not invalidate.
- **P2, the workgroup shape of small kernels:** every task is a 1024-thread workgroup at 128 VGPRs; a 64-thread
  `reshape_and_cache` or a 16-thread-block `rms_norm_3d` runs as slots with slot barriers. Unfused, this also means a
  1024-thread dispatch per launch. Signature: p concentrated in the one-block kernels; the "one block per slot" builds
  (1.5B: p 0.81 -> 0.48) move it.
- **P3, the megakernel's own tiles** (embedding/prologue, argmax) against vLLM's torch kernels: a fixed per-token cost,
  not per task. Signature: shows up as a constant in the unfused - vLLM difference, not per launch.
- **P4, code generated differently** from the same IR (register allocation at 128 VGPRs, the volatile thread-id asm, the
  LDS arena): compare per-kernel durations of the ETX unfused launch and vLLM's launch of the same kernel.

## 4. Measurements and the gate (one GPU session, dense first, then MoE)

| # | measurement | predicts / decides |
|---|---|---|
| G1 | dense 1.5B and 8B, fused, stride 32 vs `-DETX_EV_STRIDE=1`, base and xcd builds, alternated | H1: 1.5B xcd_s1 1.95-2.25 ms (was 2.248); 8B xcd 5.35-5.86 (was 5.864); base builds gain 0.3-0.7 ms |
| G2 | ETX_TRACE of 1.5B xcd_s1 fused, stride 32; per hand-off: last producer end -> first consumer start, spread, body vs vLLM b_k | h, and H2-H4 by signature |
| G3 | rocprofv3 kernel trace of ETX unfused 1.5B, per kernel against vLLM's trace of the same kernel (have) | p, and P2-P4 by where it sits |
| G4 | P1 fix (argument block loaded before the wait), fused, stride 32 | P1: gain = N x (0.3-0.8 us). **Not in this session**: built only if G3 puts p in the argument round trips (the scalar-cache route is not certain to survive the acquire) |

**Gate:** with h and p from G2-G4, `moe_predict.py` must give the MoE build below 4.65 ms (5% under vLLM). If it does,
the MoE imports (section 5) are validated and timed in the same session. If not, the session stops after the dense
measurements, and the residual h or p is the next work, not the MoE build.

**MoE correctness:** every imported launch of layers 0-1 byte-identical to vLLM's kernel on the same inputs (`--check`);
the whole model 32/32 tokens equal to HF greedy. **MoE result:** ms/token alternated with the ETX unfused build of the
same graph and against vLLM 4.898 (and batch 2 / 4: 5.724 / 7.100 ms, not a target of this build).

## 5. Import work (local, before the session)

As in `docs/RESEARCH-moe-import.md` section 3: `topkGating` and `moe_sum` are HIP kernels (manifest entries);
`fused_moe_kernel` is Triton, two specialisations (gate_up: `MUL_ROUTED_WEIGHT=False, top_k=8`; down: `True, 1`;
`BLOCK_SIZE_M/N/K = 16/64/128`, 4 warps, 20 KB dynamic LDS, naive block assignment), taken from the Triton cache vLLM
wrote (`results_2026-09-27/vlnext/moe/triton`, LLVM 23 IR). Inside the megakernel it runs one program per workgroup
(slots = 1): four programs per 1024-thread workgroup would need 80 KB of LDS and would put 192 programs on 48 CUs.

## 6. Local status before the session (2026-09-27)

- **Imports:** `topk_softmax_e128`, `moe_sum_k8` (HIP, `etx.importer.vllm_kernels`); `fused_moe_up` / `fused_moe_down`
  (Triton, `etx.importer.triton_import`, from `examples/vllm_llm/triton/`: the batch-1 cache entries, gate_up `top_k = 8`,
  down `top_k = 1` with the routing weight; both 4 warps, 20 KB LDS, argument lists checked against the TTIR signature).
  Triton's IR is LLVM 23; three constructs had to be rewritten for ROCm 7.2.4's LLVM 22 (special float literals such as
  `bfloat +nan(0x3F)` to hex, the `nocreateundeforpoison` attribute, debug info), plus the `amdhsa_code_object_version`
  module flag (Triton v5, hipcc v6).
- **ROCm 7.2.4 compiler on the laptop:** `examples/vllm_llm/rocm_build_check.sh` runs build.sh's device half with the
  VM's exact clang (rocm-llvm 7.2.4 in an x86 container). It reproduces the VM's register report (Qwen3-8B at commit
  7d8f874: 128 VGPR, 12 B/lane, 2 spills, as the VM printed).
- **Found with it:** at `ETX_EV_STRIDE` 32 the event offsets no longer fit the load/atomic immediate and LLVM hoisted
  every event address out of the persistent loop: Qwen3-8B megakernel 2 -> 35 VGPR spills, MoE 21. Event word indices
  now pass through an empty asm (`etx_evw`, abi.h; `-DETX_EVW_FOLD` restores the folded form): Qwen3-8B 0 spills,
  Qwen2.5-1.5B xcd 0, MoE base 1, MoE xcd 5 (DeepSeek's fleet-tile build was not affected: 256 VGPR + AGPRs, 0 scratch).
  Stride 1 vs 32 in the session is therefore measured with the fix in both builds.
- **Graph:** vLLM's 17 launches per layer minus the fill and the copy. With `ETX_VL_XCD` (the schedule this work is
  about): top-k and moe_sum run once per XCD, and XCD x runs expert slot x of gate_up (24 of the 192 programs), of silu
  (row x) and of down (32 of 256), each waiting only on its own XCD's part of the previous launch. Per layer that makes
  6 XCD-local and 7 device-wide hand-offs (`tests/test_vllm_moe.py` checks the placement and the event scopes).
- **Session:** `examples/sessions/moe_gate.py` (G1-G3 of section 4, the MoE check, fused and unfused timing).

## 7. Reconciliation (2026-09-28, session moe_gate, 1x MI300X enc1-gpuvm002; results_2026-09-28)

Every timed run 32/32 tokens equal to HF greedy; every MoE launch of layers 0-1 byte-identical to vLLM's kernel,
the two Triton fused_moe launches against vLLM's own code object (`CHECK OK: 0 launch(es) differ`). ms/token, two
alternated rounds on one GPU:

| | stride 32 | stride 1 | unfused |
|---|---|---|---|
| Qwen2.5-1.5B base | 2.555 / 2.562 | 2.567 / 2.568 | 2.202 / 2.211 |
| Qwen2.5-1.5B xcd_s1 | 2.410 / 2.420 | 2.491 / 2.464 | |
| Qwen3-8B base | 6.094 / 6.112 | 6.134 / 6.166 | 5.737 / 5.734 |
| Qwen3-8B xcd | 6.085 / 6.099 | 6.127 / 6.135 | |
| Qwen3-30B-A3B base | 6.755 / 6.772 | | 6.209 / 6.233 |
| Qwen3-30B-A3B xcd | 6.633 / 6.659 | | |

vLLM (2026-09-27, another VM of the same type): 1.937 / 5.039 (custom_ops=all), 4.898 (MoE). The unfused builds
reproduce the previous VM within 1% (1.5B 2.19, 8B 5.68), so the vLLM numbers are comparable; the fused 1.5B xcd_s1 is
slower here (2.49 at stride 1 against 2.248), not explained.

| prediction | result | verdict |
|---|---|---|
| G1 / H1: stride 32 gains 0.3-0.7 ms on the base builds; 1.5B xcd_s1 1.95-2.25, 8B xcd 5.35-5.86 | base 0.5-0.8%, xcd 0.6-3% | **H1 refuted** for the vLLM-kernel graphs |
| MoE below 4.65 ms | 6.63 ms (xcd), 35% above vLLM | **gate failed** |
| G2: h is 2.2-2.8 us | `compare.py ledger`, MoE xcd layer: 13.7 us of hand-offs for 13 hand-offs (XCD-local 0.3-0.5, device-wide 1.5-2.1), ramp 4.5, tail 10.7, **bodies 102 us** | h is ~1 us: not the problem |
| G3: p 0.5-1.35 us per launch | `compare.py unfused`, per launch vs vLLM: MoE +42 us per layer (fused_moe gate_up +8.2, down +3.7, reduce +3.2, topk +2.6, qkv / o_proj / cache / attn +1.4-1.8); 1.5B +9.8 us (reduce +2.2, down / gate_up +1.7, attn +1.5) | **p is the problem**, and larger for MoE |

**Where p comes from, first part (found, fix not measured):** the imported fused_moe runs one program (4 waves) in a
16-wave workgroup, so its `s_barrier`s become `etx_imp_slot_barrier` (LDS counter + polling). Triton's K loop has 4
barriers per iteration: about 69 barriers for gate_up (K 2048) and 29 for down (K 768). The excess over vLLM's fastest
launch (gate_up 23.1 vs 14.2 us, down 12.1 vs 9.5) is 0.13 and 0.09 us per barrier, consistent between the two kernels.
Fix: hardware barriers in rounds (a wave that has finished its block, or has none, keeps executing `s_barrier` until
every active wave of the workgroup has finished), valid because the waves of one block reach the same barriers.
Predicted: fused_moe back to vLLM's time, about -11 us per layer, -0.5 ms per token; the other slot-barrier imports
(attention, reduce, rms_norm, q/k norm) gain too. What remains after it (topk, cache, reduce, the GEMV +1.4 us) is the next
measurement.
