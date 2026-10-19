# Research: megakernel / persistent kernel — AMD hardware update

Status: research, no measurements · Date: 2026-10-18 · Sources: Kog AI monokernel blog (blog.kog.ai, 2026-05), AMD @ University of Tsukuba slides (2025-10-21), LLVM AMDGPUUsage "Memory Model GFX942", aicassindra MI300X deep dives (2026-10-06), arXiv 2511.08083 (HipKittens, MLSys 2026), SemiAnalysis SGLang/AMD report (2026-01), browser-verified ROCm/vLLM/SGLang ecosystem state 2026-10. **Kog numbers are vendor self-measurements: closed-source engine, self-designed 2B model, 8 GPUs. Not extrapolatable.** Each number carries its source.

Companion: docs/RESEARCH-megakernel-2026-10-18.md (Event Tensor vs MPK vs Ada-MK — all NVIDIA numbers, zero MI300X data). This document closes that hardware gap as far as public evidence goes.

## 0. One-line summary

The biggest 2026 event on the AMD side is **Kog AI running a monokernel at 3000 tok/s on MI300X** (8 cards, 2B FP16, batch-1) — the first public persistent-kernel datapoint on MI300X, directly quantifying launch and synchronization cost on AMD hardware. In parallel, AMD synchronization-primitive mechanics (atomics execute in Infinity Cache; cross-XCD plain writes need explicit coherence) now have multi-source cross-evidence, answering half of the open question in RESEARCH-sync-hardware.md.

## 1. Kog AI monokernel (2026-05, blog.kog.ai, vendor self-measured)

**Result**: 3,000 tok/s per request, 8× MI300X, FP16 2B self-designed model, batch-1; the same engine does 2,100 tok/s on 8× H200; claimed MBU ~36%. Boundary: self-designed architecture, 8 cards, **no large-model / single-card data, not extrapolatable**.

**Overhead quantified on MI300X** (directly usable by ETX):

| Overhead item | Measured | Source |
|---|---|---|
| kernel launch + cleanup | **~4.5 µs per launch** | blog body |
| 10 kernels/layer × 25 layers | 1,125 µs/token → ceiling ~890 tok/s | blog body (arithmetic) |
| grid barrier (topology-aware) | **~600 ns** | blog body |
| KCCL self-built AllReduce | **<3 µs** (vendor library ~8 µs) | blog body |
| sentinel polling instead of atomic counter | **7.8 µs → 0.9 µs**, sync still ~35% of token time | dev.to retelling |
| XCD-to-HBM placement latency skew | **up to 150 ns** + CU skew | blog body |

**Borrowable engineering moves**: IOD-aware buffer placement (put sync buffers on each XCD's own IOD's HBM); sentinel polling; handwritten HIP + CDNA ISA inline assembly (no PyTorch/Triton/RCCL general stack).

## 2. AMD synchronization primitives: first-hand mechanism evidence

| Fact | Source |
|---|---|
| On MI300, **atomic operations execute in Infinity Cache (MALL), not in L2**; L2 is private per XCD | AMD @ University of Tsukuba slides (2025-10-21) |
| Cross-XCD plain writes do **not** auto-invalidate the peer cache; explicit `buffer_wbl2 sc1` + `buffer_inv sc0 sc1` required | Chips and Cheese, citing LLVM GFX942 docs |
| 4 MB private L2 per XCD (16 slices, ~4.3 TB/s); misses travel the Infinity Fabric to the 256 MB Infinity Cache | aicassindra (2026-10-06) + ROCm docs |

**Update on the open question in RESEARCH-sync-hardware.md** (half-closed): ✅ mechanism confirmed — ET/MPK-style atomic decrement + spin-wait has a hardware landing spot (MALL atomics are globally visible), no NVSHMEM-style software layer needed; ⚠️ plain memory writes across XCDs are not auto-coherent — use MALL atomics or explicit wbl2/inv; ❌ still missing **measured latency numbers for generic atomic semantics** (Kog's 0.9 µs is their own polling system).

**Update on ET lowering to ROCm** (half-closed): integer atomics exist natively; lowering rules are now explicit (MALL or explicit coherence). Microbenchmarks still needed to close it fully.

## 3. MI300X chiplet topology: programming implications

8 XCD / 4 IOD, each IOD hangs 2 XCD + 2 HBM; CPX+NPS4 measured stream bandwidth 4,210 GB/s vs NPS1 4,010 GB/s (AMD data, via aicassindra); the `remap_pid_for_xcds` pattern exists; HipKittens (arXiv 2511.08083, MLSys 2026) proves modeling **both L2 and LLC levels** in task placement beats naive by 19%, up to 3× over Triton BF16 GEMM. **Implication: megakernel task placement must be chiplet-aware** — keep dependent chains on the same XCD to eat the 4 MB private L2.

## 4. vLLM / SGLang on ROCm: 2026 progress

- AITER preshuffled KV: decode +15–20%; handwritten-assembly `mla_decode_fwd` contributes 1.2–1.6x; 1.2–4.4x over legacy ROCm on Qwen3-235B (spheron, ecosystem reporting)
- SGLang MI300X throughput **doubled** 2025-12→2026-01 (SemiAnalysis, **independent third party**)
- HIP graphs: launch overhead ~2x improvement (CUDA graphs only ~6%)
- Independent ablation (8×MI325X): AITER +10% @conc1, +3–5% at high concurrency, but variance grows (hyperautomation)
- **Counter-example**: AITER **−0.6–2.6% on Gemma4-12B fp8** (xbill999, 2026-10-09, 9 cells×3 repeats, independent measurement) — untuned shapes go negative
- AMD official: Llama 3.1 405B FP8, MI300X 8-card vs H100 8-card geomean **1.08x**; DeepSeek-R1 5x vs H200 (claimed tier)
- **Honest annotation**: **no fused decode megakernel, no persistent kernel** — vLLM/SGLang on ROCm stay on the multi-kernel + graph route; thesoftwarefrontier warns that late-2025 AMD numbers measured the kernel library of that time — quote with version + date.

## 5. rocSHMEM and collective communication

rocSHMEM **3.6.0** exists and is in use (AMD, 2026-09-14, 8×MI300X MoE); PyTorch native AsyncTP for ROCm (message 12 days old, 4–33%, trunk only, `TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP`). **Conclusion**: NVSHMEM has an AMD-side counterpart but the ecosystem is catching up; ETX is single-card, **currently irrelevant**.

## 6. Kernel frameworks

HipKittens (tile abstraction is portable, instantiation must be rewritten for AMD; "peak AMD kernels are written in raw assembly"); AMD TileLang: FA on MI300X **0.36 ms** vs Triton 0.55 ms vs PyTorch 0.97 ms (b=1,h=8,s=4096,d=128) → 2.69x / 1.53x; AITER is a handwritten-assembly library, not a compiler. **For ETX**: when writing tiles (the MoE tile is the big one), HipKittens/TileLang fit better than Triton.

## 7. NVIDIA side 2026 H2 (one line each)

AutoMegaKernel (arXiv 2606.09682): agent-synthesized megakernels, honestly reported — int8 W8 beats cuBLAS-graph bf16 at batch-1 (1.08–1.33x), equal-precision bf16 **loses 1.13x**, GEMV reaches only 63% of HBM peak; UniEP (training-side MoE, 1.03–1.38x); RaMP (MoE inference, 1.30x over Triton FP8 on OLMoE); Emre worklog: handwritten megakernel 3.49 ms/token, **7x faster than vLLM eager but 1.65x slower than CUDA graphs** — beating eager is easy, beating graphs is hard.

## 8. Decision intervals (for ETX)

**Directly borrowable (start this week)**:
1. **Sentinel polling instead of atomic counters**: Kog 7.8→0.9 µs; every "wait for N tasks" counter in the ETX event grid is worth trying. One day to an ablation number.
2. **IOD/XCD-local sync buffer placement**: Kog holds 600 ns stable.
3. **Reconcile the ETX ceiling against Kog's breakdown table**: of the measured 4.18 ms, the 66% "fusion cannot remove" — map launch / HBM restart / intermediate-tensor spill / ~35% sync item by item, find what is uncovered.

**Design constraints (write into lowering rules)**: cross-XCD plain writes are not auto-visible — sync signals go through MALL atomics or explicit wbl2/inv; task placement chiplet-aware (dependent chains on the same XCD, eat the 4 MB private L2).

**Pitfalls**: Kog's 3000 tok/s is 8 cards + self-designed 2B, not comparable to single-card Qwen3-30B-A3B; AITER has no take-and-use (untuned goes negative); vLLM/SGLang ROCm numbers are not comparable to the ETX route; rocSHMEM is immature and unneeded on single card.

**Do not do**: reproduce Kog (closed source); wait for an official ROCm megakernel (none in sight).

## 9. Open questions update and new experiment order

| Previous open question | Status |
|---|---|
| Atomic visibility + spin-wait | **half-closed** (mechanism confirmed, generic-semantics latency numbers missing) |
| Global barrier vs fine-grained event ablation | untouched; Kog's 35% sync share is a reference |
| MPK scheduler HIP port spike | untouched; single-card part only |
| ET lowering to ROCm | **half-closed** (rules explicit, microbenchmark pending) |
| Compile-time freezing A/B | untouched; Ada-MK + Kog corroborate each other, priority can rise |
| Reproduce paper numbers | still do not (no reproducible object on the AMD side either) |

**New experiment #0**: in the existing ETX megakernel, replace one atomic-counter barrier with sentinel polling and measure the latency difference — Kog gives the 7.8→0.9 µs reference; if ETX reproduces the same order, sentinel polling becomes the main line of the sync design. Cost: one or two days.

**Honesty annotation**: Kog is the only AMD-side persistent-kernel datapoint, but closed-source + self-designed model + 8 cards — the numbers do not extrapolate. The sync-mechanism part has multi-source cross-evidence (AMD official + LLVM docs + Kog measurement), more credible than the performance numbers.

**What was not found** (recorded honestly): no AMD MI300X persistent-kernel paper on arXiv in 2026 (megakernel papers are all NVIDIA-side); no open-source AMD monokernel repo; no public measured latency for generic atomic semantics across XCDs. These three blanks are themselves the conclusion: the track is still virgin territory on AMD, and Kog is the only — closed — pioneer.
