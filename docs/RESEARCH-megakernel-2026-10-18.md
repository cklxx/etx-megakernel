# Research: megakernel compilation — Event Tensor vs MPK vs Ada-MK

Status: research, no measurements · Date: 2026-10-18 · Sources: arXiv 2604.13327 v2 (Event Tensor, full text + appendices), arXiv 2512.22219 v2 (MPK, full text), mirage-project/mirage `mpk` branch (commit ca58a02, 2026-10-16), arXiv 2605.11581 v1 (Ada-MK, abstract + §1–§4.2). **All performance numbers below are author self-measurements. No independent reproduction exists.** Each number carries its source section/table.

## 0. One-line positioning

Three solutions to the same problem. They differ only in **where the decision lives**: Ada-MK moves every decision to compile time (offline DAG search, zero runtime branches); Event Tensor makes dynamic synchronization semantics first-class tensors in the compiler (runtime instantiation); MPK keeps an in-kernel decentralized scheduler. For ETX batch-1 decode, the three lines converge on one conclusion — **you do not need a runtime scheduler**.

## 1. Three teardowns

### 1.1 Event Tensor: synchronization semantics as tensors

- **Abstraction**: an event = "a group of tasks finished" at SM granularity (§2.1). An Event Tensor = a multi-dimensional array of events, a first-class tensor in the compiler IR. The core novelty, in the authors' words: "elevating these primitives into first-class tensors within the compiler IR" (§1). The clever move: "who waits for whom" becomes a **tensor indexing problem**, which reuses symbolic-shape infrastructure. A dynamic batch B is only a symbolic dimension of the tensor. The dependency graph template is fixed at compile time and instantiated at runtime (§2.3, Figure 4).
- **Lowering**: each Event Tensor lowers to an integer tensor. `notify()` = atomic decrement, `wait()` = spin-wait to zero, "avoiding any dedicated runtime data structures" (§3.3). Runtime state is only integer tensors plus task queues.
- **Pipeline** (§3.4): input is a **hand-annotated** compute graph (operators already cut into CTA-level tiles, from the TVM-based DSL "Axe" or Triton; **not automatic PyTorch import**) → graph-level optimization → tile-level optimization → one of two scheduling transforms (static: host precomputes per-SM queues, round-robin, "For simplicity" per the authors; dynamic: lightweight on-chip scheduler, atomically pushes ready tasks when events hit zero) → prefetch rewrite → persistent-kernel codegen. §6 admits: the pass that auto-generates an ET task graph from a standard compute graph **does not exist today**.
- **Dynamics**: symbolic shapes + data-dependent event updates (top-k decides which grouping tiles update which experts' event counters, initialized at runtime) + data-dependent task triggering (exp_indptr prefix sums decide the triggered tile range) (§2.4). Two hard boundaries: the dependency chain must be **strictly feed-forward** (Attention→TopK→Grouping→GroupGEMM); under static scheduling, data dependence degrades to a global `E[0]` barrier (§3.1, an admitted crude fallback).
- **Numbers** (hardware: 8×B200 NVLink / single B200, Ubuntu 24.04, PyTorch 2.8.0, CUDA 13.0):
  - Communication overlap (GEMM+Reduce-Scatter / AllGather+GEMM): **1.40x** over cuBLAS+NCCL, 8192 tokens, TP=8 (§4.1, Fig 11/12). Baseline Triton Distributed v0.0.2-rc, which the authors admit is "not yet fully optimized for Blackwell" — **weak baseline**.
  - Full MoE layer: **up to 1.23x** at 1024 tokens, single B200, vs Triton 3.4.0 / FlashInfer 0.2.14.post1, Qwen3-30B-A3B (128 experts, top-8) (§4.2, Fig 13).
  - End-to-end decode TPOT: A3B batch-1 **1.48x over vLLM / 1.20x over SGLang**; Qwen3-32B batch-1 **1.15x** over vLLM (§4.3, Fig 14). Baselines vLLM v0.11.0rc2 / SGLang v0.5.3rc0 (both with CUDA Graph + torch.compile); prefill 512 + 100 generated tokens. Raw-kernel numbers (App D): 1.49x / 1.27x — **this is the comparable measure**; the end-to-end numbers mix in the other engines' CPU overhead.
  - TP=4: 0.99–1.06x vs vLLM, **slower than SGLang** (§4.3).
  - Scheduling ablation (§4.5, Tables 2/3): on MoE, dynamic is 1.08x vs unfused (+4.0% over static), but at batch-1 static wins instead (1.03 vs 0.95); **at TP=4, dynamic reaches only 0.82–0.89x of unfused** — pushing to remote task queues costs too much. A measured failure.
  - Warmup: ETC AOT 35 s / 0 captures vs vLLM 123 s / 67 vs SGLang 583 s / 51; offline compile 107 s (§4.4, Table 1).
- **Hardware assumptions**: Reduce-Scatter uses multimem (Hopper+ multicast, NVIDIA-only). The synchronization primitives assume cross-CTA atomic visibility under the NVIDIA memory model. The authors claim the IR is "not specific to any particular GPU generation" (§4) — true at IR level, false at lowering level.

### 1.2 MPK: tGraph + in-kernel decentralized scheduling

- **Abstraction**: tGraph — nodes alternate between SM-level tasks (compute/communication) and events (cross-task sync points). A task's out-edges connect only to triggering events, in-edges only to dependent events. Positioned as "a lower-level extension of CUDA Graphs" (§3): statically instantiated, explicit dependencies, SM granularity instead of kernel granularity. The MatMul→AllReduce example: each AllReduce task depends only on the one MatMul task that produced its input tile (§3, Figure 4) — the entire basis for compute-communication overlap.
- **Scheduling**: SMs are statically split into workers (whole SM, private task queue, FIFO) and schedulers (4 warps per SM, 16 warps total, polling event queues to dispatch tasks). **Decentralized**: "each scheduler assigns tasks using local state... avoids global coordination" (§6.1), with an unimplemented back door for global coordination. In-kernel scheduler overhead is **0.28%** of total runtime (§6.1).
- **Compiler/runtime split**: the compiler does operator decomposition (partitioned by output tensor, task count ∝ SM count by default), dependency analysis, **event fusion** (69,000–162,000 task pairs compressed into 1,142–2,366 events — **37–118× fewer sync points**, §6.7 Table 2), normalization (fan-in/out pressed to 1, <1% overhead), linearization (BFS so tasks triggered by the same event have contiguous indices; device memory down 4.4–5.9× dense / 15.0× MoE, §6.7 Table 2), per-task codegen (Mirage superoptimizer at thread-block level). The runtime does event-driven dispatch, JIT task load balancing, MoE rebalancing by meta-tensor, and continuous-batching request admission (a single in-kernel task).
- **Dynamics**: dynamic batch = **multiple tGraph versions at powers of two**, picked at runtime (§6.1) — specialization, not true dynamism. Data-dependent durations (attention) are marked JIT; downstream stays JIT until a global barrier (§5.2). MoE top-k becomes task chains + NVSHMEM all-to-all with hybrid rebalancing (§6.4). **Branching control flow is not discussed; tGraph is statically instantiated.**
- **Numbers** (hardware: single A100/H100/B200 mostly, multi-GPU only 8×H100 DGX; setup: fixed 64-token prompt + 1024 generated, greedy, offline batched with no request jitter, 4 warmup + median of 5 runs, Appendix A):
  - **1.0–1.7x** vs best baseline (§6.3, Figure 9). **1.7x is the peak: H100 + small model (Qwen3-0.6B/1.2B class) + batch-1.** Large models mostly land at 1.0–1.2x, several points tie at 1.0x. The authors' own summary: "improvements are most significant for smaller models and newer GPU generations".
  - Qwen3-8B/A100: 14.5→12.5 ms/token; authors estimate the hardware floor at ~10 ms (16 GB parameters / 1.6 TB/s) (§6.3). >10x vs native PyTorch (§6.3).
  - A3B MoE: 1.14x/1.07x/1.17x/1.18x/1.15x at BS 1/2/4/8/16 (§6.4, Fig 10). SGLang's gather preprocessing takes up to 11% of MoE time (§6.4).
  - 8×H100: up to 10x vs PyTorch; 1.1–1.4x vs SGLang/vLLM (§6.5, Fig 11).
  - Ablation (§6.6): cross-task pipelining 1.15–1.29x (B200, beats the cuBLAS baseline); compute-comm overlap 1.1x; launch overhead: 293 launches/token — on B200, eager 3.8 µs (1.1 ms/token) → CUDA Graphs 0.8 µs (0.2 ms/token) → MPK zero.
  - Baseline versions are **not disclosed in the text** (only MPK-side environment is given). The README's "1.2×–6.7×" has **no source** — treat as marketing.
- **Code maturity**: `mirage-project/mirage`, `mpk` branch. A runnable research system (993-line end-to-end qwen3 demo, OpenAI-compatible HTTP serving, CI on self-hosted B200 for correctness + perf comparison). But the paper's `torch.compile(backend=MPK)` **does not exist in the repo** (no backend registration in a full-tree grep). Real usage needs hand-written per-model graph code. Latest commit (2026-10-16) still fixes a "Hopper hang". **Most borrowable: `persistent_kernel.cuh` (2119 lines), the worker/scheduler dual-role scheduler.**
- **Hardware assumptions**: multi-GPU communication **hard-depends on NVSHMEM**. TMA is used only in Hopper/Blackwell fused gather-GEMM. "Porting only needs new per-task codegen" (§7) — but the Mirage superoptimizer + TMA have no AMD counterparts.

### 1.3 Ada-MK (short)

Baidu, arXiv 2605.11581 (2026-05-12). Opposite stance to the other two: **"under a fixed deployment configuration the optimal execution path is uniquely determined; all runtime dynamic decisions can move to compile time"** (Abstract). Offline search over a fine-grained PTX-level DAG in MLIR bakes the execution path in and **eliminates runtime branches entirely**. Two more contributions: a 3-D smem constraint model + K-dim splitting (peak smem −50%, §4.1); MegaKernel as a TensorRT-LLM plugin hybrid engine (native operators for prefill, MegaKernel for decode, §3). Numbers: **NVIDIA L20** (Ada, no TMA hardware, 128 KB smem — half of H100) single-batch throughput **+23.6% over vanilla TensorRT-LLM / +50.2% over vLLM** (Abstract); end-to-end latency −10% to −50% (§1); claimed first industrial deployment (Baidu ads). Its criticism of MPK is worth recording: MPK's Managed Pointer mechanism adds "runtime if-else branches on smem page state", which hurts instruction issue efficiency at ultra-low latency (§1). This is the front line of the "where does the decision live" dispute.

## 2. Head-to-head

| Dimension | Event Tensor | MPK | Ada-MK |
|---|---|---|---|
| Core abstraction | event (SM-level completion signal as first-class tensor) | tGraph (alternating SM-level task/event graph) | PTX-level fine-grained DAG (baked at compile time) |
| Where the decision lives | compiler (IR + transforms), runtime instantiates / light scheduling | compiler (graph building, fusion, linearization) + in-kernel decentralized runtime scheduler | **all compile time**, zero runtime branches |
| Sync semantics | counter to zero (atomic decrement + spin-wait) | reactive dispatch on event activation | static execution trace, no runtime sync decisions |
| Dynamic batch | symbolic shapes (one graph generalizes) | multi-version specialization (powers of two) | unsupported (fixed config is the premise) |
| MoE top-k | data-dependent event updates + task triggering (§2.4, strongest point) | task chains + meta-tensor runtime rebalancing (§6.4) | not discussed |
| Control-flow branches | unsupported (strictly feed-forward) | unsupported (statically instantiated) | unsupported |
| Frontend | hand-annotated graphs (TVM DSL); auto-import **does not exist** (§6 admits) | claimed torch.compile backend **absent from repo**; hand-written per-model graphs | parses fixed Transformer decoder structure (§3) |
| Scheduler overhead | static ≈ 0; dynamic has push/pop (0.82x failure at TP=4) | 0.28% (§6.1) | 0 |
| Best speedup number | 1.48x (A3B batch-1 decode vs vLLM, B200) | 1.7x (small model + H100 + batch-1 peak) | +50.2% (vs vLLM, L20) |
| Applicability of the numbers | single-GPU decode; multi-GPU dynamic failed | small models / new GPUs / small batches; large models 1.0–1.2x | L20 + Qwen + fixed batch |
| Hardware lock-in | multimem (Hopper+), NVLink (for comm numbers) | NVSHMEM (hard dependency), TMA (partial) | none (the counterexample: no TMA needed) |
| Code runnable | not evaluated | runnable research system (with gaps) | code not public (unverified) |
| Author-admitted weakness | centralized queue contention; static has no dynamic load support; TP=4 dynamic 0.82–0.89x | heaviest task locks whole-SM registers; per-arch codegen rewrite; decentralized optimality unproven | "unique optimal path" holds only under fixed config |

## 3. Conclusions per dimension

1. **Problem framing**: all three agree the bottleneck is **orchestration** (launch overhead + kernel-boundary sync + cross-operator pipelining), not tile FLOPs. They diverge on the second sentence: ET says "synchronization semantics lack a first-class abstraction"; MPK says "no automatic whole-model compilation system exists"; Ada-MK says "runtime dynamic scheduling's branch penalty is unacceptable under portability-efficiency tension". ETX's "orchestration is everything" position is compatible with all three.
2. **Core abstraction**: ET vs MPK are two solutions to the same problem — fine-grained inter-task synchronization. ET **tensor-izes** synchronization (reuses symbolic-shape infrastructure); MPK **graph-structures** it (tGraph + fusion/linearization compress the descriptors). Compiler/runtime split: ET pushes nearly everything into the compiler (after lowering, runtime is only integer tensors); MPK deliberately keeps an in-kernel runtime (pays 0.28% tax for execution-time uncertainty); Ada-MK keeps zero.
3. **Dynamics**: the three "dynamics" are different things — ET = symbolic shapes + feed-forward data dependence (strongest); MPK = multi-version specialization + duration uncertainty (most practical); Ada-MK = denies the need. **ETX's MoE top-k sits in ET's strongest quadrant.**
4. **Compilation pipeline**: no frontend is automatic — ET hand-annotates, MPK hand-writes per-model graphs (paper-vs-repo gap), Ada-MK parses fixed structure. "Fully automatic megakernel compiler" does not exist today. This is the shared unfinished item of all three. Do not treat it as solved.
5. **Honesty**: ET's 1.40x used a weak baseline (Triton Distributed not Blackwell-optimized, admitted); MPK's 1.7x is a small-model/new-GPU/batch-1 peak with undisclosed baseline versions; Ada-MK's production deployment claim has no independent verification. The most trustworthy numbers in all three are the **ablation splits** (ET §4.5 scheduling comparison, MPK §6.6 three-way split, Ada-MK's smem −50%), not the end-to-end multiples.
6. **Hardware assumptions**: ET's communication numbers bind to NVLink+multimem; MPK binds to NVSHMEM; Ada-MK proves no TMA is needed. **For MI300X**: all three sync primitives (spin-wait/atomic decrement, device-memory semaphores) assume cross-SM visibility under the NVIDIA memory model. The porting risk is in lowering, not in the IR. Communication layers (NVSHMEM/multimem) are entirely non-portable.
7. **Borrowability**: ET's event-tensor IR shape (turning a hand-managed barrier grid into tensor indices) > MPK's tGraph normalization/linearization algorithms (hardware-agnostic) + paged-smem idea > MPK's persistent_kernel.cuh (when batching comes) > Ada-MK's compile-time-freezing idea (directly guides the batch-1 route). Communication and codegen layers are not borrowable.

## 4. Decision intervals

- **Pick the ET-style abstraction** iff: dynamism = symbolic shapes + feed-forward data dependence (MoE top-k is exactly this); you want to reuse symbolic-shape compiler infrastructure; you accept single-GPU/small scale (dynamic demonstrably failed at TP=4); you can tolerate a hand-annotated frontend.
- **Pick the MPK style** iff: the target is whole-model + multi-GPU (more than one step); dynamism is mostly "execution-duration uncertainty" rather than "dynamic dependency structure"; you accept multi-version specialization instead of true dynamism; you need a runnable reference implementation to copy a scheduler from.
- **Pick the Ada-MK style** iff: the deployment configuration is fixed; the latency SLO is strict and admits no runtime branches; hardware resources are constrained. **ETX batch-1 decode matches all three** — of the three lines, this is the one that fits ETX today: with fixed batch, the optimal path is unique; static host-side orchestration to the end; no runtime scheduler.
- **Pick none** (keep hand-writing) when: the optimization target is already ceiling-proven single-step orchestration, and the compiler's abstraction tax (ET's lowering, MPK's 0.28% + register lock-in) exceeds the hand-management cost it saves. ETX is likely in this interval now — numbers first, abstractions later.

## 5. What this means for ETX (MI300X, batch-1, orchestration)

1. **The first-principles bottleneck is the same across all three**: ET §4.5's 1.08x (breaking the global barrier), MPK's 37–118× sync-point reduction, ETX's measured "in-task barriers repeatedly drain the pipeline" — all point at **sync granularity and barrier placement**, not tile FLOPs. The ceiling-driven methodology stays on this attribution path.
2. **Batch-1 needs no runtime scheduler**: ET's dynamic loses to static at batch-1 (§4.5 Table 2: 1.03 vs 0.95); MPK's 0.28% scheduling tax pays for duration uncertainty, and batch-1 decode tile durations are deterministic; Ada-MK's thesis directly supports the current scheduler-only route. **Do not add an in-kernel scheduler for "generality".**
3. **The IR idea worth stealing**: ETX hand-manages its barrier/event grid today — exactly the "scattered semaphores" pain the ET paper frames. Expressing that grid as an event tensor with symbolic shapes (or an equivalent task-event graph) and reusing existing shape/index infrastructure is the cheapest step from "hand-written megakernel" to "compiler" — an order of magnitude cheaper than porting all of MPK.
4. **No number extrapolates**: every speedup comes from B200/H100/A100/L20. There is no MI300X data anywhere. Communication-overlap numbers depend on NVLink/NVSHMEM/multimem. Quote hardware with every number.

## 6. Author-admitted limitations (separate)

- **Event Tensor**: centralized scheduling queue "acknowledge potential contention at scale" (§3.2); "Plain static scheduling does not naturally support dynamic workloads" (§3.1); frontend auto-generation is "we envision", i.e. absent today (§6); TP=4 dynamic measured 0.82–0.89x (§4.5 Table 3); the gap is attributed to engineering, not abstraction (§4.3 — **qualitative, not proven**); "merged into a mainstream open-source system" is unnamed (§1, **unverified**).
- **MPK**: heaviest task locks whole-SM registers (§7); per-architecture codegen rewrite (§7); decentralized optimality is "interesting direction for future work" (§6.1); `torch.compile(backend=MPK)` is in the paper but not in the repo; README "1.2×–6.7×" has no source; cross-node claims unverified (all experiments on one H100 DGX, §6.5).
- **Ada-MK**: validated only on L20+Qwen; "unique optimal path" assumes fixed deployment config — the premise fails under dynamic batch / multi-tenancy (**inference**, not the authors' words); industrial deployment claim unverified.

## 7. Open questions and decisive experiments (ordered by expected payoff × cost)

1. **MI300X cross-CU/chiplet atomic visibility + spin-wait cost microbenchmark** — payoff: decides whether ET/MPK-style sync primitives port at all; the prerequisite for every borrowing plan. Cost: days. **Do this first.**
2. **"Global barrier vs fine-grained events" ablation on ETX** (reproduce ET §4.5's methodology, not its numbers) — payoff: directly quantifies sync-granularity payoff and orders ETX optimization. Cost: medium.
3. **HIP port spike of the MPK scheduler skeleton** (port event counting + ring queue only, not codegen) — payoff: an MI300X reference point for scheduler overhead. Cost: medium.
4. **Confirm ET lowering primitives on ROCm** (do integer tensors + atomic decrement have matching semantics) — payoff: decides IR portability. Cost: low (docs + microbenchmark).
5. **Ada-MK-style "compile-time freezing" A/B**: current host static orchestration vs minimal in-kernel ready queue — payoff: verifies batch-1 needs no runtime scheduler. Cost: medium.
6. **Independently reproduce any paper's speedup number** — cost extreme (needs B200), payoff zero (not extrapolatable to MI300X). **Explicitly do not do this.**

## 8. Sources

- Event Tensor: arXiv 2604.13327 v2, full text read (incl. appendices A–E); experiments §4 (8×B200), §4.1–§4.5, Appendix D. arXiv lists "accepted in MLSys 2026" (verified via arXiv API).
- MPK: arXiv 2512.22219 v2, full text read (18 pages); experiments §6.1–§6.7, Appendix A; code mirage-project/mirage `mpk` branch (commit ca58a02, 2026-10-16): README, demo, CI, persistent_kernel.cuh read. Venue not annotated on arXiv (an "OSDI reviewed" claim could not be verified and was dropped).
- Ada-MK: arXiv 2605.11581 v1, abstract + §1–§4.2 read; §5–§6 not read in full (numbers per abstract/§1, marked partially verified).

**Honesty note**: all three papers are author self-measurements with zero independent reproduction. All three end-to-end multiples carry some "peak packaging" (ET's weak baseline, MPK's small-model peak and undisclosed baseline versions, Ada-MK's claimed production deployment). The hardest numbers in each are in the ablation splits. Any multiple quoted without hardware + baseline + batch is marketing repetition.
