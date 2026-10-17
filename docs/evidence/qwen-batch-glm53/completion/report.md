# Qwen batch and GLM native PP2 completion report

Implementation revision: `c906b3b` on `task/qwen-batch-glm53-multigpu` (parent `85900b2`; starting revision `5b9390f`). No merge or push.

## Qwen B2/B4/B8

Completed the standard local path first. The compiler now factors structurally identical large task dispatch cases into shared code and per-type integer tables. Numerical tile bodies, standard O3/NT/VECNORM flags, and correctness gates are unchanged. B2/B4/B8 completed ROCm 7.2.4 clang 22 device assembly, ROCm host device pass, and native host syntax within the existing 6.835 GiB VM configuration. No memory or security configuration was changed.

- B2: 120724 tasks, 872 task types, 194 imported instances per position.
- B4: 241448 tasks, 1744 task types, 388 imported instances per position.
- B8: 482896 tasks, 3488 task types, 776 imported instances per position.
- Each megakernel: compiler reports VGPR 128, scratch 32 B/lane, 7 VGPR spills; unfused VGPR 103, scratch/spills 0. These are compiler properties, not GPU timing/residency measurements.
- Old/new task/event/type/argument/static queue/task-input tables compared exactly equal.
- Existing local model/configuration matrix: **16/16 passed**.
- Complete runner asset manifest: all 20 required assets, recipes and source hashes; unbound status explicitly recorded until real staged inputs are supplied.
- Real GPU checks remain required: every imported launch at positions 0 and 1040, HF coverage, every request 32/32 in fused and unfused modes for both repetitions. No new GPU token/import result is claimed.

Commands executed:

```bash
bash examples/vllm_llm/batch_local.sh "$PWD/build/qwen-batch-complete"
# Serial equivalent of the 16-configuration local_matrix.sh default matrix:
bash build/qwen-matrix-complete.sh
.venv/bin/python -m etx.tools.gpu_jobs examples/sessions/qwen_batch_correctness.py --gpus 0 --logdir build/qwen-complete-dryrun --dry-run
```

The committed reproducible complete local entry point is `examples/vllm_llm/batch_preflight.sh`.

## GLM actual model path

Replaced preparation-only status with an executable native model implementation:

- Full ETX graph: embedding → 45 KDA/MLA and dense/MoE layers → mHC head collapse/norm/vocabulary head → reverse token feedback.
- Whole-layer PP2 ownership: rank 0 layers 0–23, rank 1 layers 24–44, 21 routed MoE layers each. Layer-local convolution, recurrent and attention/indexer state; actual `[B,4,4096]` BF16 boundary and int32 feedback.
- Native C++/HIP operators: BF16/F32/E4M3FN readers and block-scale FP8 decoding, RMSNorm, clipped SwiGLU, sigmoid/correction-bias top-k routing, shared/routed experts, mHC Sinkhorn, KDA/conv/forget/output gates, NoPE sparse MLA/indexer/kpool/tail/attention.
- Offline strict tensor stager: pinned config/revision, exact shapes and descriptor ABI, compressed FP8 payload/scales, source-shard and packed-blob hashes, native/HF names, safe HF symlink snapshots. No checkpoint download.
- Actual two-GPU launcher: bounded tensor loading, owner/free-memory/peer/atomic/residency checks, both-rank reset, per-step deadline/output, full prefill and 32-step free greedy decode.
- Independent offline HF oracle from the original snapshot; explicit BF16 dequantization of the same FP8 weights, eager operators, non-speculative 32-token decode, full logits/provenance retained.
- Strict qualification: source/generated/binary/input/reference identity before/after; exact token rows and ordered step coverage; two fresh processes; finite complete logits; no mismatch masking. Token equality does not certify all operator numerics.

Exact model pin: `zai-org/GLM-5.3-Flash@46a050bec13002ad29ee795c16e61cf4fa3c4a9c`; config SHA256 `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`.

Numerical policy is explicit: native FP8 **storage**, BF16 activations, FP32 accumulation, block decoding to BF16. Dynamic FP8 activation quantization is not emulated. The trusted HF oracle uses matching FP8-to-BF16 dequantization. This is a correctness-first model path, with no performance claim.

## Local GLM validation

```bash
bash examples/glm53/local_acceptance.sh "$PWD/build/glm53-acceptance-verified"
```

Completed:

- Same C++ numerical body as HIP tiles, independent HF GLM5Next reference: B1/B2 FP32, B4 BF16, B8 FP8→BF16, all 32-step exact greedy tokens (32/64/128/256 respectively, **480/480** total in the four persistent receipts).
- Per-layer numerical comparisons, KDA state comparisons and workspace/state canaries. Final BF16 B4 and FP8→BF16 B8 fixture logits were bit-equal to HF; final FP32 maximum logit error was 3.5763e-7.
- HF multi-token chunk prefill followed by 32 exact free greedy IDs, FP32 and FP8→BF16; BF16-conv regression also checked.
- Unique-score kpool selection beyond budget, SwiGLU clipping, stage corruption/descriptor bounds, source drift, fixture exclusion and full production-shape graph.
- **43 passed** in final focused suite (12.91 seconds in the acceptance script).
- Full pinned 45-layer/H4096/B8 graph: 384 tasks, two devices, SYSTEM/fine-grained boundary at layer 23; both device kernels and host device pass compiled at ROCm 7.2.4 O3.
- Both full-shape kernels also passed native LLVM 23 IR/ISA checks and native host syntax. Full-shape compilation uses no checkpoint weights and is explicitly non-runnable.
- Exhaustive native FP8 decode matched PyTorch for all **254 finite byte patterns** (two NaN encodings are rejected by staging); **10000/10000** BF16 rounding cases matched.
- Real model two-GPU session dry run passed, preserving `HIP_VISIBLE_DEVICES=0,1`.

## Failures and causal fixes

- Initial full suite during implementation: **2256 passed, 18 skipped, 1 failed**, 575.55 seconds. The sole failure was the producer archive fixture refusing an uncommitted tree. No remote host was contacted. Clean-tree rerun at `c906b3b`: **2280 passed, 18 skipped**, 619.87 seconds, exit 0.
- Initial deliberately tiny sparse budget produced exact zero-score top-k ties; different legal tie choices changed downstream results. Requested production context 1024+32 is below index_topk=2048; full decode fixtures use that regime. The independent over-budget operator test has unique scores. Hardware qualification still refuses any exact token mismatch.
- Added chunk-prefill regression found a real BF16-conv rounding error at decode step 22. Fixed conversion before SiLU according to the convolution weight dtype; both production FP32-conv and BF16-conv branches now pass.
- Initial CPU harness checks exposed tiny-fixture max-length and HF default grouped-mm alignment constraints; fixed fixture shape handling and explicitly selected the independent eager HF expert implementation. Their original failure logs are retained.
- Mac Bash empty-array `set -u` issue in the new acceptance wrapper was fixed and the complete script rerun.
- All failed logs remain in `build/`; selected failure logs and receipts will be retained with final evidence.

## Hardware-only limits and cost

The local host is Darwin/arm64. No runnable ROCm GPU or staged full GLM snapshot was supplied for this run; standard model/cache locations contain no matching snapshot, and the brief forbids creating paid resources or downloading very large weights without separate approval. Neither action was taken. New GPU correctness, real full-checkpoint stage/fit, peer visibility, full-checkpoint exact tokens/logits and performance are **unmeasured**. No hardware completion or production numerical certificate is claimed. GPU spending for this work: **$0**; local CPU/compiler time only.

All local implementation and acceptance paths are executable. On an authorized existing host with already staged assets, first run `examples/sessions/qwen_batch_correctness.py`; after its strict pass, run `examples/sessions/glm53_model.py`. Complete commands and conditions are in `docs/QWEN-BATCH-GLM53.md`.

## Final integrated local acceptance

- `.venv/bin/python -m pytest -q` at clean implementation commit `c906b3bf8013b66c29b0932d82384ba1d3c79fe9`: **2280 passed, 18 skipped**, 619.87 s, exit 0.
- Complete GLM local acceptance script: exit 0, including both full-shape rank compilers/host checks and **43 passed**.
- Qwen B2/B4/B8 exact ROCm/local host build script: exit 0; serial 16-build matrix script: exit 0.
- No uncommitted source changes occurred during the clean-tree suite. Final evidence is in `docs/evidence/qwen-batch-glm53/completion/`; runtime code is tested at `c906b3b`.
