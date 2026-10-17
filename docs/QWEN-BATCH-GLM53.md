# Qwen batch correctness, then GLM 5.3 Flash multi-GPU preparation

## Priority and evidence boundary

Qwen B2/B4/B8 functional support comes first. No performance tuning or new
performance claim is part of this work. GPU/cloud access and large weight
downloads are excluded. Local compilation cannot establish GPU correctness.

The starting tree is `b573664faa4e7988dff353bc03aca4ce8561a729`. Interrupted
metadata in `evidence/qwen-batch-glm53/` and the original scratch progress/logs
were preserved. The model's existing `ETX_VL_REQUESTS=N` path replicates request
graphs, shares weights, and isolates activations and KV caches. Column batching
(`ETX_VL_BATCH=1`) also exists, but is outside this qualification recipe. Historical
Qwen B2 results in `etx/ops/gemv.py` and `docs/RESULTS-2026-10-01.md` are distinct
from the completed DeepSeek B2/B4/B8 study and from this unexecuted GPU path.

## Qwen implementation

`examples/vllm_llm/batch_correctness.py` fixes an unchained, CHECK-enabled,
replicated Qwen3-30B-A3B recipe at B2/B4/B8, context 1024, 32 generated tokens,
two repetitions. `host.hip --check` now initializes and snapshots **every**
request, runs every prologue, and offsets instance launches by the number of
prologues. Every request's scratch and checked-layer KV buffers participate in
the original-versus-imported byte comparison. Chained launch tables are refused.
Any request's token mismatch in **any** repetition now returns exit code 6.

Qualification runs, in order, on the same binary:

1. All imported instances at positions 0 and 1040, byte-identical to originals.
2. Full teacher-forced HF prefill coverage for every request (existing margin
   rule: zero mismatches beyond HF top-1 margin 0.5), with an exact 32/32 tail.
3. Free-running fused decode, exact 32/32 for every request and repetition.
4. Free-running same-kernel unfused HIP graph, the same exact token gate.

It hashes packed weights, manifest, all used golden/prefill files, HF config and
checkpoint revision before and after qualification, and verifies the existing
build identity before and after. Missing, duplicate, wrong-mode, truncated or
failed output is refused. `correctness.json` schema 1 has `status` PASS or
NOT-COMPLETE, `performance: unmeasured`, batch/context/gen, checkpoint revision,
asset and build identities, checked instance counts, positions, modes and token
counts. Failure retains full logs and an incomplete receipt. Host timing text is
diagnostic only; the runner makes no timing verdict or baseline selection.

For B2/B4/B8 respectively, one successful mode must check 128/256/512 greedy
tokens (B × 32 × two repetitions); these are **required counts, not measured
GPU results**. Coverage also checks 1055 positions per request.

## Reproduction: staged inputs, no implicit downloads

Use an already available, pinned `Qwen/Qwen3-30B-A3B` BF16 snapshot. Put its
40-character checkpoint commit in `$MODEL/hf/REVISION`. Keep the original
snapshot/packing provenance; a revision text file alone does not authenticate
weights. The runner fingerprints inputs, rather than certifying their source.
Generate new references from that exact snapshot; do not substitute DeepSeek
goldens or reuse references from another checkpoint. The existing prep code
reuses golden files if present: use a fresh reference directory to regenerate.

```bash
MODEL=/absolute/path/to/packed-qwen
REF=/absolute/path/to/new-reference-directory
mkdir "$REF"
for r in 0 1 2 3 4 5 6 7; do
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python examples/llm/prep.py \
    --model Qwen/Qwen3-30B-A3B --src "$MODEL/hf" --out "$REF" \
    --context 1024 --gen 32 --rotate "$((97*r))" --golden-only
done
# Review/retain reference logs and copy the eight golden_*.txt and prefill_*.txt
# into MODEL without overwriting unrelated accepted evidence.
# If packing is needed, this uses only the already staged local snapshot:
.venv/bin/python examples/vllm_llm/prep.py --src "$MODEL/hf" --out "$MODEL" --maxlen 1072
```

Local qualification prerequisites (no GPU):

```bash
.venv/bin/python -m pytest -q
bash -n examples/vllm_llm/batch_local.sh
bash examples/vllm_llm/local_matrix.sh build/qwen-batch-matrix
bash examples/vllm_llm/batch_local.sh "$PWD/build/qwen-batch-rocm"
.venv/bin/python -m etx.tools.gpu_jobs examples/sessions/qwen_batch_correctness.py \
  --gpus 0 --logdir build/qwen-batch-dryrun --dry-run
```

`batch_local.sh` requires the cached `etx-rocm22` image and local ROCm headers,
device libraries and vLLM source. It runs the exact batch recipe through ROCm
7.2.4's compiler. The 16-build matrix separately covers the existing configurations.
If that container runs out of memory after emitting the plan/import bundle,
`bash examples/vllm_llm/batch_native.sh "$PWD/build/qwen-batch-rocm" 2 4 8`
checks those preserved plans with native LLVM. It remaps include paths only.
Each native batch has a 1200-second diagnostic budget (override with
`ETX_QWEN_NATIVE_SECONDS`), progress/probes, full logs and a completion receipt.
Native LLVM is an additional syntax/ISA check, not a substitute for the ROCm
compiler check or a predictor of its VGPR/spill allocation. Before any GPU
provisioning, complete the ROCm checks on a local environment with enough memory.

On an explicitly authorized, already prepared MI300X host, with vLLM source in
`$VLLM` or `~/vllm`, run the committed single session:

```bash
ETX_QWEN_BATCH_MODEL="$MODEL" .venv/bin/python -m etx.tools.gpu_jobs \
  examples/sessions/qwen_batch_correctness.py --gpus 0 --logdir results_qwen_batch_correctness
```

The session has a 7200-second cap; three CPU builds may overlap, qualifications
are serialized B2 → B4 → B8. Every child has a deadline, full unfiltered log,
30-second progress, and a process-state probe on 180-second silence. Quiet active
compilers retain their 1800-second hard build deadline; silence alone is not a
failure. A failed
batch blocks later qualifications. No resource rental or setup is hidden in it.
Estimate the GPU cost from the host's actual hourly rate before any authorized
run; no allocation or GPU money was spent for this preparation.

For direct reproduction of one batch (on that GPU host):

```bash
.venv/bin/python -m examples.vllm_llm.batch_correctness build --batch 2 \
  --model "$MODEL" --build "$PWD/build/qwen-b2" --out "$PWD/build/qwen-b2-logs"
.venv/bin/python -m examples.vllm_llm.batch_correctness qualify --batch 2 \
  --model "$MODEL" --build "$PWD/build/qwen-b2" --out "$PWD/results_qwen_b2"
```

Only after all correctness gates pass may a separate, authorized comparison use
the same-kernel HIP graph and a strongest-tested legal vLLM baseline. The current
runner has no performance winner. A vLLM comparison can reuse the eight distinct
files through `examples/llm/vllm_bench.py --goldens G0 G1 ... G7 --batches 2,4,8
--tokens 32`, recording version, engine configuration, every sequence's token
gate and host wall measurements. S3's historical lead is not a higher-batch
winner; select/validate a baseline on the actual session before comparison.

## GLM identity and compatibility

The exact public checkpoint is
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash).
The saved API metadata pins revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`;
its [pinned config](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/config.json)
has SHA256 `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`.
It declares `Glm5NextForConditionalGeneration`, 45 text layers (34 KDA linear,
11 sparse MLA), H=4096, mHC with four streams, 288 routed experts with top-8,
one shared expert, sigmoid/noaux_tc routing and native FP8 block quantization.
The first experiment is text-only, without the vision encoder or MTP.

The [official vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.3-Flash.yaml)
calls for newer GLM-capable vLLM/nightly builds and identifies AMD gfx950 support.
It does not establish the existing gfx942 Qwen importer/packer as compatible.
`examples/llm/model.py` now rejects both nested and flattened GLM configs with
an actionable compatibility message, before ordinary Qwen shape loading.
`python -m examples.glm53.prepare --require-model` emits the concrete missing
interfaces as JSON and exits 2 **without touching a GPU or fetching anything**.
An ordinary preparation invocation exits 0 while retaining `status: BLOCKED`.

All interrupted source files are unchanged. `source-verification.json` records
fresh checks: config and recipe remain byte-identical; the live model API has
mutable counters, while model ID, revision and dtype/parameter inventory agree.
The config was also fetched by its pinned revision and matched the saved hash.

## GLM PP2 partition and communication contract

The implemented offline partition assigns layers 0–23 to rank 0 and 24–44 to
rank 1: **21 MoE layers per rank**, with every expert of a layer on its owner.
Rank 0 owns embedding; rank 1 owns final norm and vocabulary head. KV, KDA
recurrence/conv and sparse indexer state stay with their complete layer. This
avoids expert all-to-all and tensor-parallel reductions in the initial design.
The split is a preparation choice, not a measured optimal partition.

The proposed boundary carries all four BF16 residual streams `[B,4,4096]`,
32,768 B per request per decode step, followed by a 4 B greedy-token feedback.
The backend must validate the precise mHC boundary layout before wiring real
layers. Both ranks complete reset before either launches; forward state and
backward token use SYSTEM release/acquire events, fine-grained peer-visible
payloads and shared integer counters. Per-layer caches never cross the split.

The saved dtype inventory gives an **estimated tensor payload** of 305.707 GiB,
or an ideal 152.853 GiB per rank. Assuming all routed expert matrices are FP8,
21 MoE layers per rank plus conservatively replicating every other payload
gives 163.957 GiB per rank. On hypothetical 192 GiB devices, that leaves
28.043 GiB before unbudgeted runtime allocations. This is **not** a fit claim:
actual tensor placement, conversion/scales, hybrid state, indexer caches,
workspaces and allocator overhead require a real allocation trace. A full BF16
conversion has a 299.255 GiB ideal per-rank floor and cannot fit two such GPUs.
`prepare --batch 8 --context 1024 --gpu-gib 192` recomputes the byte budget.
Context-dependent hybrid state is explicitly unknown rather than guessed as
ordinary GQA KV memory. There is no measured GLM baseline, kernel-gap ceiling,
communication latency, fusion gain or throughput prediction in this work.

## Executable two-GPU boundary experiment and oracle

`examples/glm53/boundary.py` is a real ETX graph with two device kernels:
rank 0 publishes a synthetic residual bit-pattern; rank 1 receives it and
returns a per-request checksum through the reverse event; rank 0 receives the
feedback. It changes all payloads each step, checks every element and request
on every repetition, and refuses insufficient GPUs, non-gfx942 devices, missing
peer access/atomics, residency failures, timeouts or any mismatch. This exercises
transport/reset/visibility at the proposed shape, **not GLM numerical inference**.
Its compiler plan has SYSTEM/fine-grained forward and backward events.

Local execution, without GPU:

```bash
.venv/bin/python -m examples.glm53.prepare --out build/glm53-preparation.json
.venv/bin/python -m examples.glm53.prepare --require-model --out build/glm53-blocker.json
# The second command intentionally exits 2.
ETX_DEVLIBS="$HOME/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode" \
  bash examples/glm53/build_boundary.sh build/glm53-boundary --local
bash examples/glm53/local_boundary_rocm.sh build/glm53-boundary-rocm
.venv/bin/python -m etx.tools.gpu_jobs examples/sessions/glm53_boundary.py \
  --gpus 0,1 --logdir build/glm53-dryrun --dry-run
```

Only after Qwen correctness, on a separately authorized, exclusive two-GPU
host, run `examples/sessions/glm53_boundary.py` through the normal session runner.
Its 300-second cap includes build and eight checked transport repetitions.
It deliberately uses the CPU lane with `HIP_VISIBLE_DEVICES=0,1` because the
ordinary GPU lane masks each process to one GPU; one process drives both GPUs.
Do not combine it with another session. `ETX_GLM_BATCH=2/4/8` chooses a larger
boundary at build time; the default is B1. Local checks do not establish a GPU
transport PASS. The probe's PASS also does not unblock full-model inference.

Go/no-go is correctness only: proceed to backend integration only if all bytes
and feedback IDs agree on all repetitions and peer/reset checks pass. Any other
outcome stops integration for visibility/debugging. Before implementing a full
backend, obtain an authorized, pinned checkpoint, prove per-rank memory fit and
gfx942 FP8 support, and add KDA, sparse MLA/indexer/kpool, mHC, routing/scaling,
clipped SwiGLU and FP8 pack/import interfaces with per-operator references.

`examples/glm53/oracle.py` implements the future model's offline token judge.
It requires same model/revision/config, identical per-request input IDs,
non-speculative free greedy decode and exactly 32 IDs per request, with distinct
prompts. Candidate output must match the same-checkpoint HF output exactly.
Invoke it with `--reference hf.json --candidate etx.json --out verdict.json`.
Its record schema is in the module docstring. It refuses truncated or swapped
requests and reports `model_certified: false`: token agreement cannot replace
per-operator state/logit checks and same-build import certification. Tests use
explicit synthetic records; no GLM model output was generated here.

## Local validation and remaining gates, 2026-10-17

- Clean-tree full suite at implementation commit `aaab31d`: **2253 passed,
  18 skipped**. The final focused suite covers the later supervisor, native
  wrapper and reference-preparation changes; see `focused-tests-final.log`.
- Existing local matrix: **16/16** passed, including gfx942 code generation,
  host syntax and the relevant UC checks. Full per-build logs are preserved.
- Qwen B2/B4/B8 host syntax passed; compiled plans contain **194/388/776**
  imported instances per position. **No new GPU instance check or token ran.**
- Qwen B2 exact-default O3 device compile passed with ROCm 7.2.4/LLVM 22.
  B4 exhausted the local VM (6.835 GiB RAM, 8 GiB swap); the O1 diagnostic also
  exhausted it and was not adopted. B8 ROCm compilation is NOT-COMPLETE.
- Native LLVM 23 produced B2 IR/ISA, which was independently assembled into an
  object. That older wrapper invocation exited 2 after the helper was edited
  while active; its log is retained. The current wrapper is syntax-checked and
  tested with explicit synthetic compiler processes. B4 native IR is preserved;
  B4 ISA and B8 native compilation were stopped as **NOT-COMPLETE**. Native
  checks do not replace the outstanding ROCm preflight.
- The GLM probe's two device kernels and host syntax passed both native LLVM 23
  and ROCm 7.2.4 compiler checks. Its GPU transport, all GLM model inference and
  every new performance result remain **unmeasured**.

The initial full suite's fixture errors/clean-tree refusal and all compiler
failures are retained. They consumed local CPU time only; GPU spending was $0.
The standard kernel/optimization settings are unchanged. No local Docker memory
or security configuration was changed, and no compiler/container remains active.

The next prerequisite is completing ROCm B4/B8 device builds with enough local
compiler memory, using the preserved plans/imports. Before any GPU provisioning,
also stage and bind the exact checkpoint/references. The next GPU experiment is
then the Qwen correctness session: every imported instance at positions 0/1040,
all-request coverage and exact 32/32 in both free-greedy modes/repetitions. Record
device/runtime conditions in its session record. Only after that passes should
the two-GPU transport experiment proceed. No performance tuning or winner is
part of this preparation.

## Completion continuation, 2026-10-17

The compiler memory failure is fixed in the emitter: large dispatches now share
identical control flow and load their differing integer constants from per-type
tables. Numerical tile bodies, event identities, descriptor/argument layout,
standard O3, and the default NT/VECNORM macros are retained. An inverse-emission
test reconstructs every original literal for 600 cases. This is a compiler size
fix; it does not establish GPU correctness or performance.

`bash examples/vllm_llm/batch_preflight.sh "$PWD/build/qwen-preflight"` is the
complete reproducible local gate (serial 16-build matrix, exact ROCm B2/B4/B8,
session dry run, tests). Its execution-package manifest enumerates all 20 required
model/reference assets and hashes the runner/compiler sources. Bind it to staged
assets with `batch_correctness.execution_package(Path(MODEL))` before hardware
qualification. An unbound manifest explicitly records `UNBOUND`.
