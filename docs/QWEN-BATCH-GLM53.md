# Qwen B2/B4/B8 and GLM-5.3-Flash PP2

## Implemented scope and evidence

Qwen B2/B4/B8 standard local compilation is complete. GLM now has an actual
native ETX text model: embedding, every KDA/MLA and dense/MoE layer, mHC,
final norm/head, and greedy token feedback across two devices. This supersedes
the transport-only preparation in commits aaab31d and 5b9390f.

No new GPU correctness or performance result is claimed. No paid resource or
large weight download was started. Full-checkpoint token equality and real
peer/allocation behavior still require the staged checkpoint and two GPUs.

## Qwen: standard compiler memory failure fixed

Commit 85900b2 factors structurally equal dispatch cases into one body with
per-type integer tables. It retains numerical tiles, O3, NT/VECNORM, all
checks, and every task/event/argument/queue/input-map value. No local VM memory
or compiler security setting was changed.

ROCm 7.2.4/LLVM 22 device assembly, host device compilation and native host
syntax passed for B2, B4 and B8. They contain 194/388/776 imported instances
per position and 120724/241448/482896 tasks. All three report megakernel VGPR
128, scratch 32 B/lane, 7 VGPR spills; unfused VGPR 103, scratch/spills 0.
These are compiler reports, not measured GPU residency or timing.

The 16-configuration local matrix also passed. Evidence, exact commands,
source/assembly hashes and unchanged-plan checks are under
`docs/evidence/qwen-batch-glm53/completion/`. The dispatch test reconstructs
all differing integer literals and preserves separate synchronization scopes.

```bash
bash examples/vllm_llm/batch_preflight.sh "$PWD/build/qwen-preflight"
```

This runs the serial 16-build matrix, exact standard ROCm B2/B4/B8, session dry
run and test suite. `execution-package.json` enumerates all 20 required assets
and source hashes. Bind it to existing staged assets with
`batch_correctness.execution_package(Path(MODEL))`; an unbound manifest says
`UNBOUND`. A revision text file alone does not authenticate checkpoint weights.

The fixed GPU recipe is Qwen3-30B-A3B, replicated requests, context 1024,
32 free greedy tokens, two repetitions. On the same binary it requires:

1. Every imported launch byte-identical at positions 0 and 1040.
2. HF coverage for every request: existing margin >0.5 gate, exact 32-token tail.
3. Fused and same-kernel unfused decode: every request 32/32 in both repetitions.

No last-run overwrite or partial output can pass. HF margins must be finite
and nonnegative; all prefill IDs must be within vocabulary. The runner hashes
weights, manifest, HF config/revision and every reference before and after.
Required tokens per free mode are B2 128, B4 256, B8 512; these counts are
requirements, not new GPU results.

Use an already staged pinned Qwen BF16 snapshot. Generate distinct local
references without fetching:

```bash
MODEL=/absolute/packed-qwen
REF=/absolute/new-reference-directory
mkdir "$REF"
for r in 0 1 2 3 4 5 6 7; do
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python examples/llm/prep.py \
    --model Qwen/Qwen3-30B-A3B --src "$MODEL/hf" --out "$REF" \
    --context 1024 --gen 32 --rotate "$((97*r))" --golden-only
done
```

Retain the reference logs and copy the generated golden/prefill files into the
packed model. On a separately authorized existing MI300X host:

```bash
ETX_QWEN_BATCH_MODEL="$MODEL" .venv/bin/python -m etx.tools.gpu_jobs \
  examples/sessions/qwen_batch_correctness.py --gpus 0 \
  --logdir results_qwen_batch_correctness
```

## GLM identity and actual numerical path

The exact checkpoint remains `zai-org/GLM-5.3-Flash`, revision
`46a050bec13002ad29ee795c16e61cf4fa3c4a9c`, config SHA256
`bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`.
Saved upstream sources are in the evidence directory. Scope is text-only,
45 layers, H=4096, four residual streams, 34 KDA layers, 11 sparse MLA layers,
288 routed experts/top-8, one shared expert; vision and MTP are excluded.

`model.py` constructs the full ETX graph. `tiles.hip` and `native.h` implement:

- BF16/F32/E4M3FN matrix readers with FP32 128x128 block scales; FP8 payload
  remains compressed in memory and is decoded to BF16 values during GEMV.
- RMSNorm, clipped SwiGLU, sigmoid/noaux_tc correction-bias top-k,
  normalized routing/scaling, shared and selected routed experts.
- Four-stream mHC collapse, 20-step Sinkhorn mixing and unweighted final head.
- Causal depthwise convolution, safe forget gates, L2-normalized KDA recurrence,
  FP32 recurrent/conv storage and gated output normalization.
- NoPE MLA projections, per-layer cache, normalized indexer keys, learned
  channel-wise kpool compression, complete-pool selection and visible tail,
  masked attention and output projection.

These are correctness-first complete-layer tiles, with one workgroup per
request/layer. No speed claim is made. This path uses BF16 activations and
FP32 accumulators; it does **not** emulate dynamic FP8 activation quantization.
The independent reference explicitly uses HF FP8-to-BF16 dequantization of the
same checkpoint. Raw full logits are retained; exact greedy tokens remain the
acceptance gate, and token agreement is not a blanket numerical certificate.

Ranks own whole layers: 0–23 on rank 0, 24–44 on rank 1. Each has 21 MoE
layers. Rank 0 owns embedding; rank 1 owns norm/head. All layer state stays
with its owner. The actual boundary is `[B,4,4096]` BF16 (32768 B/request),
and greedy feedback is B int32 IDs. ETX selects SYSTEM events and fine-grained
peer memory for this boundary; both ranks reset before either launches.

The host launcher loads actual staged tensors in bounded chunks, checks the
per-rank free-memory requirement, allocates state/workspaces, validates peer
access/atomics and residency, prefills every prompt token, then returns exactly
32 free greedy IDs. It prints every step, records actual allocations and has
per-step deadlines. A fresh process reloads/reset caches for the second run.

## GLM local acceptance and independent reference

```bash
bash examples/glm53/local_acceptance.sh "$PWD/build/glm-local-acceptance"
```

The same C++ body as the HIP tiles runs locally against independent
Transformers 5.19.0 GLM5Next modules, whose source hashes are recorded.
The four-layer fixture includes both attention types and both MLP types.
B1/B2 FP32, B4 BF16 and B8 FP8→BF16 each run 32 steps with exact greedy IDs,
per-layer comparisons and KDA state checks. Fixtures are explicitly rejected
by production staging/qualification. Additional tests cover unique-score
kpool selection beyond budget, SwiGLU saturation and malformed descriptors.

An early reduced-budget fixture found arbitrary top-k choices at exactly tied
pool scores. Its failure logs are retained. The requested 1024+32 production
context is below index_topk=2048, so all complete pools fit; full decode tests
use that regime. The over-budget operator test uses unique scores. Exact
checkpoint hardware output remains unmeasured; no tie mismatch is tolerated
by the qualification runner.

Both rank kernels and host device pass compile at ROCm 7.2.4 O3, also for the
full pinned 45-layer/H4096/B8 shape (384 tasks). That full-shape compiler
check uses no weights and is explicitly non-runnable. Native LLVM 23 IR/ISA
and host syntax checks are additional gates, not GPU correctness.

## Staging and two-GPU execution

The operator sequence — one fail-closed preflight, ordered Qwen then GLM
sessions, strict pass criteria, budget, evidence and teardown — is in
[QWEN-BATCH-GLM53-RUNBOOK.md](QWEN-BATCH-GLM53-RUNBOOK.md).

The release checkpoint is not fetched (owner direction, 2026-10-17). The GPU gate
for this path is the [module gate](QWEN-BATCH-GLM53-RUNBOOK.md#module-gate): a
synthetic 4-layer native-FP8 checkpoint (`examples/glm53/module_gate.py`) run
through this same stager, reference, build, launcher and qualification with
`gate=module`.

All commands below require already available local weights; none fetches or
provisions. Preserve snapshot provenance and place the exact commit in
`SNAPSHOT/REVISION`. The stager requires the byte-identical pinned config and
uses the checkpoint safetensors index. It maps native names, concatenates the
three convolution banks, retains separate expert matrices and FP8 scales,
and validates every tensor shape/dtype/offset/scale and complete layer ownership.
Missing or unsupported tensors fail closed.

```bash
.venv/bin/python -u -m examples.glm53.stage --src "$SNAPSHOT" --out "$STAGE" --maxlen 1072
.venv/bin/python -m examples.glm53.prepare --require-model --stage "$STAGE"
```

The independent oracle uses the original snapshot, never the ETX packed
weights. `prompts.json` is an array of distinct, equal-length token-ID arrays
for B1/2/4/8. It runs non-speculative free greedy decode for exactly 32 steps,
ignoring early EOS. BF16 dequantization of the complete checkpoint exceeds
two 192 GiB devices; supply adequate CPU/disk offload explicitly:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u -m examples.glm53.reference \
  --src "$SNAPSHOT" --prompts prompts.json --out "$REFERENCE" \
  --offload /absolute/reference-offload \
  --max-memory '{"0":"80GiB","1":"80GiB","cpu":"32GiB"}'
```

Disk/CPU capacities above are explicit examples, not a fit certificate. Do not
start this reference job without sufficient existing capacity and authorized
GPU use. The candidate's native compressed payload is estimated at 305.707 GiB
total. At B8/context1024, exact implemented cache shapes add about 6.85/5.74 GiB
per rank, plus scratch/activations, events and allocator overhead. Actual
weight placement and allocations must pass the host checks.

After Qwen hardware correctness, run on a separately authorized exclusive
existing pair of gfx942 GPUs. One process needs both GPUs visible:

```bash
ETX_GLM_STAGE="$STAGE" ETX_GLM_REFERENCE="$REFERENCE" ETX_GLM_BATCH=1 \
  .venv/bin/python -m etx.tools.gpu_jobs examples/sessions/glm53_model.py \
  --gpus 0,1 --logdir results_glm53_model
```

This builds the actual model and runs two strict qualifications. It verifies
native source/generated/binary identity and staged/reference hashes before
and after, all step IDs, exactly 32 IDs per request, finite complete logits,
and exact HF tokens in both processes. `qualification.json` retains full
logit errors and explicitly distinguishes token equality from numerical
certification. The boundary probe remains an optional transport diagnostic;
it is not the model execution path.
