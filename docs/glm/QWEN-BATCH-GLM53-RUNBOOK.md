# Qwen B2/B4/B8 + GLM-5.3-Flash PP2 — two-GPU session runbook

Correctness only. No model download, no performance verdict. Both sessions
run on one already-authorized `2x MI300X VM` with pre-staged assets. A staged
`~/etx` has no `.venv`, so each session starts with the `apt` and `etx_env`
setup jobs of `examples/sessions/ds_interleave.py`: they create `~/etx/.venv`
(system site packages) and install only `python3.12-venv`, PyYAML (and NumPy
for GLM) when the box lacks them.

**2026-10-17, owner direction: the 305 GiB GLM release checkpoint is not fetched.**
Run the [module gate](#module-gate) on GPUs `0,1` in place of section 3; section 3
stays for a box where the release is already staged.

Order: stage and preflight, the GLM module gate on GPUs `0,1` (minutes), Qwen
B2/B4/B8 on GPU 0, then fetch, verify, teardown. Each `start` creates a fresh logdir
`~/sess/<session id>` (printed by the runner); every job log and receipt is
written there under `ETX_SESS_LOGDIR`.

## 0. Prerequisites

- Hot Aisle `2x MI300X VM`, provisioned with `tools/hotaisle_grab.sh 2x`.
  Billing starts at provisioning; the 2x type has a 1 h minimum.
- This branch pushed with `tools/session.sh push <ip>` from a clean tree.
- Pre-staged on the box (prepared offline, outside this runbook):

| asset | contents |
|---|---|
| `~/llm/qwen3-30b-a3b/` | `hf/config.json`, `hf/REVISION`, `vl.bin`, `vl.manifest`, 8 goldens and 8 prefill references (B8 set; B2 uses the first 2 of each, B4 the first 4) |
| `/staged/glm53/` | `stage.json`, `glm53_spec.h`, 45 layer blobs + metas, `embedding`/`norm`/`head` |
| `/staged/glm53-reference.json` + `.logits` | independent HF greedy record for the same checkpoint, 32 steps |

Nothing in this runbook downloads weights or runs the HF reference. The
reference is the offloaded BF16 job from
[docs/glm/QWEN-BATCH-GLM53.md](QWEN-BATCH-GLM53.md#staging-and-two-gpu-execution),
prepared beforehand on an authorized box.

## 1. Preflight — one command, fails closed

```bash
tools/session.sh preflight <ip>          # tree, ROCm, docker, dry run of both sessions

ssh hotaisle@<ip>
cd ~/etx
python3 tools/two_gpu_preflight.py \
  --qwen-model ~/llm/qwen3-30b-a3b \
  --glm-stage /staged/glm53 \
  --glm-reference /staged/glm53-reference.json \
  --glm-batch 1
```

It checks the Qwen config shape, the 40-hex revision pin, 1024-token prompts,
32-token greedy outputs, prefill references, distinct prompts and `vl.bin` for
B2/B4/B8; the GLM pinned model/revision/config, dimensions, PP2 ownership,
every tensor descriptor and every blob/meta size; and the reference identity,
prompts, 32-step logits file and its checkpoint index/shard binding.

- Success prints one `PREFLIGHT OK <item>` per check and a final `PREFLIGHT OK`.
- Any failure prints `PREFLIGHT FAIL <item>: <reason>`, names the missing or
  wrong file/field, and exits 2. **Do not start the sessions.**
- Default checks are structural and fast. `--hash` additionally re-hashes the
  whole Qwen blob and GLM stage (minutes; use it when assets moved).

The reference batch must equal `--glm-batch`; the OK line prints the batch to
pass to the GLM session.

## 2. Qwen B2/B4/B8 (GPU 0)

```bash
ETX_QWEN_BATCH_MODEL=/home/hotaisle/llm/qwen3-30b-a3b \
  tools/session.sh start <ip> examples/sessions/qwen_batch_correctness.py 0
```

`tools/session.sh start` forwards `ETX_QWEN_BATCH_MODEL` (among the GLM
variables below) to the box. Give the box's absolute path: an unquoted `~` in
this assignment expands on the laptop (`/Users/...`), not on the box. Unset, the
session uses the box's own `~/llm/qwen3-30b-a3b`. The session runs the two
setup jobs, three CPU builds and three serialized GPU qualifications on GPU 0,
then a `final` verification job.

Poll until the runner is gone and the record is sealed:

```bash
tools/session.sh status <ip>       # repeat until it prints "runner not running"
```

Fetch the sealed record and verify it:

```bash
tools/session.sh fetch <ip> ~/etx-captures
.venv/bin/python tools/two_gpu_verify.py ~/etx-captures/<record-id> \
  --qwen-only --require-seal
```

## 3. GLM-5.3-Flash PP2 (GPUs 0,1)

Both model jobs are CPU-lane, so the runner does not mask either process to one GPU;
the qualification job sets `HIP_VISIBLE_DEVICES=0,1` itself. `ETX_GLM_BATCH`
must match the preflighted reference.

```bash
ETX_GLM_STAGE=/staged/glm53 \
ETX_GLM_REFERENCE=/staged/glm53-reference.json \
ETX_GLM_BATCH=1 \
  tools/session.sh start <ip> examples/sessions/glm53_model.py 0,1
```

Then, as above:

```bash
tools/session.sh status <ip>
tools/session.sh fetch <ip> ~/etx-captures
.venv/bin/python tools/two_gpu_verify.py ~/etx-captures/<record-id> \
  --glm-only --require-seal
```

## 4. Strict pass criteria

Both session files end with `tools/two_gpu_verify.py` in their `final` lane.
A record that lacks any item below fails its session; `session.sh fetch` also
refuses to call an incomplete or tampered record VERIFIED.

Qwen, per batch in `qualify_b{2,4,8}/correctness.json`:

- `status=PASS`, `performance=unmeasured`, the fixed recipe from
  `batch_correctness.recipe(batch)` with `check_enabled=true`;
- `positions=[0,1040]`; imported instances **194/388/776** for **B2/B4/B8**
  (`docs/evidence/qwen-batch-glm53/qwen-plan-inventory.json`) and exactly
  **97 per request** at both positions;
- the qualification requires every imported instance in `check_p{0,1040}.log`
  to be `identical` with `first_byte=-1`, a valid request id and the right
  position, else it cannot reach `status=PASS`;
- **32/32** tokens per request in **both repetitions** of coverage, fused and
  unfused runs (`greedy_tokens_per_mode = batch × 32 × 2`).

GLM, in `qualification/qualification.json`:

- `status=TOKENS-EQUAL`, 2 independent repetitions, `matched_each=[32]×batch`,
  `generated_each=32`, pinned model/revision;
- each verdict has `status=TOKENS-EQUAL`, `checkpoint_execution="real two-GPU
  ETX model"`, finite full-logit error, 32 per-step errors and matching argmax;
- build hashes, reference + logits hashes, `model_certified=false`.

`logit_gate=REPORTED` is the session default: the full-logit error is recorded
but no tolerance is enforced. An `ENFORCED` gate needs `--logit-atol` added to
the session command, with a tolerance justified beforehand; do not invent one
after seeing the result.

## 5. Budget

Wall clock — **no GPU wall time has been measured**. The only fixed numbers are
the configured caps; plan the allocation from them:

| item | value | kind |
|---|---|---|
| Qwen aggregate hard stop (`MAX_SESSION_SECONDS`) | 7200 s | configured |
| Qwen builds (each) / qualification (each) | 2100 s / 4800 s caps | configured |
| GLM aggregate hard stop | 18000 s | configured |
| GLM build / qualification caps | 2100 s / 15000 s | configured |
| local B4 exact-default compile | OOM (exit 137) | **measured**, laptop |
| local B8 native IR compile | stopped after >20 min | **measured**, laptop |
| Qwen expected wall | builds dominate; allow the full 2 h cap | estimate |
| GLM expected wall | build then two 32-step runs; no GPU time measured | estimate |

The local compile failures show the build is heavy, not that the VM will fail;
build times on the VM are unmeasured. If `HARD-STOP` appears, the record is
incomplete — rerun with a larger cap (`ETX_SESS_MAX_SECONDS` is forwarded)
rather than treating the shorter run as a result.

VRAM — `examples.glm53.prepare.budget` labels every figure an ESTIMATE from the
pinned dtype inventory and implemented cache shapes (context 1024, 192 GiB per
device); `examples/glm53/model_host.hip` checks `hipMemGetInfo` per rank at
runtime:

| item | value | kind |
|---|---|---|
| GLM checkpoint payload | 305.71 GiB | estimate |
| GLM native per rank | 163.96 GiB (ideal half 152.85) | estimate |
| GLM payload reserve per device | 28.04 GiB | estimate, not a fit certificate |
| GLM BF16 per-rank floor | 299.26 GiB — does **not** fit; reference uses offload | estimate |
| GLM cache per rank B1/B2/B4/B8 | 0.856/0.718, 1.713/1.436, 3.426/2.872, 6.851/5.743 GiB | estimate |
| GLM scratch per rank B1/B2/B4/B8 | 0.017/0.015 … 0.134/0.117 GiB | estimate |
| Qwen BF16 weights | 56.87 GiB from 30.53 B parameters | estimate, computed from the pinned config |
| Qwen KV at B8, context 1072 | 0.785 GiB (48 layers × 2 × 4 KV heads × 128 × 1072 × 2 B × 8) | estimate, computed from the pinned config |

No Qwen allocation is measured or enforced by the session; a runtime OOM fails
the qualification.

## 6. Evidence to pull back

`session.sh fetch` copies the whole sealed record (minus `b_*/run`, which stays
on the box) into `~/etx-captures/<record-id>` and verifies its MANIFEST/SEAL.
The files that carry the verdicts:

Qwen record:

- `status.json`, `runner.log`, `verify.log`, `qwen-verification.json`
- `qualify_b{2,4,8}/correctness.json`
- `qualify_b{2,4,8}/{check_p0,check_p1040,coverage,fused,unfused}.log`
- job logs `build_b{2,4,8}.log`, the inner build log `log_b{2,4,8}/build.log`,
  and the generated plan/binaries under `build_b{2,4,8}/`

GLM record:

- `status.json`, `runner.log`, `model_build.log`, `model_qualify.log`,
  `verify.log`, `glm-verification.json`
- `build-log/build.log`, `model/build-identity.json`, `model/plan.json`,
  `model/load.tsv`, `model/glm53_spec.h`
- `qualification/qualification.json`, `qualification/prompts.txt`
- `qualification/run{0,1}.log`, `.json`, `.tokens`, `.logits`

Keep each record directory intact: `python3 -m etx.session.record verify <dir>`
is what makes it a result.

## 7. Teardown

Only after every fetch reports VERIFIED (or an intentionally abandoned capture
is recorded as incomplete):

```bash
tools/session.sh status <ip>      # runner not running
ssh admin.hotaisle.app            # Hot Aisle TUI
# select this VM -> delete -> answer y
```

Confirm the TUI no longer lists the VM (no VM of that type / "No virtual
machines"), then record the closing balance. Delete the VM when the planned work
is done; do not leave a billed 2x VM running.

## Module gate

The two-GPU GLM model path at module scale, on a synthetic checkpoint, so no
release download is needed. `examples/glm53/module_gate.py` writes a checkpoint in the
release's own layout: native tensor names, FP8 E4M3 matrices with 128x128 FP32
`weight_scale_inv` blocks, BF16/F32 elsewhere. Its dtype rule reproduces the
release's pinned F8_E4M3 and F32 parameter counts exactly
(`tests/test_glm53_module_gate.py`). The values are deterministic random numbers
(numpy PCG64, seed 20261017). The module config is the release `config.json` with
only sizes changed:

| | module gate | release |
|---|---|---|
| layers / per rank | 4 / 2+2 (KDA+dense, MLA+MoE / MLA+MoE, KDA+dense) | 45 / 24+21 |
| hidden, intermediate, MoE intermediate | 512, 1024, 256 | 4096, 12288, 2048 |
| routed experts / top-k / shared | 8 / 2 / 1 | 288 / 8 / 1 |
| MLA heads x head dim, q/kv LoRA rank | 4 x 256, 384 / 256 | 64 x 256, 1536 / 512 |
| KDA heads x head dim | 4 x 128 | 64 x 128 |
| indexer heads x dim, kpool, top-k | 4 x 128, 4, 64 | 32 x 128, 4, 2048 |
| vocab, batch, prompt | 4096, 2, 128 tokens | 154880, 1, 1024 tokens |

The pinned identity is `docs/evidence/glm53-module-gate/` (config, model id
`etx-synthetic/GLM-5.3-Flash-module-gate`, revision, `weights_sha256`). Everything
after the generator is production code run with `gate=module`: `stage.stage` and
`stage.validate`; the HF reference `reference.run`, an independently instantiated
Transformers GLM5Next that dequantizes the same files itself and must load every
text tensor with no missing, unexpected or mismatched key; the build
`build_model.sh`; the launcher `model_host.hip`; `run.qualify`; and
`two_gpu_verify.py`. The production gate refuses a module stage, reference or
receipt, and the module gate refuses the release's.

**Why the head is structured, and the added rule.** These were measured locally
before any GPU run, with the CPU build of the same native body as the HIP tiles
against this reference.

- With plain random weights, MoE routing sits on near-ties: in about 1% of
  decisions the 2nd and 3rd expert scores are within 1e-3. BF16-level
  differences flip those choices, giving logit spikes up to about 4 at
  otherwise ~0.05 agreement.
- Greedy decoding on a random head also has near-ties (top-2 gap as small as 0.0).
  A correct body then diverged from HF at decode step 3, which would be a false FAIL.
- With the head carrying a permuted copy of the embedding (`HEAD_ALPHA = 2.0`;
  every layer weight stays random), the HF minimum top-2 margin is 3.56
  (median 10.4). Tokens are then 32/32, but they also stay 32/32 with real bugs.
- The pre-registered rule `MEDIAN_LOGIT_BOUND = 1.5` separates the correct body
  from all three such bugs:

| CPU native body vs HF | tokens (2 requests) | median / max per-step logit error |
|---|---|---|
| correct | 32/32, 32/32 | 0.453 / 4.98 |
| shared expert dropped | 32/32, 32/32 | 5.02 / 7.61 |
| KDA beta without sigmoid | 32/32, 32/32 | 9.13 / 11.7 |
| routed scaling dropped | 32/32, 32/32 | 5.59 / 8.69 |
| kpool tail off by one | 0/32, 0/32 | — |

The module gate passes only if the production qualification is TOKENS-EQUAL
(32/32 per request, two independent repetitions) and, for every repetition, the
median of the 32 per-step max-abs logit errors is at most 1.5. That bound is about
3x above the correct CPU median and 3x below the smallest bug median. It was
registered before the GPU run and must not be changed after seeing it.
`tests/test_glm53_module_gate.py` shows that each of the four bugs fails the gate.

### 1. Offline, on the laptop (once; no GPU)

`prepare` needs torch, transformers and `accelerate`. The FP8 dequantize path
requires `accelerate`, so it runs in a separate venv that inherits the etx venv:

```bash
python3.13 -m venv ~/code/.venvs/glm53-module-gate
echo ~/code/etx/.venv/lib/python3.13/site-packages > ~/code/.venvs/glm53-module-gate/lib/python3.13/site-packages/etx-venv.pth
~/code/.venvs/glm53-module-gate/bin/pip install accelerate        # accelerate + psutil only

M=~/code/etx-cache/glm53-module
~/code/.venvs/glm53-module-gate/bin/python -m examples.glm53.module_gate prepare --out $M
#   -> MODULE-GATE CPU PREDICTION OK: 32/32 x 2, median logit error 0.453 <= 1.5   (else exit 1: do not book a GPU)
ETX_GLM_GATE=module ETX_GLM_BATCH=2 bash examples/glm53/build_model.sh $M/stage $M-build --local
bash examples/glm53/local_model_rocm.sh $M-build     # ROCm 7.2.4 device + host compile (GLM-ROCM-* PASS)
~/code/.venvs/glm53-module-gate/bin/python -m pytest -q tests/test_glm53_module_gate.py   # 15 passed
```

`prepare` writes `checkpoint/` (29 MB), `stage/` (29 MB), `prompts.json`,
`reference.{json,logits}` (1 MB) and `cpu-prediction.json`. The box needs only
`stage/` and `reference.*`.

### 2. On the box

```bash
tools/session.sh push <ip>
tools/session.sh preflight <ip>
rsync -az $M/stage $M/reference.json $M/reference.logits hotaisle@<ip>:glm53-module/
ssh hotaisle@<ip> 'cd ~/etx && python3 tools/two_gpu_preflight.py --glm-gate module --glm-batch 2 --hash \
    --glm-stage ~/glm53-module/stage --glm-reference ~/glm53-module/reference.json'
#   -> PREFLIGHT OK glm-stage: 4 layers, ranks 0-1/2-3 ...; PREFLIGHT OK glm-reference: batch 2, prompt 128 ...

ETX_GLM_GATE=module ETX_GLM_BATCH=2 \
ETX_GLM_STAGE=/home/hotaisle/glm53-module/stage \
ETX_GLM_REFERENCE=/home/hotaisle/glm53-module/reference.json \
  tools/session.sh start <ip> examples/sessions/glm53_model.py 0,1

tools/session.sh status <ip>           # until "runner not running"
tools/session.sh fetch <ip> ~/etx-captures
.venv/bin/python tools/two_gpu_verify.py ~/etx-captures/<record-id> --glm-only --gate module --require-seal
```

The session's `final` job already runs `two_gpu_verify.py --glm-only --gate
module`, so a record whose qualification exited 0 but misses the rule fails the
session. Then run the Qwen session (section 2) on the same VM.

### 3. What a PASS proves, and what it cannot

A PASS proves, on two real MI300X GPUs:
- The production build (codegen, `hipcc`), launcher and qualification run end to
  end on staged native-FP8 weights. That covers two-GPU allocation, peer access,
  fine-grained buffers, SYSTEM-scope events, PP2 ownership (layers 0-1 on GPU 0,
  2-3 on GPU 1), the cross-GPU residual hand-off and token feedback, over 159
  steps per repetition (128 prompt positions, then 31 fed-back decode steps).
- The native body, run on the GPU, reproduces an independently instantiated HF
  GLM5Next on the same files: 32/32 greedy tokens for each of two requests in two
  repetitions, with the full logits within the registered median bound. This
  covers FP8 block dequantization, mHC/Sinkhorn, KDA convolution and recurrent
  state, sparse MLA with kpool selection beyond its budget (128 > 64 tokens),
  sigmoid routing with correction bias, routed and shared experts, the dense MLP,
  and the head.
- ETX's native tensor names, shapes and storage dtypes load exactly in HF. This is
  already proven offline by `prepare`.

It cannot prove:
- The release checkpoint itself: its published tensor names (`model.layers.*`
  vs `model.language_model.*`), how HF applies the release's
  `modules_to_not_convert`, or any value. Only the dtype rule is checked against
  the release inventory.
- Release scale: 45 layers, 288 experts / top-8, hidden 4096, vocab 154880,
  1024-token prompts, index top-k 2048. Accumulation lengths are 8x longer at
  release scale. The memory fit (164 GiB per rank, estimated), load time and
  per-step deadlines at that size are also outside the gate.
- Trained-weight behaviour. The gate's tokens are robust by construction
  (structured head), so its layer evidence is the median logit rule, not the
  tokens. Real routing margins are unknown.
- Batch 1/4/8 (only batch 2 runs), MTP, vision, or any performance number.

Measured locally: HF's own chunked prefill and its token-by-token decode already
differ by up to 0.076 in logits at 128 tokens, so ETX's token-by-token prefill is
compared against a numerically different HF path. The gate includes that, and so
will a release run.

### 4. Budget

| item | value | kind |
|---|---|---|
| `prepare` (checkpoint, stage, HF reference, CPU prediction) | 14.5 s, laptop CPU | measured |
| ROCm 7.2.4 device compile, both ranks | 16 s under x86 emulation; 176/174 VGPRs, 0 B scratch, 2 waves/SIMD | measured |
| `tests/test_glm53_module_gate.py` | 15 passed in 64 s (module-gate venv) | measured |
| session caps (`ETX_GLM_GATE=module`) | 2700 s session, 1200 s build, 1800 s qualify | configured |
| setup jobs + build on the VM | 1-4 min | estimate |
| GPU execution, 2 repetitions x 159 steps, both GPUs | 10-45 s (15-110 ms/step: lane-serial mHC, kpool and KDA loops dominate) | estimate, no GPU time measured |
| whole session wall | about 3-6 min, about 2 GPU-minutes of device time | estimate |

The 2x VM's 1 h minimum dominates the cost. Run the gate first in the Qwen
session's allocation; it adds only its wall time to that allocation.
