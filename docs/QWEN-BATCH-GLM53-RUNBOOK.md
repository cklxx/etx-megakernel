# Qwen B2/B4/B8 + GLM-5.3-Flash PP2 — two-GPU session runbook

Correctness only. No download, no model fetch, no performance verdict. Both
sessions run on one already-authorized `2x MI300X VM` with pre-staged assets.

Order: stage and preflight, Qwen B2/B4/B8 on GPU 0, then GLM PP2 on GPUs
`0,1`, then fetch, verify, teardown. Each `start` creates a fresh logdir
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
[docs/QWEN-BATCH-GLM53.md](QWEN-BATCH-GLM53.md#staging-and-two-gpu-execution),
prepared beforehand on an authorized box.

## 1. Preflight — one command, fails closed

```bash
tools/session.sh preflight <ip>          # tree, ROCm, docker, dry run

ssh hotaisle@<ip>
cd ~/etx
.venv/bin/python tools/two_gpu_preflight.py \
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
ETX_QWEN_BATCH_MODEL=~/llm/qwen3-30b-a3b \
  tools/session.sh start <ip> examples/sessions/qwen_batch_correctness.py 0
```

`tools/session.sh start` forwards `ETX_QWEN_BATCH_MODEL` (among the GLM
variables below) to the box. The session runs three CPU builds and three
serialized GPU qualifications on GPU 0, then a `final` verification job.

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

Both jobs are CPU-lane, so the runner does not mask either process to one GPU;
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
