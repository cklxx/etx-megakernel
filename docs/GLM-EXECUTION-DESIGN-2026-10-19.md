# GLM execution design and its gates, 2026-10-19

Offline analysis; no GPU was used. This file consolidates the former
`GLM-LAYER-BANDWIDTH-CACHE-AUTOCONF-2026-10-19.md` into sections 4–5 and adds
the four gates (6–9) that make any further optimisation judgeable. Labels:
[M] measured, [E] arithmetic from stated inputs, [F] gfx942 fork fact,
[U] absent from the record.

**Blocked until §6 exists: §5 (tile auto-optimisation).** Work order: §6 oracle
first; §7's probe may run any time because it needs no weights; §9 measures only
what §6 has certified; §5 runs last, against §8's fixed bar.

## 1. Kernel selection

The GLM numerical path is entirely native. `examples/glm53/tiles.hip` and
`native.h` implement FP8 (E4M3FN) matrix readers with FP32 128×128 block scales
decoded to BF16 during GEMV, RMSNorm, clipped SwiGLU, top-k routing, the
four-stream mHC collapse and Sinkhorn mixing, KDA (convolution, forget gates,
L2-normalised recurrence), and NoPE MLA with the indexer and kpool
(`docs/QWEN-BATCH-GLM53.md:87-99`). These are correctness-first complete-layer
tiles, one workgroup per request/layer, with no speed claim
(`docs/QWEN-BATCH-GLM53.md:100-104`). Unlike the Qwen path, the GLM path
imports no vLLM kernels; a build requires exactly `megakernel_d0.hip` and
`megakernel_d1.hip` (`examples/glm53/run.py:46`).

Selection is therefore by bytes, not by name. One expert is
3 × 2048 × 4096 × 1 B = **25.17 MB** FP8 (scales add 6.1 KB, 0.024%, ignored).
A batch-1 layer is 8 routed experts (201.3 MB) + shared expert (25.2 MB) +
router (1.18 MB) = **227.7 MB**; 45 layers is **10.25 GB per step** → a
**2.88 ms floor** at the measured 3.56 TB/s [E]. At B8 the same layer is
1,611 MB replicated, or 1,462 MB as the routed union (58.1 experts), only 9.2%
saved because 288 experts spread the routing wider than Qwen's 128 [E].

The bandwidth basis: `hw_bench lf` measured **3.56 TB/s at 192 × 64 KB in
flight, 3.93 TB/s maximum** (`examples/vllm_llm/batch_pp.py:10`) [M] — 67.2%
and 74.2% of the 5.3 TB/s peak. No in-flight sweep explains the remaining
1.37 TB/s, so that gap is [U]. Every floor in this file uses 3.56.

Rule: a tile stays or is replaced by its time against the floor for the bytes
it moves. A tile is never selected by name, and no per-layer gfx950 budget is
transferable (the gfx942 GEMV pass is 6 tokens wide, §4).

## 2. Fusion

Fusion is already at its ceiling within a rank: the plan compiles to exactly
two kernels per decode step, one per device (`examples/glm53/model_build.py:21`,
`examples/glm53/run.py:46`), and the host launches both
(`examples/glm53/model_host.hip:64-75`). All layer boundaries, residual
carries, the mHC state and the per-layer caches stay inside one kernel.

Consequences:

- The Qwen fusion result (4.015% on the ETX step-mean basis, ESTABLISHED per
  `CLAUDE.md` §1) has **no GLM analogue**: it recovered gaps between ~40
  per-layer launches, and GLM has no such chain [E].
- E1–E3 are Qwen experiments against imported kernels and replicated dense
  columns; neither exists in the GLM path [E].
- Fusion cannot reduce bytes; §1's floor survives any fusion.
- The one fusion boundary left is the PP2 seam between two kernels, costed in
  §7 [E].

## 3. Pipelining

Static partition: whole layers, split at the median sparse layer
(`examples/glm53/prepare.py:55-67`); rank 0 owns layers 0–23, rank 1 owns
24–44, with 21 MoE layers each, no all-to-all and no tensor sharding
(`docs/QWEN-BATCH-GLM53.md:107-108`). Embedding is on rank 0; norm/head on
rank 1.

Per step the host resets both ranks, launches both kernels, and rank 1's layers
wait on the boundary carried over SYSTEM events and fine-grained peer memory
(`docs/QWEN-BATCH-GLM53.md:109-111`; `examples/glm53/model_host.hip:73-75`).
The boundary is crossed **once per decode step** (`prepare.py:88,101` sets
`boundary_bytes_per_decode_step` to exactly one `[B,4,4096]` transfer).

- Rank 1 can do no layer work before the boundary; its 21 layers run after
  rank 0's 24 [E]. No per-layer timing exists, so the bubble's size is [U].
- No cross-step overlap: both ranks reset before either launches
  (`model_host.hip:73-74`), so step t+1 cannot start on rank 0 while rank 1
  runs step t.
- A per-layer hand-off schedule or cross-step pipelining would change this
  shape; neither is designed here, and both need §7 measured and §9 in place
  before they can be judged.

## 4. Cache use

**What a layer re-reads.** Every weight is re-read on every decode step. One
B8 layer's routed stream is 1,611 MB against **4 MB of L2 per XCD (32 MB for 8
XCDs)** (`docs/RESEARCH-sync-hardware.md:13,62`) and a **256 MB MALL**
(`docs/RESEARCH-sync-hardware.md:98`) — 50× L2 and 6.3× MALL. Nothing holds the
routed stream; it is HBM-bound by construction. The shared expert and router
(26.4 MB per layer) are the only reused part: they fit MALL at 10.3% of it.

**Which loads carry which bits.** In this repo's ISA `sc0` = GLC (L1-level
coherence; `etx/tools/isa_hops.py:112`,
`etx/runtime/include/etx/primitives_amdgcn.h:27-35`) and `sc1` = SLC
(system/L2 level; `etx/tools/isa_hops.py:116`, `etx/ops/coh.py:2,27`).
Recommended policy [E, gated by §6 and measured by §9]:

| Data | Bits | Why |
|---|---|---|
| Routed-expert stream, read once per step | `slc` (SLC, bypass L2) | cannot be reused in-step; allocating it evicts the shared expert and router |
| Shared expert, router, dense projections | plain cached, no `glc`/`slc` | reused within the step and across steps; the only MALL-resident candidates |
| Cross-XCD activations and payloads | keep `sc1` agent-scope loads and `sc0 sc1` system-scope atomics (`etx/ops/coh.py:27-35`) | already measured; changing them reopens a correctness surface |
| Weight pre-warming | none | L2 warming measured **−3%** (`docs/RESEARCH-sync-hardware.md:89`) |

**LDS residency at 64 KB per CU.** Staging bf16 activations of K = 4096 costs
4096 × 2 B = 8 KB per token: **6 tokens = 48 KB (75%), leaving 16 KB for
double-buffering; 8 tokens = 64 KB (100%), no headroom.** That is the gfx942
port's 6-token pass width (0.75 × 64 KB ÷ 8 KB = 6) [F]. Weights are never
LDS-resident — one expert is 25.17 MB against 64 KB — so every weight byte
streams through the CU and LDS holds only the activation slab. B8 means two
passes (6 + 2), or FP8 activation staging (8 × 4096 × 1 B = 32 KB, 50%),
numerically unproven [U].

## 5. Tile auto-optimisation

**What `examples/vllm_llm/autoconf.py` already searches.** Five build switches:
`ETX_VL_CHAIN` (2), `ETX_VL_QKNR` (3), `ETX_VL_DSL_GEMV` (2), `ETX_VL_DSL_TOPK`
(2), `ETX_VL_DSL_ATTN` (2) — 48 combinations, one invalid pair rule
(`autoconf.py:25-31,34-36`). It builds each plan locally and ranks it with the
XCD simulator and the measured phase model (`autoconf.py:57-69`). No GPU. It is
Qwen-specific: none of the five switches exists in the GLM path.

**What is missing.** The search space none of it touches:

- **BM/BN** — tile constants in `etx/ops/gemv.py` and the GLM tiles, not build
  switches;
- **waves per SIMD** — occupancy fixed by the workgroup shape;
- **VGPR budget** — the compiler report (VGPRs, spills, scratch) never returns
  to the search;
- **LDS footprint** — the 6-token/64 KB constraint in §4 is not modelled, so
  the simulator will rank tiles that do not fit gfx942's LDS;
- no per-config measurement: the ranking is simulated, and the measured
  reference (`autoconf.py:75`) is a single point.

**Closing the loop within the box budget.** One session script, prepared
offline:

1. enumerate the space locally (BM × BN × waves × LDS budget), free;
2. compile every candidate for gfx942 with the local matrix
   (`examples/vllm_llm/local_matrix.sh`), dropping any over the VGPR / spill /
   LDS limits from the compiler report, free;
3. measure survivors on one MI300X in one script: pinned, under `flock`, five
   interleaved rounds, B8 fixed at 8 rows, one microbench per candidate;
4. feed the measured µs back into `pmodel`'s constants and re-rank.

Budget [E]: 36 survivors × 2 min = 72 min plus 20 min setup ≈ 1.5 h ≈ **$3 at
$1.99/h**, paid once in the same session as another measurement.

**Blocked:** this section cannot start until §6 exists. Every GLM per-layer
number above is arithmetic on stated dimensions; tuning against it would tune
to an unrun model.

## 6. Real-weight correctness oracle

**Status.** Fixture correctness is 480/480 exact greedy tokens on a synthetic
4-layer, small-dimension checkpoint [M]. Real weights have **never run** [U]
(`docs/TWO-WEEK-RESULTS-2026-10-18.md:68-72`); the only plan on record is a
4-layer slice, ~29 GB, ~30 min of GPU, not run (`...:73`). This section fixes
that build and makes it the gate every later change must pass.

**Slice.** Four *sparse* layers, indices 22–25: ~29.0 GB of FP8 expert payload
(4 × 7.25 GB) plus embedding [E], matching the recorded ~29 GB. Their types are
K, M, K, K (KDA, sparse MLA, KDA, KDA), so both attention kinds appear; the
layer-23/24 split reproduces the real PP2 boundary (22–23 on rank 0, 24–25 on
rank 1), so this one slice exercises **both ranks and the seam**. A prefix of
layers 0–3 would be ~9 GB and would stay on rank 0; it is a second choice only
if the dense-MLP path must be covered.

**The slice is a code change first — implemented 2026-10-19.** `prepare.py`
gains a `slice` gate next to `production`/`module` over the same pinned release
evidence, plus `parse_layers`, `check_slice` and `slice_config`; the derived
four-layer config (same hidden/expert dims, four layer types) and the source
window are recorded in `stage.json` and in the reference record, and
`oracle.compare` refuses reference/candidate pairs whose windows differ.
`stage.py:238-240` and `reference.py:168-171` take `--layers 22-25 --gate
slice`, `run.py` builds and qualifies with the same gate, and both the stager
and the reference **refuse any window that does not straddle the PP2 split at
layers 23/24** (`prepare.py:check_slice`). The reference hashes the derived
snapshot's `config.json`, index and every shard before HF loads it, rechecks
them after the run, and records them in its provenance
(`reference.py:snapshot_hashes`). Tests: `tests/test_glm53_slice.py`, CPU only.

**Box commands.** One ssh invocation runs the whole oracle: staging, HF
reference, candidate build and run, comparison, certificate. No machine path is
baked into the code or this document; `HOST`, `MODEL_ROOT` and `OUT_ROOT` are
supplied at call time, and `REPO`, `GLM_BATCH`, `GLM_CONTEXT` and
`GLM_MAX_MEMORY` have box-side defaults. Run the first block from any machine
that can reach the two-GPU box; the heredoc executes on the box.

```bash
HOST=${HOST:?ssh target of the two-GPU box, e.g. user@gpu-box}
MODEL_ROOT=${MODEL_ROOT:?pinned GLM-5.3 release directory on the box}
OUT_ROOT=${OUT_ROOT:?new output directory on the box}
GLM_BATCH=${GLM_BATCH:-4}      # 1/2/4/8; one build and one certificate per batch
GLM_CONTEXT=${GLM_CONTEXT:-1024}

ssh "$HOST" "MODEL_ROOT='$MODEL_ROOT' OUT_ROOT='$OUT_ROOT' REPO='${REPO:-}' \
GLM_BATCH=$GLM_BATCH GLM_CONTEXT=$GLM_CONTEXT GLM_MAX_MEMORY='${GLM_MAX_MEMORY:-}' bash -s" <<'BOX'
set -euo pipefail
: "${REPO:=$HOME/etx}"          # checkout on the box
cd "$REPO"
[ -x .venv/bin/python ] || { echo "no .venv in $REPO" >&2; exit 2; }
if [ -z "${GLM_MAX_MEMORY:-}" ]; then GLM_MAX_MEMORY='{"0":"80GiB","1":"80GiB","cpu":"32GiB"}'; fi
if [ -e "$OUT_ROOT" ]; then echo "OUT_ROOT must be new: $OUT_ROOT" >&2; exit 2; fi
mkdir -p "$OUT_ROOT"
STAGE="$OUT_ROOT/stage"; REF="$OUT_ROOT/reference.json"; BUILD="$OUT_ROOT/build"
QUALIFY="$OUT_ROOT/qualification"; CERT="$OUT_ROOT/certificate"

# 0. prompts: 1/2/4/8 distinct equal-length rows (reference.py:31-36), deterministic seed
.venv/bin/python - "$MODEL_ROOT" "$OUT_ROOT/prompts.json" "$GLM_BATCH" "$GLM_CONTEXT" <<'PY'
import json, random, sys
src, out, batch, context = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
vocab = json.load(open(src + "/config.json"))["text_config"]["vocab_size"]
rows = [random.Random(20261019 + r).sample(range(vocab), context) for r in range(batch)]
assert len({tuple(r) for r in rows}) == batch
json.dump(rows, open(out, "w"))
print(f"PROMPTS {out} batch={batch} context={context} vocab={vocab}", flush=True)
PY

# 1. stage layers 22-25 as a derived, pinned identity (refuses a window not straddling 23/24)
.venv/bin/python -u -m examples.glm53.stage --src "$MODEL_ROOT" --out "$STAGE" \
  --maxlen 1072 --layers 22-25 --gate slice
.venv/bin/python -m examples.glm53.prepare --require-model --stage "$STAGE" \
  --gate slice --out "$OUT_ROOT/prepare.json"

# 2. independent HF reference on the same four layers and prompts, 32 greedy steps;
#    writes $REF, $REF's .logits and the derived snapshot $REF.slice/
mkdir -p "$OUT_ROOT/offload"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u -m examples.glm53.reference \
  --src "$MODEL_ROOT" --prompts "$OUT_ROOT/prompts.json" --out "$REF" \
  --offload "$OUT_ROOT/offload" --max-memory "$GLM_MAX_MEMORY" \
  --layers 22-25 --gate slice

# 3. build the ETX PP2 model for this batch, then the candidate run (two repetitions)
.venv/bin/python -m examples.glm53.run build --stage "$STAGE" --build "$BUILD" \
  --out "$OUT_ROOT/build-log" --batch "$GLM_BATCH" --gate slice
.venv/bin/python -m examples.glm53.run qualify --stage "$STAGE" --build "$BUILD" \
  --reference "$REF" --out "$QUALIFY" --batch "$GLM_BATCH" --gate slice

# 4. offline comparison of each candidate record with the HF record
for r in 0 1; do
  .venv/bin/python -m examples.glm53.oracle --reference "$REF" \
    --candidate "$QUALIFY/run$r.json" --out "$OUT_ROOT/oracle-run$r.json" --gate slice
done

# 5. timing certificate: reruns qualification under $CERT, then writes bench_result.json
.venv/bin/python -m examples.glm53.bench --stage "$STAGE" --build "$BUILD" \
  --reference "$REF" --out "$CERT" --gate slice
BOX
```

**What each path produces.** `$OUT_ROOT/prompts.json`; `$STAGE/stage.json` and
`$OUT_ROOT/prepare.json` (staged identity validated); `$OUT_ROOT/reference.json`,
`$OUT_ROOT/reference.logits`, `$OUT_ROOT/reference.json.slice/` (the derived
snapshot HF loaded, with its hashes in provenance); `$QUALIFY/qualification.json`,
`$QUALIFY/run0.json`, `$QUALIFY/run1.json` and their `.tokens`/`.logits`/logs;
`$OUT_ROOT/oracle-run0.json` and `oracle-run1.json`; and the certificate
`$CERT/bench_result.json` (with the qualification it re-ran under
`$CERT/qualification`). Every stage is fail-closed: a mismatch exits non-zero
and, where the tool defines one, still writes its receipt. `OUT_ROOT` must not
already exist; the stage, reference, build, qualification and bench output
directories refuse to overwrite (the oracle reports overwrite the named file,
so name new paths). Paths must not contain single quotes or spaces (the ssh
argument is single-quoted). To run directly on the box, export `MODEL_ROOT`,
`OUT_ROOT` and `REPO` and execute the heredoc body unchanged.

`prompts.json` follows the existing contract: 1/2/4/8 distinct equal-length
arrays (`reference.py:31-36`), each `length + 32 <= max_position_embeddings`.
Everything else is the already-implemented strict path:
two independent processes, each reloading weights and resetting every cache
(`run.py:64-67`); `TOKENS-EQUAL` requires 32/32 per request (`run.py:75`);
full logits are compared even when tokens match, and `--logit-atol` turns that
report into a gate (`run.py:15-34,85-88`); `qualification.json` is written in
every outcome (`run.py:97`). The certificate re-runs the same qualification and
refuses a receipt that is not `TOKENS-EQUAL` (`bench.py:118-119`).

**Per-token metric and tolerance.** For every generated position t (1..32) and
request r, compare the full `[batch, vocab]` logit rows:
`cos = cos(logits_etx[t,r], logits_hf[t,r])` and
`rel = max|Δ| / max|logits_hf[t,r]|`. Pass requires **cos ≥ 0.999 and
rel ≤ 0.02 at every position** (§8) plus exact greedy tokens 32/32, in both
repetitions. Any one failure rejects the build; the numeric certificate is the
receipt, not the token count alone (`run.py:93-94`).

**Gate rule.** No timing number, tile change or schedule change is accepted
until this oracle passes on the same build identity. The full 45-layer,
2-GPU qualification (`docs/QWEN-BATCH-GLM53.md:190-214`) stays the end-to-end
gate; this slice is its affordable predecessor, not a replacement.

## 7. PP2 boundary cost

**What crosses.** `[B,4,4096]` BF16 = **32,768 B per request per step** plus
B int32 token IDs (`docs/QWEN-BATCH-GLM53.md:109-111`;
`examples/glm53/prepare.py:88,101-102`; B8 = 262,144 B + 32 B).

**Per-hand-off latency: [U].** No timing exists. The boundary probe compiles
and has simulator-static numbers only: `send_state` static queue estimate
**6.16 µs** in `docs/evidence/qwen-batch-glm53/glm-boundary-local-final.log:4`
— a simulator estimate, not a measurement. It must not be used as one.

**Step cost.** Per the design instruction, step boundary cost =
**24 × per-hand-off latency** [U]. Note once: 24 is the number of rank-0 layers,
while the implemented schedule crosses the boundary **once per decode step**
(`prepare.py:101`); the ×24 applies to a per-layer hand-off schedule, and 1× to
today's schedule. The probe measures the per-hand-off latency either way, so
both numbers become computable from it.

**The probe that measures it.** Already implemented as a two-launch ping-pong
with no weights: `send_state` (0→1, B×16384 uint16), `receive_state`, then
`send_token` (1→0), with events in between (`examples/glm53/boundary.py:10-30`);
it compiles for both devices (`local_boundary_rocm.sh`) but has no timing
driver. Required addition (XS–S): run 1,000 iterations per B ∈ {1,2,4,8}, five
rounds, host wall around each send/receive pair plus device `s_clock`
timestamps, report median and spread of the per-iteration latency and its
split (copy vs wait). Success: a measured median replaces this [U], and §3's
bubble and §9's step time use it; no code beyond the driver.

## 8. Numerics tolerance (fixed before optimising)

Mixed FP8/bf16 fusion changes summation order: GLM decodes FP8 to BF16 during
GEMV with FP32 accumulators (`docs/QWEN-BATCH-GLM53.md:89-91`), so a fused
path cannot reproduce operator-by-operator bit patterns. The bar is therefore
at the output, fixed now:

1. **Greedy token exactness: 32/32 per request** against HF greedy. This is
   the Qwen work's number — every Qwen benchmark certificate requires exactly
   32/32 per request (`examples/vllm_llm/bench.py:147-151`), and it is the
   claim-level gate in `CLAUDE.md` §1. GLM keeps it unchanged.
2. **Per-token output cosine ≥ 0.999 and relative max error ≤ 0.02**, the
   repo's existing boundary threshold (`docs/DEEPSEEK-PREP-TWO-GAPS-2026-10-15.md:41`,
   `docs/NEXT-GPU-2026-10-15.md:75`), applied per generated position.
3. **Per-layer cosine ≥ 0.9999** when a layer-level oracle exists — the Qwen
   numpy reference's per-layer precedent (`docs/PLAN-moe-en.md:83`) —
   reported from the slice's per-layer comparisons, not required at every
   optimisation step.

All three apply simultaneously, in both repetitions. Token equality alone is
explicitly not a numerical certificate (`run.py:93-94`). If a change passes
32/32 but fails cosine or relative error, it is rejected: the bar exists so
faster summation orders cannot pass on tie-breaking.

## 9. Timing harness

**What exists.** `examples/vllm_llm/bench.py` is the Qwen contract: one build
and one certificate per batch, never shared; it prints `BUILD-IDENTITY`,
`CORRECTNESS PASS` and `TIMING` lines and writes `bench_result.json` with
`median_decode_wall_ms`, `tokens_per_s = batch*1000/median`, `ms_per_token`,
`spread_ms = max−min`, the raw `samples_ms`, the build/source hashes and the
checked import positions (`bench.py:268-289`). Steps timed are 1024–1054, 31
samples (`bench.py:182-187`).

**What GLM had.** `examples/glm53/model_host.hip:75` recorded `start` but the
step line printed no time; `run.py` had no timing mode and no
`bench_result.json`. Both are closed 2026-10-19: the step line now carries
`ms=` (host wall of the two launches until both streams retire, before the
completion checks and their D2H copies), and
`examples/glm53/bench.py` implements the contract below. No timing number for
GLM has been produced yet [U]: the harness exists, the run does not.

**Spec: `examples/glm53/bench.py`** (size S [E]), reusing the certificate
contract:

- **Per batch** B ∈ {1,2,4,8}: its own build (`ETX_GLM_BATCH=B`), its own
  qualification, its own certificate; no gate or number shared between batches.
- **Identity:** stage and build validated before and after, source hashes
  (`model_build.source_identity`), reference hashes, and the exact build files
  (`run.py:46`), recorded in the certificate as in Qwen's `BUILD-IDENTITY`.
- **Timing definition:** host wall from the two launches to both streams
  retiring (`model_host.hip:75-79`: `wait_stream` on every rank, then the
  timestamp, then `after_wait`, whose 8-byte ctrl and optional trace D2H copies
  are therefore excluded), also excluding the returned-ID and logit copies; one
  sample per position, positions 1024–1054 (31 samples, matching Qwen's
  1024–1054 window).
- **Reported:** `median_decode_wall_ms`, `tokens_per_s = B*1000/median`,
  `ms_per_token = median/B`, `spread_ms`, every sample, conditions (two
  MI300X, context 1024, 32 greedy tokens, PP2, layer count), and the
  correctness receipt (32/32 per request, §8 metrics).
- **Refusal:** if any identity changes during the run, if the correctness
  receipt is not `TOKENS-EQUAL`, or if a step sample is non-finite or
  non-positive — the same fail-closed pattern as `bench.py:290-297`.
- **Implemented:** `model_host.hip:75-83` (print at `:83`) prints the step's `ms=`, `run.py`
  parses it (optional field, so old logs still qualify), and
  `examples/glm53/bench.py` runs one qualification per invocation and refuses a
  log without `ms=`, a missing/duplicated step, a non-finite sample, a changed
  identity or a receipt that is not `TOKENS-EQUAL`. It takes `--stage`,
  `--build`, `--reference`, `--out`, `--gate`; it writes `bench_result.json`
  under `--out`. Tests: `tests/test_glm53_slice.py`. No device code and no
  timing in this document.

A timing number from any other source is not accepted by this design.
