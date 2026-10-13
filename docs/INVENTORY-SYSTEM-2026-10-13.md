# The option inventory, re-anchored on the system goal (2026-10-13)

Goal (kailun, after jyoti, 2026-10-13): **a reusable inference system -- event tensor plus megakernels -- where similar
models plug in and run fast without per-model optimisation.** DeepSeek-V2-Lite is first only because fleet's kernels
exist; Qwen follows, then larger batches and larger models. Rules carried from today: every lever lands as a mode,
switch or check in the compiler, the plan or the session machinery, with the D-series shape (switch default off,
threshold written first, build-time check, negative test). **Hard constraint: weight precision does not change --
bf16 stays, FP8 is out** (权重精度先不改). The 300 tok/s DeepSeek target is superseded as the goal.

Tags: **[M]** measured (source given), **[M-partial]** measured but the run did not complete (not a result), **[E]**
estimate with its basis, **[U]** unknown, **[closed]** decided by a measurement or a pre-registered rule.
"System" = lives in `etx/` (IR, passes, codegen, runtime), in a DSL op every model uses (`etx/ops`), or in the
session machinery; "model" = a patch of one model's tiles or graph (`examples/dsv2lite/levers`, `fleet_shim.hip`).

## 1. Where today's work conflicts with the pinned claim and with the new goal

1. **Different comparison.** The pinned claim (`CLAUDE.md` section 1) is Qwen3-30B-A3B with ETX DSL tiles + vLLM's
   kernels, fused, against the SAME kernels replayed through a HIP graph, on host wall time. Today's DeepSeek numbers
   compare ETX builds of fleet's hand-written tiles with each other and with fleet's own runtime, on device time. No
   DeepSeek session today measured the claim's comparison (same kernels unfused), so none of today's numbers supports
   or weakens the pinned claim.
2. **Tile tuning.** `CLAUDE.md`: "Kernel quality is not ETX's claim; tuning our own tiles to win a comparison still
   needs the owner's request." The five fleet levers, D3's chunk wrappers and D4's bodies are tuning of one model's
   hand kernels. They had the owner's request (the D-series), but under the new goal they do not transfer: they are
   patches of `fleet_kernel.hip` copies, valid for DeepSeek only.
3. **What did transfer today** is the infrastructure around them: two compiler coherence modes, a load-scope
   microbenchmark, build-time plan/tile checks, a placement check, one-verdict-per-arm session reporting, hard
   preflight gating, box-side hashing. Section 2 marks each.
4. **DeepSeek is outside the generic path.** Qwen, Mixtral and Phi run from `examples/llm/configs/*.json` through the
   DSL ops; DeepSeek runs on `examples/dsv2lite` because MLA attention has no DSL op
   (`docs/PLAN-generic-decoder-2026-10-08.md:292,449`). Bringing it in (G1) is the system-level form of every DeepSeek
   item below.

## 2. Inventory

### 2.1 Today's DeepSeek items (fleet-tile path), all bf16

| id | item | class | transferable form | status [tag] |
|---|---|---|---|---|
| D1 | safe acquire (`buffer_inv sc1`) as the default; relaxed only as a named arm | **system** | acquire mode is a codegen default + a session rule (`examples/dsv2lite/acquire.py`, test) | cost of safety 232 us/token vs relaxed, spread 11 [M] (run 2, pairs_relaxed) |
| D2 | `coherence="coherent_loads"`: sc0 sc1 payload loads, write-back kept | **system** mode; model accessors | `etx/passes/p4_schedule_mode.py`, `etx/codegen/kernel.py`; fleet accessors by lever (model) | NO-GO: +9 us vs default, 237 us behind relaxed [M] (run 2) [closed] |
| D2b | `coherence="coherent_l1"`: agent-scope payload loads, no per-task invalidate, no added write-back | **system** mode + bench; model accessors | compiler mode + `coh_check.py` + `bench/sync/sync_bench.hip loadscope` (system); `levers/l1_loads.py` (model) | bench: sc1 loads 0 stale words without invalidate [M] (run 5); arm: l1 3.329-3.341 vs default 3.661-3.671 ms in 3 of 5 rounds [M-partial]; best arm + D2b never ran [U] |
| D3 | gate_up -> down in four 352-row K-chunks | **model** (in-body counters outside the IR) | system form: one event per K-chunk in the IR, waits placed by the compiler (not built) | built, locally proven, unmeasured [U]; fleet's 512-row tiling 4.020 ms, cost 0.42 ms [M, fleet] |
| D4 | placement: shared-expert XCDs start without routing (0 slice rows) | **model** placement; **system** check | `d4_placement.check/verify_graph` generalises to an IR validator rule (G4) | alld4r0 -61 us vs all, spread 8, KEEP [M] (run 4) |
| D4s | routed slices on the shared XCDs, batched (Astra) | **model** | — | unmeasured [U]; the unbatched form +271 us vs default [M] (run 2) [closed] |
| D5 | compiler weight prefetch (p7 + kernel hook) with a fleet prefetch body | **system** pass/hook; model body | `etx/passes/p7_prefetch.py`, `etx/codegen/kernel.py` (system); `fleet_d5.hip` (model) | unmeasured [U]; fleet's register-held form 4% slower [M, fleet `docs/STATUS.md:30`] |
| L1-L7 | fleet levers Q, Qall, N, Nall, argmax, topk, prep | **model** | the concepts exist as DSL op variants (topk KEY32, norm tree, gemv tails) | singles vs default: topk -64, Nall -41, N -39, prep -31, argmax -6, Qall -5, Q +16 us [M] (run 2); all five -108 us, GO [M] (run 1) |
| T1 | ROCm 7.2.3 compiler instead of 7.14 | **system** (toolchain) | build image choice, applies to every model | default -62 us, spread 14 [M] (10-13 follow-up E2); on the best arm [U] |
| P1 | expert-weight prefetch by routing prediction | system idea | — | NO-GO [E]: ceiling 97.8 us/token, needs >= 30.7% exact-set persistence [U] (`examples/dsv2lite/results/EXPERT-PREFETCH.md`) [closed] |
| R1 | relaxed acquire on the best arm (allrelaxed) | measurement only (unsafe) | — | -326 us vs default, spread 8 [M] (run 3): the prize a safe cheap acquire can win |
| M1 | per-stage stamps, ledger, stage table | session tooling, partly **model** (phase names) | `etx/tools/timeline.py` is the system form | run-1 stamps intact [M]; D4 arms' stamped record never measured [U] |
| M2 | one verdict per arm, preflight hard-gating, box-side hashes, decision register | **system** (session machinery) | `examples/dsv2lite/inplace_session.py` today -- model-specific location, model-agnostic logic (G5) | built and tested; two session bugs found by it today [M] |

### 2.2 The Qwen / generic-path items (from the existing plans), all bf16

| id | item | class | status [tag] (source) |
|---|---|---|---|
| Q1 | top-k KEY32 (DSL topk flag) | **system** | spent: 0.231 ms wall [M] (`docs/KEY32-MEASURED-2026-10-11.md:3-7`) |
| Q2 | interleaved GEMV rows (variant registry) | **system** | absent [U] (`docs/SCOPE-300TOKS-2026-10-11.md:41`) |
| Q3 | ETX attention tile (prep, attention, merge) | **system** (op family; also the MLA route, G1) | absent; 2-4 agent-days [E] (`SCOPE:42`) |
| Q4 / A1 | host reset submission | **system** (runtime) | 0.049 ms [E], ceiling 0.039 [M] (`docs/PLAN-300-CERTAIN-2026-10-12.md` A1) |
| Q5 | GEMV body excess | **system** (gemv op) | breakdown: MoE 429, qkv 199, o_proj 14 us per step (`SCOPE:44`, from `docs/FLOORS-2026-10-08.md:263-265`) |
| A2 / C6 | attention boundary fusion, head-local protocol | **system** (compiler protocol + hook) | 0.096 ms gross [E], net [U] (PLAN-300-CERTAIN A2) |
| A3 / P | prep body, issue order | **system** (DSL attnprep) | 0.126 ms [E], <= 0.034 named (A3; `docs/LEVERS-MORE-2026-10-12.md` P) |
| A4 / C5 | cached norm-weight hoist (`-DETX_DSL_NORM_HOIST`, built, off) | **system** | 0.005-0.035 ms [E] (`docs/PLAN-300TOKS-2026-10-11.md` 5.3) |
| A5 / C1 | narrow router (`ETX_VL_ROUTER_NARROW=1`, built, off) | **system** | up to 0.193 ms [E] (5.3) |
| A6 | MoE-up body: XOR LDS layout (S) or segment reuse (X) | **system** (DSL gemv) | 0.015-0.023 ms [E] (A6) |
| A7 | top-k residual after KEY32 | — | no lever named (A7) |
| B1 | vendor attention + reduce (AITER) | system (importer) | closed: AITER 11.42 vs ETX 10.65 us/layer, slower (B1) |
| B2/B3 / X7 | late and free event edges, frozen layout | **system** (scheduler/layout) | <= 0.039 ms [E] (B2) |
| R, T, A | issue-order changes in reduce, top-k, attention | **system** (DSL ops) | each <= 0.013 ms [E], below the 0.030 KEEP bar (`LEVERS-MORE` section 2) |
| I1 | import AITER CK MoE | **system** (importer) | [U]; bound 0.158 ms (`docs/LEVERS-300TOKS-2026-10-11.md` row 6) |
| H1 | cross-step resident loop | **system** (runtime) | increment 0.012 ms [M, older build] -> closed by the bar (`PLAN-300TOKS:194`) |
| X1 | cross-phase prefetch into idle XCDs / bandwidth (P1-P3) | **system** (scheduler) | bound 0-121 us, central 63 us [E] (`docs/XCD-PREFETCH-2026-10-11.md` section 4) |
| F1 | FP8 weights | — | **OUT: violates the bf16 constraint** (`docs/SPEC-fp8-2026-10-11.md`) |

### 2.3 The growth items the goal names

| id | item | class | status [tag] |
|---|---|---|---|
| G1 | DeepSeek into the generic path: an MLA op family in the DSL, the model from a config | **system** | absent; TileLang publishes an MI300X MLA decode tile (`PLAN-generic-decoder:418-420`); size [U] |
| G2 | larger batches: the multi-request graph (`ETX_DSV2_REQUESTS=2` exists for DeepSeek) as a generic plan option | **system** | two requests interleaved, best build: 1.30x fleet's and 1.37x ETX's single-request throughput, 3% below vLLM at batch 2 (5.45 vs 5.32 ms per request) [M] (`docs/ETX_Technical_Design.md` 15.12, line 1079); the 1.72-1.78x estimate did not hold [closed] -- the two requests fell into lockstep |
| G3 | larger models (multi-GPU) | **system** | multi-device plan exists for gemm_rs (`examples/gemm_rs.py`); decode models [U] |
| G4 | placement constraints as an IR validator rule ("stages sharing a domain-local buffer share a domain"), generalising D4's check | **system** | D4's model-level check + negative tests exist [M, tests]; IR form not built |
| G5 | the session machinery as a model-agnostic package (verdict register, hard preflight gates, box-side hashes, stamped traces) | **system** | logic built for DeepSeek (M2); package not built |
| G6 | D2b's mode for the DSL ops: an agent-scope variant of `ETX_DSL_COH` so Qwen gets `coherent_l1` | **system** | not built; the bench result (sc1 loads safe) is model-independent [M] |

## 3. Proposed order under the new goal (each with its threshold written before its code)

1. **Measure what is already built, in the paid hour** (`examples/sessions/ds_hour.py`): the D2b rerun decides whether
   `coherent_l1` is the system's acquire mode. The DeepSeek-only arms in that session (D3, D4s, D5 on the fleet path)
   stay, since they cost about 4 GPU-minutes and are built, but they are model results.
2. **G6 then G2's cross-model check:** the same `coherent_l1` mode on Qwen's DSL tiles, measured under the same rule on
   both models. One compiler mode, two models, no per-model patch -- the first direct evidence for the goal.
3. **G4 and G5:** move the placement check into the IR validator and the session machinery out of `examples/dsv2lite`,
   each with its negative tests, so the next model inherits them.
4. **G1:** an MLA op family in the DSL, so DeepSeek leaves its hand-tile path; the fleet levers then retire.
5. **G2:** the multi-request graph as a generic plan option (larger batches), on the register budget it needs.

Nothing in this order changes weight precision.

## 4. Can the generic-path levers be DIRECTLY reusable? Classification and plan (2026-10-13, kailun)

**The criterion for "reusable": ONE mode, TWO models, zero per-model patch, measured under ONE decision rule.** One
mode = one switch spelled once, in the compiler / plan / runtime, that every graph builder inherits. Two models = it
is built and measured on two different model graphs. Zero per-model patch = flipping it edits no model file. One
rule = the same pre-registered KEEP / NO-GO rule, same basis, on both. Anything less is "a mechanism that could be
reused", not "reusable".

Three classes. Facts used: `examples/vllm_llm/model.py` is the SHARED graph builder of the generic family (Qwen2.5/3,
Mixtral, PhiMoE, from configs; its docstring and lines 113, 246, 361, 449-460), not one model's file;
`examples/dsv2lite/` is DeepSeek-only (fleet tiles). Both compile through `python -m etx compile` (`etx/tools/cli.py`).

- **(a) reusable now**: mechanism in `etx/` AND switched by the plan / feature manifest (`etx/codegen/manifest.py`), so
  a new model inherits it with no patch.
- **(a') reusable mechanism, per-builder switch**: mechanism in `etx/`, but the switch is an `ETX_VL_*` (or
  `ETX_DSV2_*`) variable read by one builder. Lifting = move the switch into the plan options / manifest. Cheap.
- **(b) model patch**: the mechanism itself is a patch of one model's files. Lifting = re-implement it in `etx/`.

| row | class | why (location of mechanism / switch) | lift |
|---|---|---|---|
| D1 acquire default | a | codegen default (`etx/runtime`, lowering); relaxed is a compile flag | — |
| D2 `coherent_loads` | a' / b | compiler mode (a); its use is decided in `dsv2lite/graph.py` (`ETX_DSV2_COH`) and the accessors are a fleet lever (b) | closed NO-GO; nothing to lift |
| D2b `coherent_l1` | a' / b | compiler mode (a); switch `ETX_DSV2_COH=l1` (a'); accessors: fleet lever `l1_loads` (b); the DSL ops have no agent-scope variant | **G6** |
| D3 K-chunks | b | fleet in-body counters, dsv2 shim wrappers | IR chunked edges (later) |
| D4 / D4s placement | b + a(check idea) | dsv2 graph + shim; the check is `dsv2lite/d4_placement.py` | G4 (IR validator rule) |
| D5 prefetch | a' / b | p7 + hook are `etx/` (a); switch `ETX_DSV2_D5` (a'); body `fleet_d5.hip` (b) | switch -> plan option; body -> DSL `touch` |
| L1-L7 fleet levers | b | patches of a fleet copy | retire with G1 |
| T1 toolchain | a | build image | — |
| M1 stamps / stage table | b | `dsv2lite/ledger.py`, phase names | `etx/tools/timeline.py` is the system form |
| M2 verdict / gates / hashes | b (logic is model-agnostic) | `dsv2lite/inplace_session.py` | G5 |
| Q1 KEY32 | a' | `etx/ops/topk*.py`, `tile_util.h`; switch via `ETX_VL_*` / `ETX_TOPK_KEY32` in the vllm builder | switch -> manifest |
| Q2 interleaved rows | (absent) | would be a variant in `etx/ops/gemv.py` + `etx/select/variants.py` | build it as (a): second experiment |
| Q3 attention tile | (absent) | would be an `etx/ops` family | G1 shares it (MLA) |
| Q4 / A1 host reset | a | `etx` runtime / host | — |
| Q5 GEMV body | a | `etx/ops/gemv.py` | — |
| A2 / C6 head-local protocol | b | `examples/vllm_llm/head_local_protocol.py` | protocol -> compiler |
| A3 / P prep order | a | `etx/ops/attnprep.py` | — |
| A4 / C5 norm hoist | a' | `etx/ops/local_fold.py`, `gemv.py`; switch `-DETX_DSL_NORM_HOIST` from the vllm build | switch -> manifest |
| A5 / C1 narrow router | a' | `etx/ops/gemv.py`; switch `ETX_VL_ROUTER_NARROW` in `vllm_llm/model.py` | switch -> plan option |
| A6 MoE-up layout | a (if built) | `etx/ops/gemv.py` | — |
| R / T / A issue order | a | `etx/ops` | — |
| B1 vendor attention | a | `etx/importer` | closed |
| B2/B3 / X7 layout | b | `apply_layout.py` hook, not wired | -> scheduler pass |
| I1 import CK MoE | a | `etx/importer` | — |
| H1 resident loop | a | runtime | closed |
| X1 cross-phase prefetch | a (not built) | scheduler | — |
| touch / argpf / bar2 / chain / xcd | a' | `etx/codegen/kernel.py`, `etx/ops`; switches `ETX_VL_*` in the vllm builder | switch -> plan option |
| G2 multi-request | a' | plan supports requests; switches `ETX_VL_REQUESTS` / `ETX_DSV2_REQUESTS` | one plan option |

Count: of the rows with a built mechanism, most generic-path levers are (a) or (a'); the DeepSeek fleet-path items are
(b) by construction. The cheapest systemic step is therefore to lift switches (a' -> a); the first experiment lifts
the one with a measured payoff: `coherent_l1`.

### 4.1 G6 -- `coherent_l1` as ONE compiler mode on Qwen AND DeepSeek (thresholds written before the code)

Design:
- **One switch, in the compiler**: `PassOptions.coherence` / `etx compile --coherence {declared,l1}`, default from the
  environment variable `ETX_COHERENCE` (unset = `declared` = today, byte-identical). With `l1`, every grid whose tile
  declares the capability gets `coherence="coherent_l1"`; others are untouched.
- **Capability, declared by the tile family, not the model**: `TileBody.coherence_caps`. The DSL family declares it
  in the shared builder's tile constructor (`coh_of` in `examples/vllm_llm/model.py`, one place for every generic
  model); the fleet family in `examples/dsv2lite/graph.py` `_tile`.
- **Accessors**: DSL ops get an agent-scope variant of `etx/ops/coh.py` (loads `sc1` through address_space(1), stores
  plain), published by the plan through the feature manifest (`coh_l1` -> `ETX_DSL_COH_L1`; takes precedence over
  `ETX_DSL_COH`). Fleet tiles use the existing `l1_loads` lever, which the DeepSeek build adds whenever the plan is
  `l1` (one rule in `examples/dsv2lite/build.sh`); `coh_check.py` keeps refusing a plan without its tiles.
- Basis for safety: the loadscope bench (sc1 loads, no invalidate: 0 stale words, run 5).

**One decision rule for both models** (device time; 5 interleaved rounds, order reversed on odd rounds): the arm is
the model's reference build with `ETX_COHERENCE=l1`, the reference is the same build without it. **KEEP if every
pair is faster, median >= 30 us and median > 2 x spread; otherwise NO-GO.** Every invocation must pass the model's
correctness gates: DeepSeek 32/32 tokens + 27/27 layers; Qwen 32/32 tokens + the same-build import certificate (97/97,
`examples/vllm_llm/import_certificate.py`). References: DeepSeek `default` (and, separately, `alld4r0`); Qwen the
kept build (`examples/vllm_llm/kept_build.py`). Reusability (the criterion) needs the mode to build and pass the gates
on both with the one switch; a performance claim needs KEEP on both.

CPU proof before any box: default plans byte-identical (DeepSeek 09011c20; Qwen's kept-build plan hash recorded before
the change); both models compile with `ETX_COHERENCE=l1` and nothing else changed; the generated code has no acquire
on the declared tiles and no added write-back; ROCm 7.2.4 builds at 0 scratch; negative tests: a plan in `l1` with
tiles lacking the accessors is refused at build time for both families.

**Status (2026-10-13 evening).** DeepSeek half measured: the paid hour's `pairs_l1` (examples/dsv2lite/results/
2026-10-13-hour) gives +330 us vs default, spread 13 us, every pair faster: **KEEP under this rule** (re-derived from
the stored pairs; DeepSeek's `ETX_DSV2_COH=l1` is the same compiler mode as `ETX_COHERENCE=l1`, proven on CPU earlier).
Qwen half prepared, not measured (no paid GPU without approval):
- `examples/vllm_llm/arms.py`: the Qwen arm registry (etx/session/experiment.py). `kept` = the kept recipe, `kept_l1` =
  the same build + `ETX_COHERENCE=l1` and nothing else (tested); the rule above, verbatim, decided in integer ns.
- `examples/vllm_llm/g6_session.py` + `examples/sessions/qwen_g6.py`: preflight (frozen shape, bf16, golden), both
  builds, the same-build import certificate (positions 0 and 1040) + 32/32 + CHECK-DSL per build, then five
  interleaved rounds. Dry run: 6 jobs; every GPU job reads what it runs.
- `examples/vllm_llm/rocm_build_check.sh qwen3-30b-a3b arm:<name>` builds a registry arm with the VM's compiler.
  ROCm 7.2.4, the full kept recipe (router narrowing included): kept and kept_l1 both 128 VGPR, occupancy 4, 7 VGPRs
  spilled / 32 B/lane scratch (a property of the kept recipe, unchanged by the mode), identical plan (60362 tasks,
  436 events). Megakernel ISA: system-scope loads `sc0 sc1` 82 -> 0, agent-scope `sc1` loads 1305 -> 1387 (+82),
  `buffer_inv` 98 -> 98 (imported tiles keep their acquire), `buffer_wbl2` 50 -> 49 (one write-back fewer, none added).
- Estimated session cost [E]: 2 certificates (3 host invocations each) + 10 timed invocations, each under the 240 s
  invocation deadline; the session's hard stop is 3600 s.

### 4.2 Q2 -- interleaved GEMV rows (second, if cheap)

A new row-ownership variant in `etx/ops/gemv.py` registered in `etx/select/variants.py` (class a from birth); its rule
and cost estimate get written here before its code.
