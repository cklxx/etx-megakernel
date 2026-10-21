G09/G10/G11 are isolated from the production emitter and use the frozen kept
revision b573664faa4e7988dff353bc03aca4ce8561a729. No GPU results are supplied.

Local preparation:

```
PYTHONPATH=. /Users/ckl/code/etx/.venv/bin/python -m pytest -q tests/test_experiment_row_ops.py
PYTHONPATH=. /Users/ckl/code/etx/.venv/bin/python -m examples.experiment_program.row_ops dry-run --out build/experiment_program/row_ops
PYTHONPATH=. /Users/ckl/code/etx/.venv/bin/python -m examples.experiment_program.row_ops build --out build/experiment_program/row_ops
```

The build uses an existing offline etx-rocm22 image, read-only existing local
linker dependencies, ROCm7.2.4 clang22 and lld22, and no GPU/network. Every original
source, assembly, object, HSACO, native replay host, ISA, metadata and full compiler
log is retained. identity.json seals them. Device resource rejection is independent
of correctness or performance. Actual residency remains unmeasured.

G09 retains exact fp32 activation segments in each existing whole-row wave's
registers under K2048/64 combined rows/16 waves/EPL8/NB1 guards. Original ordered
segment dots, WSUM, +0 partial initialization, scratch gather, gate/up rounding,
SiLU, publication and generic fallback remain. Requested operand reads fall by
four only in the logical address model, not a claim about physical LDS traffic.

G10 starts with the existing exact guarded up clone. The prepared down extension
keeps K768/64 rows/16 waves/EPL4, the same mapping, original wave reduction, initial
+0 and 16 ordered virtual-zero additions, and original weighted output rounding.
It removes only the partial matrix. All gate/up joins and LDS reuse barriers
remain. `g10_down` is not eligible until `g10_up` has five exact positive pairs
passing the local admission bar; CLI requires an actual up-admission receipt.

G11 instruments the real current `dsl_stage_norm_at` folded qkv path. Seven
landmarks mark first input issue, fold begin, residual-load issue, ordered
read/fold end / norm begin, norm-ready after the original barrier, output issue,
and publication retirement after the original fence. The R-batched slot reads
and fold are interleaved: their issue interval must not be reported as an
exclusive load/compute decomposition. There is no inserted forced drain. K2048
vector norm has four active load waves; idle norm waves have no residual-load
stamp. Per-wave private probe LDS never aliases xs, red or another consumer's
scratch. One selected task/wave has internal probes per invocation; all 16 waves
have completion companions. The declared observation population scans every
executed task/wave in one resident-input process, including the final consumer.

The G11 g11_control/probe/empty triplets are qualification observations, not five
optimization pairs. Qualification requires exact outputs, complete selected
population and all-wave completion, identical native allocations and at most
0.5 us perturbation in every declared observation. Raw final compiler counts differed (fold58VGPR/77SGPR versus probe58/81).
The dedicated g11_control and empty controls now explicitly reserve the same
SGPR allocation as the probe, through assembly allocation directives and
corresponding metadata. Native fold notes are all58VGPR/81SGPR, with equal actual
allocation descriptor words, LDS and zero scratch. Original raw assembly is
retained by SHA; ELF .text bytes of probe/empty are byte-identical before/after
reserve matching. The optimization control receives no such padding. No causal
exposure claim is issued merely because this build succeeds; runtime perturbation
and byte/coverage qualification are still mandatory. Any mismatch is UNQUALIFIED. Even a qualified interval
is exposure, not legally removable work or predicted model savings. G11 remains
opportunistic under a matching G12 trace and an active sharing decision.

Native snapshot path:

* `adapt-moe --pack ORIGINAL_BF16_CAPTURE/pack.json --out capture/pack.json` reuses
  original existing bf16_300 captures for G09/G10, including slice hashes.
* `capture-host --original examples/vllm_llm/host.hip --out build/capture_host.hip`
  emits an isolated checking-host source. The G12 checking-host build may use this
  source instead of the original host; no production source changes. Existing
  model inputs, original import/DSL checks and external goldens remain required.
  Set `ETX_ROW_CAPTURE_DIR` to a new existing directory and
  `ETX_ROW_CAPTURE_LAYER` to 1,24 or47, run the original checking host at positions
  0 and1040, then `seal-capture --directory capture`. This snapshots native
  W/x/output/rows/residual/norm pointers, exact aliases, and independent original
  output, normalized input and residual bytes. Weight files remain referenced by
  exact offset and hash; no model-sized duplication.
* Replay schema is `etx-row-replay-v1`, with native argument hash, all live pointer
  relocations, immutable buffer slice hashes and independent original reference
  slices. Fold capture requires real predecessor/queue metadata in addition to
  the native predecessor identity; a printed phase name is not a queue proof.

Future authorized local device execution (not executed during preparation):

```
python -m examples.experiment_program.row_ops execute --variant g09 \
  --pack capture/pack.json --build build/experiment_program/row_ops \
  --output results/g09_pos0 --deadline 120 --dry-run
```

Remove `--dry-run` only in a separately authorized GPU session. Native replay
loads all captured inputs once, restores mutable state using resident device
copies, preserves pointer aliases, runs declared warmups and deterministic
predeclared arm orders, byte-checks every invocation and saves every raw output,
task/wave record and host/device sample. It has stream and outer process
deadlines and progress output. Unwritten output bytes cannot inherit another
arm's values. Output prefixes cannot overwrite prior evidence. Optimization
pairs are exactly five. No shorter/incomplete run receives a verdict. Both
positions and signed-zero/cancellation fixtures are correctness gates before
any model admission; local CPU tests do not replace device correctness.

The standalone G11 replay starts at current-consumer issue. A native input
snapshot does not recreate the preceding expert-down publication timestamp.
`plan_edges` binds exact static queues and event producer identities; a complete
predecessor-publication interval additionally requires the matching G12 current
trace. The standalone interval is never relabeled as that full-chain window.
Use `seal-capture --directory capture --plan ACTUAL_BUILD/plan.json`; the capture
population must equal the selected emitted grid. The checking-host source also
supports `ETX_ROW_CAPTURE_ROLE=dsl_moe_up` and `dsl_moe_down` to capture the full
192/256-task original window. Single-expert captures are rejected for decisive
streaming-window replay.

Separate `_check` native exports capture gate/up's separately rounded fp32 LDS
values and the current folded consumer's own normalized staging. The resident
host compares these with the unchanged original export before any warmup or
timed invocation. These extra stores are absent from the decisive exports.

Winner integration: `row_ops.integration.emit_winner(variant, original_tiles, out)`
accepts G09/G10_UP/G10_DOWN, preserves the complete original DSL TU outside the
mechanism, emits no probes/check exports, and returns the exact
`ETX_VL_DSL_TILES` injection path plus original/candidate source hashes. Prior
certificates are explicitly invalidated. Generation is local preparation;
execution still needs identity-bound admission and all fresh G12 gates.

Output reservation estimate for G11: the default 256-task x16-wave internal
observation scan declares4096 triplets. Saving all three arms' complete record
arrays needs approximately3.6GB per window, plus exact output captures. This is
an estimate of future diagnostic storage, not measured GPU time or a current
write. A future session must reserve that space before execution; six
layer/position windows scale separately. Preparation produced approximately20MB
of build artifacts and no GPU outputs.

Each output-only native reference region is initialized to the bitwise
complement of its independent original reference before every correctness and
timed invocation, using resident device copies outside the measured interval.
Thus even zero outputs cannot pass through unchanged initialized output bytes.
A region overlapping any original live read span preserves original input
bytes instead; disjoint output slices in the same canonical allocation are
still poisoned. Split relocations of one original native allocation are
rejected; no alias input is poisoned. Full declared task/wave records and
unchanged guarded source ownership supply execution coverage for these aliases.
