# Shared state: local gate

This gate implements the local part of candidate 3 in `NEXT-2026-10-03.md`.
It does not rent a GPU or execute a kernel.
The round48 evidence came from the main checkout at `results_2026-10-01/round48/`.
That archive is absent from this worktree.

Run the gate and write the inventory:

```sh
/Users/ckl/code/etx/.venv/bin/python -m etx.tools.shared_state --output docs/shared-state-inventory.json
```

The [inventory](shared-state-inventory.json) gives source references for writers, readers, and order at each site.
The tool derives LDS sites from emitted tile source and runtime headers.
It follows arena aliases and helper arguments to a fixed point.
It keeps source expressions for the thread, wave, and row indices.
The local C preprocessor selects the coherent and scalar paths before the scan.
The inventory also includes the software import barrier path.

The tool derives argument regions from the Python graph, host plan, and import layouts.
Each argument region lists its byte range, tile reader, pre-wait reader, and task events.
The host upload orders these immutable blocks before the kernel launch.
A task event does not protect a pre-wait read.
Thus, an argument block must remain immutable for the full step.

## Required checks

`tests/test_coverage_gate.py` runs the gate through the normal pytest acceptance path.
It builds the Qwen3-30B-A3B, Mixtral-8x7B, and Phi-3.5-MoE graphs through `build()` and `compile_graph()`.
The full cases include all 48, 32, and 32 layers, respectively.
The condition is batch 1, maximum length 1072, and the default tile path for each model.
The fixtures use source-only import descriptors.
No weight values affect these address checks, so the fixtures allocate no weights.

Required output extents come from model dimensions, independent of the candidate launch extent.
The checker visits every output element in these extents.
It checks dense rows, expert slots, attention partitions, norm replicas, native outputs, and fold side outputs.
The attention extent includes every partition that can execute as the sequence grows to the maximum length.
For cache writes, the extent is the current token's key and value slice.
Existing cache positions belong to earlier steps.

The checker expands actual task coordinates with a local copy of the adapter's block rules.
It does not call the adapter implementation.
It rejects holes, duplicate writers, absent static queue entries, and absent tasks.
It also checks argument ranges, replica strides, event counts, and cross-domain event scope.
It checks that compilation preserves required API edges against an independent copy of the input contract.
A removed edge fails even if another data structure still contains its old copy.

The source checks retain text-order checks for writes, barriers, and reads in the supported tile protocols.
The dense, expert, fold, and import checks use source expressions and text order.
These checks do not prove control flow for those stores or the import protocol.

The statement parser in `etx/tools/source_control.py` adds control-path checks for nine LDS helper and tile functions.
It parses blocks, branches, and loops after preprocessor branch selection.
Each barrier must be a separate statement on its specified control path.
The approved conditions and loop bounds are uniform across the workgroup in the supported callers.
The parser rejects barriers in dead branches, branches that only some threads reach, or new control paths.
It also rejects hidden barrier calls and unsupported control statements in these functions.
It rejects early exits from these protocols.
Only the final return in each block reduction is allowed.

The LayerNorm branch must start with its barrier.
The native checks require the exact output stores and their thread or loop paths.
These checks include `token_out[t]`, optional `token_in[t + 1]`, timestamps, metadata, and the residual copy.
The graph checks still require one native task per output region.
A source change that alters these expressions needs a contract review.
This is a conservative source check, not a general C++ proof.

## Sites and highest risk

| Shared region | Writers | Readers | Required order |
|---|---|---|---|
| `xs`, with alias `xl` | Stage threads and norm-fold threads | Dot-product waves and LayerNorm mean threads | Stage barrier, then fold exit barrier |
| `part`, with aliases `pl` and `pb` | Zero-fill threads and lane 0 of each wave | Reduction threads | Zero-fill barrier, reduction barrier, reuse barrier |
| `g` and `gb` | Gate and up epilogues | SiLU output threads | GEMV epilogue barrier before each return |
| `red` | Lane 0 of each wave | All reduction threads | Write barrier and reuse barrier |
| `lids` and `lwts`, passed as `ids` and `wts` | Wave 0, lane 0 | Expert tasks within the workgroup | Top-k fold exit barrier |
| `sv` and `si` | Argmax threads | Partner threads in the reduction | Initial barrier and a barrier per reduction step |
| `etx_lds_arena` | Imported bodies and DSL bodies | Those bodies and later tasks on the workgroup | Import rounds and task boundaries |
| `etx_imp_bar_count`, `etx_imp_bar_gen` | Slot wave leaders and import initialization | Slot wave leaders | Workgroup release/acquire protocol |
| `etx_imp_done`, `etx_imp_exit`, `etx_imp_nbar` | Wave leaders and import initialization | Import exit protocol | Counted workgroup barrier |
| Fold `resid_out` and `xout` | Dense block 0 | Later norm and expert phases | The producer's output event |
| Phase outputs | Tasks in the generated graph | Successor phases | Per-coordinate ETX events in the inventory |
| Imported argument blocks | Host `Args` methods and upload | Pre-wait hook and tile body | Host upload before launch, then immutable storage |

The highest race risk is `dsl_stage_norm_at:xs` before the LayerNorm mean.
The coherent path gives each thread eight values to write.
The mean loop gives each thread a different index sequence to read.
One wave can therefore read values before another wave writes them.
Round48 showed that isolated checks did not expose this failure.
The negative test removes this exact barrier and requires a failure with `dsl_stage_norm_at:xs` in the diagnostic.

Other negative tests remove a real output task, a tile store, an event edge, an event count, or a queue entry.
A vocabulary test removes 1536 rows from the launch extent but retains the full model vocabulary as the required extent.
It fails without an argmax reference.
The tests also remove each checked LDS barrier separately.
Three regression tests disable all barriers, restrict a reduction barrier to lane zero, and delete the native token store.
Each test failed before the source fix.
Separate mutations test each source guard without an error from another guard.
The LayerNorm mutation moves its barrier outside the mean branch but preserves text order.

## Limits

The row checker is an address model of the supported host and tile rules.
It does not execute imported machine code or prove cache visibility.
The import descriptors in these fixtures do not replace an ABI check against a real kernel import.
The graph contract checks compiler preservation of declared edges.
It cannot prove that an arbitrary new pointer alias has the correct declared dependency.
The LDS inventory is a source index, not a complete C++ alias analysis.
The statement parser treats expressions as tokens.
It does not prove variable values, helper effects, pointer aliases, or arbitrary C++ behavior.
The uniformity of approved conditions and bounds remains a contract with the supported callers.

### Named gap: imported and dense build coverage

The local graph fixtures cover only the DSL path of the three MoE models listed above.
`build_local()` clears `ETX_VL_IMPORTED` and other `ETX_VL_*` options before each build.
The row checker rejects GEMV exports that do not start with `dsl_gemv`.
Thus, this gate does not cover the all-imported build with `ETX_VL_IMPORTED=1` or dense models such as Qwen3-8B.
A missing output in either build can remain undetected by this suite.
These builds need separate launch descriptors and output contracts before this gate can cover them.

The local checks do not establish a new GPU correctness result.
The planned 1000 repetitions with real weights and the 2 bf16 ulp limit remain GPU work.
No GPU session or performance claim is part of this change.

## Local verification

On 2026-10-06, the final local suite passed with 349 tests and 6 skips.
The command used `/Users/ckl/code/etx/.venv/bin/python -m pytest -q`.
This run included all 74 tests in `tests/test_coverage_gate.py`.
The three requested source mutations produced 9, 1, and 1 errors, respectively.
The first full run found one error in a new mutation test.
A source comment blocked that test's replacement expression.
The corrected test uses preprocessed source and passes for both activation paths.

The existing inventory records zero errors, 76 LDS sites, 1416 output sites, and 2113 argument regions.
This change does not regenerate that inventory.
No GPU time or rental cost occurred.
