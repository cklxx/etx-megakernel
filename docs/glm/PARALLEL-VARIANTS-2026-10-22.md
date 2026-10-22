# GLM parallel variants — 2026-10-22

## Status and scope

**CPU options/manifest generation completed; variants remain uncompiled,
untested and unmeasured.** The lead ran only the options generator; all regression
tests, native compilation and GPU execution remain unrun. No GPU, SSH,
provisioning or stopped-box access occurred. Options files are not build,
capacity, numerical or speed certificates. The surrounding production integration
still needs validation.

The measured default remains `devices=2, phased=1, parallel=0, attention_tp=0,
up_tail_balance=0, up_pairs=36` (PP2). Every new feature is independently explicit;
balancing is never silently enabled. PP4/PP8 are separate reference configurations;
historical B1 timings do not qualify the newly integrated builds or other batches.

## Fixed options matrix

For each `N` in **2, 4, 8**, the emitter defines these ten independent arms:

| Options file | Mode | `parallel` | `attention_tp` | `up_tail_balance` |
|---|---|---:|---:|---:|
| `ppN.json` | PP reference at world N | 0 | 0 | 0 |
| `ppN-up-balanced.json` | PP, up-tail balance only | 0 | 0 | 1 |
| `ffn-tpN.json` | FFN intermediate TP, serial attention | 1 | 0 | 0 |
| `ffn-tpN-up-balanced.json` | FFN TP plus up-tail balance | 1 | 0 | 1 |
| `epN.json` | Routed-expert EP, serial attention | 2 | 0 | 0 |
| `epN-up-balanced.json` | EP plus up-tail balance | 2 | 0 | 1 |
| `full-tpN.json` | Head TP plus FFN TP | 1 | 1 | 0 |
| `full-tpN-up-balanced.json` | Full TP plus up-tail balance | 1 | 1 | 1 |
| `head-tp-epN.json` | Head TP plus routed-expert EP | 2 | 1 | 0 |
| `head-tp-epN-up-balanced.json` | Head TP + EP + up-tail balance | 2 | 1 | 1 |

Here `N` is replaced literally, e.g. `full-tp8-up-balanced.json`.
All entries keep `phased=1` and raw `up_pairs=36`. The balance switch changes
`effective_up_pairs` to 32; it rejects a simultaneous `up_pairs` override.
`attention_tp=1` requires `parallel=1` or `2`; it is not a PP option.

### Retained CPU-generated options

The lead executed this generation-only command from the repository root:

```sh
python3 -m examples.glm53.parallel_variants --out results/o55-parallel-variants-20261022
```

Reported output:

```text
GLM-VARIANTS options=30 launches=0 status=uncompiled-untested out=results/o55-parallel-variants-20261022
```

The 30 options JSON files and `manifest.json` are retained under
`results/o55-parallel-variants-20261022`. No tests, compiler or GPU launcher ran.
The generator refuses any existing output path: do not rerun this command against
that directory. Any later generation needs a separately authorized new directory.
The manifest records mode, degree, complete options, canonical `Options.digest()`
hashes, and all qualification still required. The option hash is SHA256 of sorted
compact options JSON, not the pretty-printed file bytes. Successful generation and
the lead's reported `git diff --check` passes are not test or build passes.

## Actual parallel boundary and limits

- In TP/EP graphs, norms/mHC, router, embedding and final head/sampling remain at
  root. EP leaves the dense FFN at root; FFN TP also shards dense intermediates.
  PP instead keeps stage ownership, including its final-stage norm/head.
- KDA heads are sharded. MLA **QA/KVA/indexer projections are replicated**;
  **QB/KVB, attention heads and O input columns are head-sharded**. MLA p1
  low-rank/indexer work and IQ/indexer expansions remain replicated. This is not
  a claim that every attention operation or weight scales as `1/N`. Production
  MLA layers use `deepseek_sparse_attention` (`full_attention` is a fixture alias).
- Each attention rank has local weight/meta, dispatch input, projection/attention
  scratch and recurrent/cache state. State uses the **full global-sized stride
  per rank**, but is PRIVATE and accessed only over that rank's head interval.
  MLA replicated indexer data is also rank-local. No divided-cache capacity claim
  follows from head TP. Partial outputs and dispatch sources cross ranks via
  explicit publication; remote-rank dependencies require SYSTEM events, while
  same-rank phase dependencies stay device-local.
- Output partials are reduced in root order. TP changes projection/output FP
  summation; byte-identical weight slices do **not** imply numerical equivalence.
  Head-TP+EP therefore needs TP numerical qualification, not only EP preservation.
- Up-tail balance changes **only up grids**, not head count or any other phase
  cap. For the unsharded production sparse up shape, 36 → 32 pairs changes
  **512 → 576** tiles (up cap remains 608); TP applies that geometry to its local
  intermediate width. No head-cap change is bundled with this option.

**Neither EP alone nor FFN TP alone is projected to deliver 2×.** Full TP8 is a
plausible, **unmeasured** candidate, not a forecast or acceptance result. Root
work, replicated projections, FP32 state capacity, reduction and fabric traffic
can dominate. See [capacity accounting](PARALLEL-CAPACITY-2026-10-22.md) for context;
its estimates
are not measurements of these new arms. In particular, the earlier **9.598 ms
ideal8 estimate is not this implementation's predicted latency**: it divided all
p1+p2+O work despite the replicated MLA work above. Per-phase scaling and
communication still need measurement.

The independent oracle backend labels are `etx-tp` for `parallel=1, attention_tp=1`
and `etx-head-tp-ep` for `parallel=2, attention_tp=1`. Future logs must preserve the
host's `GLM-VARIANT` line (devices/parallel/attention_tp/up_tail_balance/up_pairs)
and the generated identity; the label alone is not a correctness certificate.

## Gates before any performance claim

1. **Local/source gates, later:** execute the authored option, graph, layout and
   emitter regressions; inspect exact generated options/header/source identity.
   Use the native gfx942 compiler and VM-matching device libraries, record
   VGPR/spills/LDS/occupancy and exact rank allocations. A planner register budget
   is not a compiler resource result. None of these checks ran in this preparation.
2. **Layout/capacity:** prove every shard's bytes and metadata against staged
   originals, replicated rows and ownership; count peak per-rank weights, full
   private state, scratch, communication and allocator reserve. Do not use ideal
   divided-cache estimates as fit certificates.
3. **Correctness:** EP-only must preserve selected experts, expert-local arithmetic
   and ordered combination. FFN/head TP must pass independent numerical tests
   covering changed summation, KDA recurrence and MLA/indexer state over decode
   positions. All arms require **32/32 tokens equal to HF greedy**, request
   isolation and publication correctness. For imported launches, preserve the
   repository's same-build byte-identity check requirement. Reject incomplete runs.
4. **Deciding experiment first:** derive bandwidth floor from bytes/achievable
   bandwidth, trace baseline gaps and root/straggler work, then write a numeric
   go/no-go bar before spending GPU time. Measure reduction/publication/traffic
   and actual overlap in a bounded microbenchmark before a full decode campaign.
5. **Timing:** pair host-wall measurements at fixed GPU type, world, batch,
   context/decode positions, compiler and correctness-qualified options. Compare
   each arm with same-world PP, balanced versus unbalanced at the same compute
   mode, same-kernel unfused replay, and the strongest legal real-system reference.
   Keep device span secondary and label its basis. Use interleaved repeats; report
   every pair, median, spread, correctness, root critical path and observed fabric
   traffic/overlap. A new world/GPU session is not an absolute-ms comparison to an
   old one. For a fusion claim retain the repository's five-pair positive-gain,
   median >2% and >2× spread rule; this is not a promise of parallel speedup.

No GPU session or new expenditure is authorized by this runbook. The stopped
GPU box must remain untouched. Future execution needs the owner's separate
session authorization and the repository's preflight/budget/logging gates.
