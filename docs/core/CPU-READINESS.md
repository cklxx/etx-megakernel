# G0: reproducible CPU acceptance before a GPU session

Goal: from a clean checkout, one command validates the source-only CPU foundation,
preserves every failure and produces an auditable acceptance directory. No GPU
provisioning or kernel execution occurs. Passing G0 does **not** complete roadmap
stage 1 or establish Qwen/Mixtral numerical parity, hardware visibility correctness,
performance transfer or serving readiness.

## Completion criteria

| Gate | Required result |
|---|---|
| Numerical acceptance | All 24 synthetic MoE cases produce the expected verdict; finite stale outputs with missing writes fail |
| Operator contracts | Qwen and Mixtral imported MoE grids/constants match fingerprint-bound captures; conflicting shapes and historical grid/CuCount errors fail before launch |
| Source regression | All available tests pass; no unexpected skips; known external skips are individually listed |
| Code generation | Split-K, MoE and two-device GEMM/reduce-scatter emit plans, headers, source and decisions |
| Protocol benchmark | All 36 paired simulator samples pass; offline summary recomputes byte-for-byte |
| Local automation | Shell scripts parse; failed child processes, exceptions and report failures return nonzero |
| Formal checks, when dependencies are installed | Lean proves `gemv_correct` without `sorryAx`; three TLA+ models pass and both known-bad configurations produce counterexamples |
| Device compilation, when toolchain is installed | ROCm compiles and links the three generated examples with no GPU devices exposed; executables are not run |
| Production build matrix, when dependencies are installed | All 16 imported-operator configurations across four models compile; every child failure reaches the caller |

The report distinguishes `cpu_core_passed` from `all_external_checks_passed`.
Use `--require-external` to refuse acceptance while **any** external check remains
pending. A new, unrecognized pytest skip fails the core gate. Missing proprietary
artifacts, historical data or toolchains are not converted into fabricated inputs
or successful tests.

## Run

Linux/macOS source-only prerequisites: Python 3.11+, Bash, Git and a C++
preprocessor (`c++`). Run from the repository root:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m etx.tools.cpu_check --out build/cpu-g0
```

Use a fresh output directory for each run. The command writes an initial
`status=incomplete` record before checks, preserves full per-step logs, prints
progress at least every ten seconds for subprocesses, and applies timeouts. Its
final `acceptance.json` contains source provenance, step statuses, test counts,
skip reasons and SHA256 hashes of output artifacts; `README.md` summarizes it.
Numerical fixtures, operator specs, generated sources and the full protocol
benchmark are retained alongside it. Exit status is nonzero on failed acceptance.

To include the optional CPU-only formal and ROCm compile checks (a local
installation/extracted package root can replace Docker via `--rocm-root`):

```bash
LEAN=/path/to/lean TLC_JAR=/path/to/tla2tools.jar \
  .venv/bin/python -m etx.tools.cpu_check --formal \
  --rocm-image rocm/dev-ubuntu-24.04:7.2.4-complete \
  --out build/cpu-g0-extended
```

Add `--local-matrix` to run the full 16-entry imported-operator build matrix
(Qwen2.5-1.5B, Qwen3-8B, Qwen3-30B-A3B and Mixtral-8x7B). Set `ETX_LLVM`,
`ETX_VLLM`, `ETX_ROCM_HEADERS`, and `ETX_DEVLIBS` to the matching compiler, vLLM
checkout, header tree and bitcode directory. The matrix now forces the imported
path, does not pass retired feature macros and propagates every child exit code.
Import compilation clears the final-link device-library setting so ID calls are
not lowered prematurely.

Install those dependencies beforehand. No script installs packages, pulls images,
provisions a GPU or runs a compiled GPU executable. The ROCm step uses an already
installed image, disables networking and records its image ID/digest and compiler
version. The formal script runs Java/Lean directly, saves unfiltered logs and
checks actual counterexamples rather than treating a tool/parse error as success.
For a rootless ROCm installation, the local script selects HIP/LLVM/library paths,
fixes the compilation target to gfx942 and records the compiler hash. The four
actual captured MoE IR modules are imported and then verified by LLVM before
compiling/linking the generated examples.

`DataVisBad.cfg` is applied to `DataVis.tla`; there is no `DataVisBad.tla`.

CI runs the source-only command on pull requests and main pushes and uploads the
acceptance directory even on failure. CI does not silently claim the optional
checks ran.

## What was made independent of external data

`examples.vllm_llm.bench_params` can now parse and derive parameters from its input
logs without opening hidden `results_2026-09-28` files. Historical fitted simulation
is still available by explicitly passing **both** `--plan` and `--measured`.
Without them, JSON contains derived parameters and an empty `sim`; no timing is
invented. The parser test continues using synthetic printf-format input, explicitly
as a parser fixture rather than measured evidence.

`etx.bench.numerical` compares a vectorized float32 MoE block against an independent
scalar float64 calculation: RMSNorm, routing, top-k, optional renormalization,
gate/up/SiLU/down and residual. It uses tiny generated weights and 8/128 experts
with top-2/top-8. These are semantic fixtures, not actual Mixtral/Qwen layers or
bf16 production kernels. NaN, infinity, missing writes, wrong values and conflicting
top-k are injected. Routing ties, invalid references, output shape and tolerance
validation have separate tests. Numerical timings are not reported.

## Single source for the first imported-operator contracts

`etx/importer/specs/moe-decode-v1.json` records the exact supported H/MI/E/TOPK
shapes, compiled BM/BN/BK/top-k/weighting constants, resources and SHA256 fingerprints
of all four committed MoE captures. Fingerprints bind reviewed constants to the
actual TTIR, LLVM IR, metadata and binary; they do not prove kernel semantics.
`launch_spec.py` generates grid and N/K/top-k information plus dtype/layout and
read/write descriptions. Both graph construction and importing verify the same
registry. Unsupported shapes request a recapture instead of guessing from expert
count. A changed capture requires explicit review of the registry.

BM/BN/BK are also checked directly against the captured TTIR dot shapes, so a
conflicting specification fails even if the capture fingerprints are unchanged.

The imported path's grid no longer owns an independent BN constant. The graph
publishes BM through the existing feature manifest, and the host reads it from
that generated definition. A separate check enforces wvSplitK's CuCount/grid
invariant. This is a deliberately bounded first contract, not full unification of
all operator argument ABIs, arbitrary shapes or DSL operators. Existing ABI
packing and most host scalar derivations remain unchanged and need GPU parity
checks before claiming full safe-reuse acceptance.

## Checks requiring additional inputs

The final report lists concrete skipped tests. Typical inputs are:

- ROCm/LLVM and the matching vLLM source/header tree for real operator import,
  linking and the full `local_matrix.sh` / `rocm_build_check.sh` production-model
  matrix. Compiling the three small examples does not replace this matrix.
- The original September 30 measured plans for historical calibration regression.
  Retrieve authentic archived data; do not recreate it from today's compiler and
  present it as the original measurement.
- A real Qwen checkpoint plus PyTorch/Transformers for the existing NumPy-versus-HF
  reference check. Tiny fixtures are additional tests, not a substitute.

GPU-only work remains: imported/original byte parity, full output-write and
finite-value checks on actual device buffers, Qwen/Mixtral 32-token parity,
visibility/fault tests on hardware, matched timing boundaries and repeated paired
measurements with a preregistered precision threshold. No speedup is claimed here.

## Reproducing the extended checks used during development

ROCm 7.2.4 Ubuntu 24.04 packages can be extracted with `dpkg-deb --extract` into
a user-owned directory; no root install or GPU is required. Packages used were
`rocm-llvm`, `rocm-device-libs`, `hip-dev`, `hip-runtime-amd`, `hipcc`, `hsa-rocr`,
`hsa-rocr-dev`, `rocm-core`, `comgr`, `rocprofiler-register`, `hipcub-dev`,
`rocprim-dev` and `rocthrust-dev`. Verify package SHA256 against the versioned
repository's Packages index. A complete container image exceeded this workspace's
32 GB capacity; extracted packages provided the compiler without that image.

The vLLM checkout was v0.27.1 (`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`).
The PyTorch header-only checkout was v2.11.0
(`70d99e998b4955e0049d13a98d77ae1b14db1f45`); its generated `cmake_macros.h`
used the template with all optional C10 switches undefined. A header tree exposes
`inc` → ROCm include, `clang` → LLVM's clang/22/include, and `pytorch` → that
checkout. This is compilation setup, not validation of a deployed vLLM build.

Original September 30 records are available in the `etx-results-2026-10-02.tar.gz`
asset of `cklxx/etx-data` release `data-2026-10-02`. Extract `results_2026-09-30/`,
verify its `SHA256SUMS` file hashes to
`d4031ea630a8df134c57c43f269a2f090ef0396f6167dec82b4d0585ee2b5f71`
(as indexed in DATA.md), and verify every listed member. Do not commit the raw
archive or weights to this repository.

For the real small-checkpoint reference, install CPU PyTorch and the `reference`
extra, set `ETX_REFERENCE_MODEL` to a local Qwen2.5-0.5B directory, and optionally
`ETX_REFERENCE_PYTHON` to its Python interpreter. The test now executes both NumPy
and HF and requires their four greedy tokens to match, not only a stored string.
The CLI exits unsuccessfully on token mismatch. The pinned public checkpoint
used here is `Qwen/Qwen2.5-0.5B` revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`. Package versions and checkpoint-file
hashes are recorded in the acceptance manifest. This does not validate the large
Qwen/Mixtral GPU paths.

## Completed G0 run — 2026-10-03

The strict unified command passed with `cpu_core_passed=true`,
`all_external_checks_passed=true`, no external checks pending, and GPU execution
false. The tested source tree before this evidence-only documentation addition was
`f69f6c1969346f61de35a982ab4697266ddff744`. The run began from a dirty working
tree based on `afa921c` (the same tree as remote protocol PR commit `e9aa974`);
the manifest retains per-source hashes instead of claiming the parent was tested
without changes.

```bash
# Configure the dependencies/environment documented above first.
.venv/bin/python -m etx.tools.cpu_check --formal --local-matrix \
  --require-external --rocm-root "$ETX_ROCM_ROOT" --out build/cpu-g0-final
```

- 190 pytest tests passed, zero failed or skipped.
- All 24 synthetic numerical verdicts matched expectations.
- All 36 paired simulator samples passed; report recomputation was byte-identical.
- Lean 4.34.1 proved the theorem; TLC 1.7.4 accepted three models and found the
  required counterexamples in both known-bad configurations.
- ROCm 7.2.4 / LLVM 22 verified four imported IR modules and compiled/linked
  three HIP examples without executing them.
- All 16 imported-operator matrix builds passed with the same ROCm compiler.
- The real small Qwen checkpoint's NumPy and HF paths matched all four tokens;
  the original archived calibration data passed checksum verification.

Full artifacts remain in the local `build/cpu-g0-final/` directory, outside git.
The following hashes identify the retained evidence; they are not GPU measurements.

| Artifact relative to that directory | SHA256 |
|---|---|
| `acceptance.json` | `5548f3ac8944aad22f3f1f72ebbc79ce4c9d15d004530d3cac8f24cd915436cb` |
| `pytest.xml` | `a556cafd913d29b4645aa871ad50e2841d05987011b6ca78af84eb9e88a10e4a` |
| `numerical.json` | `e6debdde0219dde815284a3a39edd79c45eb6db0dcee9c65796bd6c577b65e0f` |
| `protocol/samples.jsonl` | `a083ba80b895f0065541c73552e9998a2431a79a76b73145d11fee65c89220ff` |
| `protocol/summary.json` | `b9766181c4662116a0614c067f9b68e325a52177c4bf3c995a3fc6ca120a376f` |
| `formal.log` | `0280d3456bc3ba34499f65e876660cd5bfb37c83cd7479031ca1201c63d47cbb` |
| `device-compile.log` | `79f015b7f5d7130ab223aa4364b86e6bd19036724f2ca0f30bacda485b74892f` |
| `local-matrix.log` | `218bb845d3270dadac5acb5c7a5e8bed363b4a8e7ce34fe7baa11aabe750199c` |

## Input integrity gates

The production build matrix clears inherited `ETX_VL_*`, `ETX_LLM_*`,
`ETX_DSL_*`, and `DSL_ABLATE_*` experiment settings before defining each variant.
Toolchain paths remain configurable. Each build saves `configuration.json` with
its model, variant, model config path and explicit overrides; generated plans
and manifests retain derived settings.

wvSplitK supports only a one-dimensional launch grid. Its kernel uses
`blockIdx.x`, so equality between CuCount and the product of arbitrary grid
axes is insufficient to establish output coverage. Grid dimensions, CuCount,
row count, tile size and wave count must use supported integer values.

If the September 30 archive is present, acceptance verifies the pinned
`SHA256SUMS` digest and every listed file before starting tests. Missing, changed,
unsafe, duplicate or unlisted members reject the run. A dependency failure saves
`status=failed` and its reason in the acceptance manifest instead of producing a
successful report from unverified inputs. An absent archive still follows the
explicit external-skip policy; strict acceptance does not permit those skips.
