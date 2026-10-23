# Source record: kernels copied from pegainfer-project/kern

- Repo: https://github.com/pegainfer-project/kern
- Commit: `ff88b4f24e3a455491af02bedb1639bde6a56b88` (2026-10-20)
- License of the copied files: Apache-2.0, see `LICENSE` in this directory.
- What we took: the standalone device sources under `tools/kernels-src/*.cu`, copied verbatim, plus `kernel_index/*.toml` summarised into `manifest.json`.
- What we did NOT take: the Rust crates, the cubins, the `tools/dsv41` tree, and any third-party kernel bodies. The upstream kernels named in `manifest.json` (FlashKDA, FlashInfer, DeepGEMM and others) are separate projects with their own licenses; they are listed as candidates only.

## No coupling

This directory is a copy, not a dependency. No submodule, no package reference, no import from
`kern`, and `kern` is not built by our CI. If the copy is ever updated, it is updated by hand
against a named commit.

## Why these kernels

They cover the decode path of our models:

1. `gemma_rms_norm`, `silu_mul`, `sigmoid_mul`, `weight_prep` — elementwise and norm.
2. `argmax`, `topk_row`, `markov_rows` — sampling and candidate selection, needed by speculation.
3. `gdn_decode`, `gdn_advance`, `dflash_select`, `dflash_conv` — linear-attention decode steps.
4. `attn_prep`, `spec_round` — attention preparation and a speculation round.
5. `copy_rows`, `line_copy` — layout movement.
6. `peer_barrier`, `peer_collective`, `peer_allreduce` — multi-GPU synchronisation.

## Status

Compiled, not yet timed. Each kernel needs a launch driver with CUDA events before any number is a
measurement. `build.sh` compiles every source to a cubin for the local architecture.
