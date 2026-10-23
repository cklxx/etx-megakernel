# Copied kernel microbench

Kernels copied from `pegainfer-project/kern` (Apache-2.0) so we can time candidate kernels on a real
GPU instead of trusting published numbers. See `kern/PROVENANCE.md` for the exact commit and scope.

## Layout

- `kern/src/*.cu` — 18 device sources, verbatim copies.
- `kern/manifest.json` — the 45 kernel families `kern` indexes, with upstream project and license.
- `kern/build.sh` — compiles every source to a cubin in the CUDA devel container, with a Hopper fallback.
- `kern/LICENSE` — Apache-2.0 text from the source repo.

## Compile result, desktop RTX 4070 Ti SUPER, CUDA 12.4.1

18 of 18 compile. 12 build for `sm_89` and run on the local card. 6 need Hopper ISA
(`griddepcontrol`, `cluster_group`) and build only for `sm_90`, so they compile here and run on H100/H200.

| Kernel | Arch | Cubin |
|---|---|---|
| argmax | sm_89 | 26,784 B |
| copy_rows | sm_89 | 11,880 B |
| dflash_conv | sm_89 | 18,600 B |
| dflash_select | sm_89 | 26,144 B |
| gdn_advance | sm_89 | 10,792 B |
| line_copy | sm_89 | 9,512 B |
| markov_rows | sm_89 | 31,456 B |
| peer_barrier | sm_89 | 7,712 B |
| peer_collective | sm_89 | 26,528 B |
| spec_round | sm_89 | 20,712 B |
| topk_row | sm_89 | 27,424 B |
| weight_prep | sm_89 | 43,064 B |
| attn_prep | sm_90 | 32,872 B |
| gdn_decode | sm_90 | 115,456 B |
| gemma_rms_norm | sm_90 | 52,648 B |
| peer_allreduce | sm_90 | 78,736 B |
| sigmoid_mul | sm_90 | 19,864 B |
| silu_mul | sm_90 | 17,032 B |

## What is missing before a number counts

A compile is not a measurement. Each kernel needs a launch driver that allocates its arguments,
launches it, and times it with CUDA events over a stated iteration count. Until that exists, no
kernel here has a measured time, and no claim about it being optimal is allowed.
