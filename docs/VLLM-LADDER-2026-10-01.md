# What vLLM's own fusion is worth, and how ETX's megakernel compares

VF box (1x MI300X VF, ROCm 7.14), Qwen3-30B-A3B bf16, batch 1, greedy, context 1024, the protocol every vLLM
number in this session used (`vllm/vllm-openai-rocm:v0.27.1`, vLLM 0.27.1, `examples/llm/vllm_bench.py`).
Three runs of each configuration, interleaved one-of-each per round so drift cannot favour a row. Every run
matched 32/32 tokens against the golden.

All day we compared our best against vLLM's best without asking how much of vLLM's number is its own fusion
machinery. This is that ladder.

## The ladder

| configuration | runs (ms/token) | median | spread |
|---|---|---|---|
| vLLM default | 4.874, 4.862, 4.854 | **4.862** | 0.020 |
| vLLM `--custom-ops all` (its own HIP ops, no Inductor fusion) | 4.884, 4.936, 4.889 | **4.889** | 0.052 |
| vLLM `cudagraph_mode=NONE` (compiled and fused, no graph replay) | 33.084, 32.710, 32.124 | **32.710** | 0.960 |
| vLLM `enforce_eager` (no compile, no graphs) | 34.049, 35.265, 35.134 | **35.134** | 1.216 |

**vLLM's graph is worth 27.8 ms, a 6.7x speedup** (32.710 -> 4.862). **Its op fusion is worth 0.027 ms, 0.6%**
(4.889 -> 4.862). The two differ by three orders of magnitude.

`cudagraph_mode=NONE` keeps torch.compile and the fused kernels and only drops the graph replay, so its 6.7x
cannot be a device-side effect: it is host dispatch. Everything vLLM's graph buys at batch 1 is the removal of
per-op host overhead. Compilation without graphs accounts for only 2.4 ms of that (35.134 -> 32.710).

## Putting ETX on the same scale, and a timing basis that was not like-for-like

`vllm_bench.py` times wall clock around `llm.generate` and differences two lengths, so every vLLM figure here is
**host** time. ETX's runs report both: the device span (`hipEvent` around the kernel) and the host wall per step.
Today's comparisons used ETX's *device* median against vLLM's *host* figure. Both ETX numbers, same five runs:

| | device median | wall median | host overhead |
|---|---|---|---|
| ETX fused (megakernel) | 4.907 | **5.104** | 0.197 |
| ETX unfused (HIP graph of the same tiles) | 5.107 | **5.143** | 0.036 |

The megakernel carries 0.197 ms of per-step host work that the device timer does not see — argument blocks and
event state refilled every step — while the unfused path replays a graph instantiated once. On the basis vLLM is
measured on:

| step | device basis | wall basis |
|---|---|---|
| ETX megakernel over a HIP graph of the same tiles | 0.200 ms (3.9%) | **0.039 ms (0.8%)** |
| ETX best vs vLLM default | 0.045 behind (0.9%) | **0.242 behind (5.0%)** |
| vLLM's op fusion | — | 0.027 ms (0.6%) |
| vLLM's graph | — | 27.8 ms (6.7x) |

## What this says

1. **Our fusion step and vLLM's op-fusion step are the same order of magnitude** — 0.039 ms against 0.027 ms on
   the wall basis, both under 1%. Measured on the device, ours is 0.200 ms and clearly larger; the difference
   between those two readings is our own host-side per-step cost, which is real and currently eats 80% of the
   device-side win.
2. **Neither is close to what a graph buys.** Both systems' numbers at batch 1 are dominated by having removed
   host dispatch: vLLM through CUDA graphs, ETX through one persistent launch. Once that is gone, the remaining
   differences are small and device-bound.
3. **The honest gap to vLLM is 5.0%, not 0.9%**, until the megakernel's 0.197 ms of per-step host work is
   removed. That work is a concrete, addressable target and it is larger than anything left in
   `docs/BUDGET-2026-10-01.md`'s ETX-owned column — the next thing to look at is what `run()` does per step
   before launching (argument refill, event reset, residency checks) and how much of it can be hoisted or
   captured once.
4. vLLM's Inductor fusion being worth only 0.6% is consistent with what we import: the kernels ETX takes from
   vLLM are already the hand-written HIP ones, and `--custom-ops all` barely moves the number.

Logs: `results_2026-10-01/vllm_ladder/` (12 runs, logs and JSON). The `--eager`, `--cudagraph-mode` and
`--compile-mode` knobs were added to `examples/llm/vllm_bench.py` for this (it hardcoded `enforce_eager=False`).
