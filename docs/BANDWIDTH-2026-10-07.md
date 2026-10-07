# Decode phase byte budget, 2026-10-07

The round11 trace measures one Qwen3-30B-A3B batch-1 decode step on an MI300X VF.
The plan has 60,362 tasks. The trace has a 5,081.5 us device span.
The traced step has 1,055 KV positions. The 1,024-position ledger describes a different step.

Each line gives phase, ledger GB, device us, derived GB/s, percent of 4.3 TB/s, and excess us.
The list ranks known-byte phases by excess us.

1. `qkv`: 1.0068 GB; 769.8 us; 1,307.9 GB/s; 30.4%; 535.7 us.
2. `moe_up`: 2.4159 GB; 1,051.1 us; 2,298.4 GB/s; 53.5%; 489.3 us.
3. `moe_down`: 1.2080 GB; 748.9 us; 1,612.9 GB/s; 37.5%; 468.0 us.
4. `router`: 0.0254 GB; 438.9 us; 57.8 GB/s; 1.3%; 433.0 us.
5. `attn`: 0.1037 GB; 353.4 us; 293.4 GB/s; 6.8%; 329.3 us.
6. `prep`: 0.0001 GB; 319.2 us; 0.4 GB/s; 0.0%; 319.1 us.
7. `o_proj`: 0.8053 GB; 382.8 us; 2,103.6 GB/s; 48.9%; 195.5 us.
8. `lm_head`: 0.6223 GB; 171.0 us; 3,639.8 GB/s; 84.6%; 26.3 us.
9. `prologue`: 0.0000 GB; 11.3 us; 0.4 GB/s; 0.0%; 11.3 us.
10. `in_norm`: 0.0000 GB; 5.6 us; 0.7 GB/s; 0.0%; 5.6 us.

Known traffic totals 6.1875 GB. The 4.3 TB/s floor is 1,439.0 us.
Known bytes divided by the device span give 1.218 TB/s, or 28.3% of that floor rate.
The floor is an estimate, not a GPU measurement.
At 1,024 positions, the ledger instead gives 6.1845 GB and a 1,438.3 us floor.

## Method

`bw_spec.py` reads the plan through `etx.tools.timeline.load_plan`.
It gets tensor sizes from `byte_ledger.py` and task counts from the plan.
The spec assigns each phase's ledger bytes across its plan tasks.
The trace supplies the first ready time for each phase.
Each phase receives the time until the next phase's first ready time.
The first phase also receives the initial delay. The last phase ends at the trace end.
These intervals total the device span.
The rate equals ledger bytes divided by phase time.
Excess time equals phase time minus ledger bytes divided by 4.3 TB/s.

Run the spec tool with `--config examples/llm/configs/qwen3-30b-a3b.json --context 1055 --output bytes.json`.
Pass the round11 plan as its first argument.
Run `python -m etx.tools.timeline PLAN TRACE --bytes bytes.json --bandwidth-report` for the list.

## Limits

`topk` has 422.0 us of device time, but its bytes are unknown.
`reduce` has 397.4 us of device time, but its bytes are unknown.
`argmax` has 10.1 us of device time, but its bytes are unknown.
The list cannot rank these phases by excess time.
The ledger omits activation traffic, metadata traffic, cache effects, and duplicate weight loads.
It counts one rotary row, although each layer can fetch that row.
The regex format assigns average bytes per task. It cannot show uneven attention partitions or empty router tasks.
The rates are derived from ledger bytes, not hardware memory counters.
The attention KV cache can reside in MALL. Its derived HBM percent does not prove an HBM fault.
Each device interval includes dispatch tails and phase handoffs, not only memory transfer.
These limits prevent a direct claim that each excess microsecond is recoverable bandwidth time.
