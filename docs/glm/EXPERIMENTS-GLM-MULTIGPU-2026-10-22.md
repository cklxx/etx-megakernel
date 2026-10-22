# GLM multi-GPU experiment list, 2026-10-22

Owner question: how do we measure an ETX multi-GPU performance baseline, and what is
the parallel strategy for GLM.

Box: 8x MI325X, gfx942, 2 TB disk, $30.4 per hour.
Status column: EXISTS means the code path is present and runnable today.
MISSING means the code has to be written before the row can run.

## 1. Parallel strategies to test

| id | strategy | cards | weights per card | sync per step | path |
|---|---|---|---|---|---|
| P1 | PP2, whole layers, experts replicated | 2 | 153 GiB | 1 handoff, hidden 4096 bf16 = 32 KB per request | EXISTS, `model_host.hip:19` requires exactly 2 |
| P2 | PP4 | 4 | 77 GiB | 3 handoffs, 32 KB per request each | MISSING, `run_all.py:196` rejects any card count but 2 |
| P3 | TP2, hidden split | 2 | 153 GiB | 2 all-reduces per layer, 45 layers = 90 | MISSING |
| P4 | TP4 | 4 | 77 GiB | 2 all-reduces per layer, 45 layers = 90 | MISSING |
| P5 | EP8, experts split | 8 | 38 GiB | all-to-all per MoE layer, 2 per layer | MISSING |

## 2. What each row measures

One number decides between them: microseconds added to one decode step per
synchronisation, against the milliseconds of weight streaming the split saves.

Byte account, computed: 305.71 GiB FP8 total, `[U]` until staged.

Handoff cost: PP2 sends 32 KB per request per step. At the measured streaming rate
of 4.24 TB/s a 32 KB copy costs 7.5 ns, so PP2 pays almost nothing in bytes and
everything in the serial dependency: rank 1 cannot start a layer until rank 0 ends
the previous one.

TP pays in collectives: 90 all-reduces per step, 8 KB each at hidden 4096 bf16.
90 x (measured all-reduce latency) is the number to watch.

## 3. Experiments

Measured today on this box, before the table: barrier inside a persistent kernel,
304 workgroups, one per CU, MI325X gfx942.

| variant | per barrier |
|---|---|
| clean lines, relay off, uncached | 5.46 us |
| clean lines, relay on, uncached | 5.60 us |
| 64 KB dirty + 128 KB clean L2 load | 15.62 to 18.12 us |
| relay on, no load, stale count 2 427 104 | 3.25 us, INVALID, staleness not assumed away |

The repo documents 8.6 us for this barrier on MI300X. This box measures 5.46 us
clean, so a synchronisation is 1.6x cheaper than the budget assumed, and a barrier
with a dirty L2 costs 2.9x more than a clean one.

| id | experiment | command | acceptance number | GPU minutes | status |
|---|---|---|---|---|---|
| E0 | fabric baseline: one-way flag latency between two cards, idle and under load | `/root/runs/calib/p2p_sync` | one-way under 2 us idle, under 4 us loaded | 3 | built today |
| E1 | barrier cost inside a persistent kernel, 304 workgroups | `/root/runs/calib/sync_bench barrier` | median under 10 us, matches the 8.6 us in `docs/core/RESEARCH-sync-hardware.md` | 3 | built today |
| E2 | PP2 step time, batch 1 to 8 | `python -m examples.glm53.bench --stage ... --build ... --reference ... --seconds 300` | median step time per batch, gate state recorded | 6 | running |
| E3 | PP2 step time, full 45 layers | needs the full gate implemented in `prepare.py`, `stage.py`, `bench.py` | step time and tok/s at B1 and B8 | 25 | MISSING |
| E4 | PP4 handoff cost | needs the 4-card path in `model_host.hip` and `run_all.py` | added latency per extra handoff | 20 | MISSING |
| E5 | TP2 all-reduce cost | needs a TP path | 90 all-reduces under 0.5 ms total | 30 | MISSING |
| E6 | EP8 dispatch cost | needs an all-to-all path | all-to-all under 0.2 ms per layer | 40 | MISSING |

## 4. What the fabric numbers decide

Sync cost is not the blocker on this box.

TP4 needs 90 all-reduces per decode step, two per layer over 45 layers.
One all-reduce costs about two barriers, so 11 us.
90 x 11 us = 0.99 ms added to every step.

PP2 sends 32 KB per request per step. At 4.24 TB/s that copy is 7.5 ns.
PP2 costs one serial dependency, not measurable bytes.
PP4 has three dependencies and the same 32 KB per request.

So the choice order is PP2, then PP4, then TP, then EP.
TP is only justified when the weights no longer fit, and then the 0.99 ms is the
price to pay per step.

The measured bottleneck is elsewhere: the per-layer single-workgroup weight
streaming dominates, not the fabric.

## 5. How the baseline is defined

A baseline row is valid only when all four hold:
the gate states 32 of 32 tokens equal and the cosine at or above threshold;
the run uses a committed golden, 1024 prompt tokens and 32 greedy tokens;
host wall stops before any device-to-host copy, and device time is a separate column;
the run is serial on one card pair, never parallel across pairs.

A row that fails the gate is reported as FAILED with its time marked UNVERIFIED.
It is never quoted as a baseline.

## 5. Order for the next hour

1. E0, E1: fabric numbers, 6 minutes total, already built.
2. E2: PP2 step time for batch 1, 2, 4, 8, 6 minutes per batch.
3. Report: the fabric numbers, the per-batch step times with gate state, and the
   written cost of E3 to E6 with their code gaps.

## 6. Known gaps that block a full-model baseline

`prepare.py` knows the `production` qualification gate only; a full model run is
still the missing gate. `run_all.py` passes no full-model gate in its steps.
`run_all.py:283` hardcodes `--gpu-gib 192`; MI325X needs 256.
`model_host.hip:19` exits unless the card count is 2.

## 6. Full-model batch 1, measured 2026-10-22

Per-task trace from ETX_TRACE=1 on the 45-layer build, batch 1, two cards.

| item | value |
|---|---|
| total step span | 51 111 760 us |
| embedding task body | 66.6 us |
| layer task body, first | 914 768 us |
| layer task body, typical | 900 000 to 1 190 000 us |
| layers | 45, one task per layer, tasks=48 |
| wait between layers | under 1 us; tasks run back to back |

The layers are serial, one task each, and only one rank runs at a time. Rank 1's
first task waits 26 345 766 us of the 51 111 740 us span, so that card is idle for
51 percent of the step.

Bytes: the layer task streams 7.5 GB, the full dense and expert set of the layer,
at 6.8 GB/s. A batch-1 token needs only 9 of 288 experts, about 0.53 GB per layer
and 25 GB per token, so 14 times the necessary bytes are streamed.

So the 51 s has two independent factors, both measured: 14 times the bytes because
every expert is read, and 1/600 of the bandwidth because one 256-thread workgroup
owns a layer.
