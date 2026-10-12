# Fleet port: expected gain, and the session that measures it (2026-10-12)

## What the scheme can touch

The ported scheme (fleet's F, per-XCD local assembly, applied to ETX's expert folds) can only remove
assembly, publication, wait and consumer-staging work. In the device-basis decomposition that work sits in the
`machinery` and `gap` columns of the fold's consumer stages:

| stage | machinery us | gap us | sum us |
|---|---:|---:|---:|
| qkv (its fold feeds the norm chain) | 179.0 | 183.2 | **362.2** |
| lm_head (final fold) | 7.2 | 1.7 | **8.9** |
| **pool the scheme competes for** | | | **371.1** |

Source: `docs/MASTER-TABLE-2026-10-11.md`, rows `qkv` and `lm_head`. These are **gross containing pools on the
device basis**, not predicted recovery. The device-to-wall transfer is 1:1 [M], so the same order applies to
wall.

## What is honest about the expectation

- The scheme **adds** work: one local assembly, one publication and one local wait per fold. The net is
  **unknown**, and the pool above is an upper bound, not a promise.
- The closest precedent is adverse: fleet's **global** last-writer fold lost about **4 us** (`docs/STATUS.md:67`).
  The local variant is a different scope and has never been measured on either project.
- The ETX side has its own adverse evidence: W's historical start-delay accounting assigns only **0.048 of
  0.414 ms** to ETX; the rest was hardware.

So the honest statement is: **ceiling 371.1 us, expected net unknown, decision by measurement only.** Do not
budget the 371.1 us as a gain.

## The decision rule, fixed before the run

`examples/sessions/fold_local_arms.json` carries it:

- 5 interleaved pinned pairs against `kept`; every delta negative; median **>= 0.030 ms** and **> 2x spread**.
- 32/32 greedy tokens; `host --check` on every imported launch; 97/97 import coverage on the timed build.
- **Reject** if the complete reader-ready interval (assembly + publication + wait + consumer staging) is not
  shorter than the baseline fold path, or resources regress past 128 VGPR / 29,736 B LDS.

## The session

Script: `examples/sessions/fold_local_arms.json` (arms) + `examples/sessions/fold_local.json` (config).
Config fields to fill on the fresh box: `source_commit` (the commit that lands the port) and the two build
paths under `/root/etx/build/foldlocal/`.

Order inside the session:

1. CPU prewarm and the gates in `cpu_gates` (free, before the paid clock).
2. Build `kept` and `LOCAL_FOLD` with the **full baseline flag set** plus `-DETX_DSL_LOCAL_FOLD` for the arm.
   Confirm the define reached the compiler; a `rocm_build_check.sh` run without the full set compiles the fused
   branch out and reports unchanged resources.
3. One environment check and one unstamped baseline, then the five pinned pairs.
4. `examples/vllm_llm/pairs.py` for the wall verdict, `host --check` for the imported launches.

Budget: build and gates are CPU; the five pairs plus the environment check are about **30 GPU minutes =
$1.00**, plus cold staging. Rate $1.99/h.

**Fetch before destroy**: copy the session logdir off the box into the tree *before* deleting the machine. A
destroyed box takes its numbers with it.
