# Batch 2/4/8, 2026-10-14 — box session 20261014T134920Z-324307 (first run of section 10's recipe)

Same condition as the session that follows it (bf16, context 1024, 32 greedy tokens per request, five interleaved
rounds, vLLM AITER at the same B). This run is kept because it is the first measurement of the batch study: two arms
passed their gates and one arm was lost to bookkeeping, not to a result.

     BREPORT b=1 vs vLLM aiter: throughput WIN median +32.58% spread 1.23% (274 vs 206 tok/s)
     BREPORT b=2 NOT COMPLETE
     BREPORT b=4 vs vLLM aiter: throughput LOSS median -24.50% spread 5.31% (410 vs 545 tok/s)
     BREPORT b=8 NOT COMPLETE

    BGATE arm=x4 b=4 PASS
    BGATE arm=x8 b=8 NOT COMPLETE: request 5: 3 layers within the gate (need 27)

What is different from session `20261014T142757Z-03ff3e` (its successor, and the record used for the report):

- `build_x2` failed its recorded scratch allowance check (`arm x2 spills to scratch on this box beyond its allowance
  (0 B/lane): {... 'scratch': 32 ...}`), so `bgate_x2` was skipped and B = 2 had no gate. Section 10's amendment had
  not yet re-measured the allowance for the new recipe; the measurement (x2 32, x4 64, x8 144 B/lane) is now recorded
  in `examples/dsv2lite/arms.py` and the allowance is 64.
- `brounds.json`: `order [["etx",1],["etx",4],["vllm","aiter"]]`, `skipped ["x2","x8"]`; round 1 step times B = 1
  3.651733 ms (worst request 3.76904), B = 4 9.723749 ms (worst request 10.00316).
- vLLM reference steps: B = 1 4.833368, B = 2 5.697327, B = 4 7.032626, B = 8 9.356861 ms.
- Qwen's five jobs were skipped: `qwen_preflight` failed then on the HIP headers, before the tree `.venv` and the
  `libhipcub-dev` gaps were found.

Seal: `VERIFIED: session 20261014T134920Z-324307, 106 files match, 0 bad, 60 left on the box by design`.
