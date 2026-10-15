# Multi-request continuation status

2026-10-15T16:53Z — COMPLETE ATTRIBUTION / CANDIDATE NOT IMPLEMENTED OR MEASURED.

New serialized attempt20261015T164723Z-d0e0cb:9/9traced runs passed all strict per-request gates and exact task coverage. Runner/job exit0; execution terminal16:48:59Z after96s. Full32-file sealed export verifies exit0,0bad/0omitted. GPU observed idle0% and no project processes at16:50:02Z. No restart, new machine, cloud/cron action or push. Amber owns teardown; deletion not claimed.

Decision: B4/B2 tasks2×, body-work1.982145×, span1.998101×. B4 median trace8.670160ms; median-span run CP bodies7.340840ms +dep0.963120ms +queue0.366200ms. Recorded gaps~1.33ms are smaller than the3.3723632ms registered host reduction needed; prioritize repeated tile-body reduction. No HBM/candidate speedup claim. Baseline ETX/AITER reused: B1 3.222786/4.097652ms,310.290537/244.042198tok/s; B2 4.532310/4.732610ms,441.276082/422.599792tok/s; B4 8.921324/6.165512ms,448.363942/648.770126tok/s.

Executed orchestration edd7457a333fd25dcf90402c91b8af24cb563cfe; frozen executable/source554133d744b4591c9cf87c68b6817df781137c9b, unchanged hashes before/after. Local targeted tests9+3passed; no broad suite/compiler reruns. All raw evidence under /LOCAL_WORKSPACE/.hermes/cache/scratch/etx-multibatch-continuation-20261016-evidence. Full result: /LOCAL_WORKSPACE/.hermes/cache/scratch/etx-multibatch-continuation-result.md.

Remaining prerequisites: reviewed column-batched implementation, arithmetic/LDS and full per-request gates, paired same-box candidate measurements; accepted traffic counters if making HBM claims. Same-kernel graph unavailable/LDS and old x8 failures retained. Original17:22:10Z stop unchanged. Completion event will be signalled after local receipt commit.
