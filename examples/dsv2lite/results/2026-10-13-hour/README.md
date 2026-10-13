# DeepSeek-V2-Lite paid hour, 2026-10-13 (`examples/sessions/ds_hour.py`)

**Box.** One MI300X VF on DigitalOcean: droplet 608683342 at 165.245.136.206, HIP 7.14.60850. Billing ran from 11:44 UTC, and the droplet was destroyed at 12:33:55 UTC (`DELETE` returned 204; `GET /v2/droplets` then reported 0).

**Code.** The session ran from the frozen worktree at commit **2ef1e58**. The staged tree was clean ("dirty 0", tree 84b7f33). It did not run from main.

**Runner.** The runner started at 11:50:00 UTC and printed "session done" at 12:00:53, after 10 m 53 s. All 30 jobs ended ok, with no hard stop. Preflight confirmed that all five inputs match the pinned, previously measured inputs. The GPU jobs ran in the planned order.

**Verification.**
`python -m examples.dsv2lite.verify_hour <dir>` exits 0 (HOUR VERIFIED): every promised verdict and stage table is present and is re-derived from the raw logs.
`shasum -a 256 -c SHA256SUMS` on the full fetch gives 210 OK and 0 mismatches. The other 152 listed files were not fetched by design (see "Known gaps").

## Verdicts (device time, five interleaved rounds; each rule was committed before the run)

| Arm | Rule | Verdict | Median gain (spread) |
|---|---|---|---|
| alld4r0l1 | KEEP vs alld4r0 (8d69e52) | **KEEP** | +334 µs (19) vs alld4r0; +498 µs (16) vs default |
| l1 (D2b) | D2 rule: (a) at least 70 µs faster than default; (b) within max(2× relaxed spread, 20 µs) of relaxed | **NO-GO** | (a) +330 µs (13) vs default, passes; (b) l1 is 99 µs (12) faster than relaxed, outside the ±26 µs tolerance |
| alld3 (D3) | GO / DROP (1f6e3cc) | **NO-GO** | −741 µs (19) vs all |
| alld5 (D5) | Drop if neutral (84eaa1b) | **DROP** | +2 µs (9) vs all |
| alld4s | KEEP vs alld4r0 (ff15c2c) | **NO-GO** | −13 µs (12) vs alld4r0; +159 µs (15) vs default |

**alld4r0l1 absolute timing.**
- The five-round median per token is **3.238 ms device**, which is 309 tok/s. The rounds measured 3.236, 3.240, 3.237, 3.247 and 3.238 ms.
- The wall-clock median is **3.272 ms**, which is 306 tok/s.
- It matched 32/32 tokens and 27/27 layers.

**l1.** The pre-registered D2 rule gives NO-GO. Part (b) was written to show that l1 costs about the same as relaxed. It did not anticipate l1 being faster than relaxed, and l1 was faster by 99 µs. That measurement answers the owner's question about coherence cost. The rule's verdict is recorded as written and has not been reinterpreted.

**Stage spans** (stamped token): default 3997.8 µs, alld4r0 3770.0 µs, alld4r0l1 3491.3 µs.

## Known gaps

- **Not fetched by design (152 files, about 32 MB).** These are the build binaries `b_*/run` and the patched fleet sources `b_*/fleet_src/`. They are lost with the box. All of them can be rebuilt from 2ef1e58 plus fleet-3fb67a6. Their box-side hashes are still listed in `SHA256SUMS`.
- **Uncaptured golden cache.** The generic fetch step that copies the fleet golden cache reported `UNCAPTURED fleet-mi300x/build/golden*` and `fleet_cache.bin`. That build directory did not exist on this box.
  - The inputs this session used live in `~/dsv2-data`.
  - Preflight checked those inputs against `PINNED_INPUTS` before any measurement ran.
  - No result depends on the missing cache, but the cache itself was not kept.
- **Stamp files are stored gzipped.**
  - The three files `trace_*.stamps.bin` are 117 MB each, over GitHub's 100 MB file limit, so they are stored as `trace_*.stamps.bin.gz` (made with `gzip -9 -n`).
  - `gunzip -k trace_*.stamps.bin.gz` restores the exact bytes; each one was checked against its `SHA256SUMS` hash before the raw copy was removed.
  - Run that `gunzip` before running `shasum -c` or `verify_hour`.
