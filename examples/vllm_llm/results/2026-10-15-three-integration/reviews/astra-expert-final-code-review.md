# Expert row merge: final-code review

**Decision: FIX BEFORE READY.** Two concrete defects must be fixed before this probe is marked ready for common pre-GPU integration. One can corrupt the mandatory diagnostic reference. The other blocks real-input capture on the prescribed archive deployment. Neither is a GPU-only unknown.

Reviewed commit: `172a22f3809f0ef9a31ebd369fe48fb175dcddd0`.
Base: `c432c77b7f778799ecbb89db9af89902d5923c00`.
Repository: `/Users/ckl/code/.wt/etx-expert-early-merge`.
All repository file references below are relative to this root and refer to the reviewed commit. The worktree was clean. I compared all 16 added files with their pinned Git objects; all matched. This is an implementation review of both numerics/protocol and measurement/readiness. It replaces neither the prior design report nor the pending hardware gates.

## 1. P1 — The diagnostic control reads LDS before its writers join

**Source:** `examples/vllm_llm/expert_merge/source.py:45–57`, especially lines 51–52. The inserted capture runs immediately after the original vector fold. That fold assigns eight elements to each active thread. The diagnostic copy assigns elements by `threadIdx.x` with stride 1024. It therefore reads values written by other threads and waves. There is no block join between these writes and reads.

**Concrete failing order:** In consumer block zero, thread 64 must write `xs[512]` as part of its eight-element fold. Thread 512 has no fold work, but copies `xs[512]` to the merged diagnostic output. Wave 8 can reach this copy before wave 1 completes its contribution loads and LDS writes. The copy then records old LDS contents. The later `etx_block_sum` barriers cannot repair that captured value.

The retained generated source confirms the issue:
`/Users/ckl/.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/probe/qkv-control-diag/tiles.hip:1240–1248` has the writes and copy without an intervening join. The original reduction follows at line 1277. I verified this file against its manifest hash. The same source generator serves the head diagnostic build.

**Decision affected:** `runner.py:162–165` uses `control_diag_fused` as the exact reference for every arm. A valid candidate can therefore fail the required merged-value check, or the check can become schedule-dependent. This is a defect in the experiment's correctness reference, not evidence against row merge. The CPU arithmetic test does not exercise this path: `cpu.py:112` extracts the non-diagnostic candidate and its barrier stub is empty at line 46.

**Minimum acceptance fix:** Before the diagnostic copy, add a block-wide join reached by all required waves, or make each writer copy only its own eight completed values. Keep the fix inside the diagnostic build. Inspect both generated control diagnostic variants and add a check that captures the producer/reader ordering requirement. Retain the existing repeated-payload GPU byte gate. Do not weaken that gate to accept this race.

## 2. P2 — Capture attestation requires Git metadata absent from staged archives

**Source:** `examples/vllm_llm/expert_merge/capture.py:55–58`, specifically the unconditional `git rev-parse HEAD` at line 57. The generated private build script invokes this attestation after compilation (`capture.py:38–39`).

**Failing condition:** Use the documented transport on a fresh machine. `tools/session.sh:23–31` creates a Git archive, extracts it into a new directory, and writes `SOURCE_IDENTITY`. The extracted tree has no `.git`. Attestation then exits with Git status 128 and does not write `capture-build.json`. `capture run` requires that file at `capture.py:66`, so the real-input path cannot proceed after the build.

**Independent reproduction:** I executed only the pinned `attest` function in a temporary scratch directory. It contained a `SOURCE_IDENTITY` record, matching tiny source hashes, a compiler-command receipt, and explicitly labelled non-executable binary/plan stand-ins. No compiler or GPU was called. All preceding checks passed. The function then failed with `git rev-parse HEAD`, exit 128, and produced no `capture-build.json`. The temporary directory was removed.

**Decision affected:** The claimed real-input capture path is not runnable through the prescribed archive staging flow. The synthetic isolated runner does not need this attestation and is not blocked by this second defect alone.

**Minimum acceptance fix:** Obtain revision identity from the staged `SOURCE_IDENTITY` contract when Git metadata is absent. Retain the source, binary and plan hashes. Reject a missing or malformed identity instead of inventing a revision. Add a scratch-only successful attestation test with no `.git`, plus a missing-identity rejection case. Regenerate capture receipts after the fix.

## Checked coverage and limits

I found no additional blocking defect in this bounded pass:

- The candidate calls the original up/down bodies. Its extracted row fold preserves slot order, both bf16 boundaries and eight original thread statistics per row group. The final owner loads 256 partials and runs the original block sum. Qkv writes the canonical residual; the final head has a null residual-output pointer. Evidence: `source.py:72–111,126–149`; `host.hip:82–89`.
- All producer waves drain and join before each custom arrival. Election is broadcast in LDS. Non-owners do not spin. The original full down event follows appended work. The 33 custom counters are separate from generated events and reset on the same stream before each replay. Evidence: `source.py:87–110`; `graph.py:13–35`; `host.hip:93–115`.
- Capture selects the eight expert matrices in slot order and remaps IDs to 0..7. It obtains consumer norm and matrix pointers from the next qkv or final-head argument block. The documented packed-address/cache-context limit remains valid; it is not a new finding. Evidence: `capture.h:14–56`.
- Build and input hashes, byte-certificate checks, failed-recheck invalidation, alternating pairs, the five-pair noise rule, and reset-inclusive event windows are connected. The runner retains attempt directories and a fixed session deadline. Evidence: `prepare.py:44–89,108–117`; `runner.py:76–109,147–239`; `host.hip:134–160`. I did not audit these as a security boundary against deliberate receipt editing.

No GPU result was obtained. Device visibility, native load/residency, real-input equality and performance remain pending. The isolated screen does not need a production-scale implementation first. A synthetic pass still cannot establish full-model acceptance.

Repositories remained read-only. No tests, builds, network, GPU, merge or push ran. Only tiny source/manifest checks and the isolated attestation reproduction ran. Former reports were preserved. GPU cost: $0.

中文结论：**先修两处，再标记可进入统一上机准备。** 控制组诊断读取 LDS 前缺少同步，会污染字节检查基准；真实输入采集的构建凭证依赖 `.git`，在规定的归档部署目录必然失败。其余已检查路径未发现新增阻断项；这不代表 GPU 正确性已通过。
