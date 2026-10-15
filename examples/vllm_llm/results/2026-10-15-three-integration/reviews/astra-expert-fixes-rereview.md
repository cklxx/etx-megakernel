# Expert fixes re-review — NUMERICS/PROTOCOL

Date: 2026-10-15.

**Verdict: both prior defects are CLOSED. No blocking regression was found in the scoped repair diff.**

Reviewed repository: `/Users/ckl/code/.wt/etx-expert-early-merge`.
Reviewed range: `172a22f3809f0ef9a31ebd369fe48fb175dcddd0..11743ed0c18ce4a2acc4af5493c678af1a62691a`.
The checkout matches the final commit and is clean. This review covers the two findings and the four-file repair diff only.

1. **P1 — control diagnostic LDS race: CLOSED.** In `examples/vllm_llm/expert_merge/source.py:48–57`, the new `__syncthreads()` joins all block threads after the vector fold and before the scalar-indexed diagnostic copy. It is outside the `store` branch. Thus the thread-64 write to `xs[512]` precedes the thread-512 read. Both qkv and head use this generator. The retained generated sources independently show the join at line 1247 and the copy at line 1251 in `probe/{qkv,head}-control-diag/tiles.hip`; their manifest/source hashes match the repaired sources. The later reduction joins protect the copy from subsequent normalization writes. The added join is inside `EEM_DIAGNOSTIC`, so timed configurations gain no barrier. This closes the identified ordering defect; device byte equality remains an execution gate.

2. **P2 — staged archive capture attestation requires Git: CLOSED.** In `examples/vllm_llm/expert_merge/capture.py:48–83`, `source_identity()` uses Git only when the source root has `.git`. Otherwise it requires the complete four-field `SOURCE_IDENTITY` receipt emitted by unchanged `tools/session.sh:23–31`. It validates commit/archive hex lengths and the UTC date, permits an empty detached-checkout branch, and rejects missing, duplicate, incomplete or malformed records. It cannot accidentally use a parent repository's HEAD. Attestation retains the source, binary, plan and command fields and adds provenance plus the receipt hash. The final-commit archive CLI receipt records revision `11743ed0c18ce4a2acc4af5493c678af1a62691a`; missing and malformed identity attempts fail. The acceptance inputs are explicitly non-executable fixtures, so this proves attestation transport, not a model build or capture.

Evidence was read from `/Users/ckl/.hermes/cache/scratch/codex-expert-final-fixes-result.md` and `/Users/ckl/.hermes/cache/scratch/expert-final-fixes-evidence/`. Retained logs and true-exit receipts show:

- Regression sensitivity: three failures on the old code (exit 1), then three passes after the repairs (exit 0).
- Scoped experiment/session tests: 46 passed in 27.15 seconds (exit 0).
- Archive CLI: success exits 0; missing and malformed identity cases each exit 1.
- Both changed control diagnostic configurations have successful local/offline compiler and HSA device-link receipts. No native HIP executable was linked, loaded or run.

These are the implementer's retained runs, independently inspected here; this re-review ran no tests or builds. `build-reuse.json` records six unchanged preprocessed device units after build-path/identity normalization. `host-reuse.json` records unchanged plan headers and host configurations apart from identity. Original full-suite, matrix, arithmetic and protocol results remain historical evidence.

**Repair regression assessment:** no blocker found. Candidate arithmetic, publication/reset protocol, strict correctness gates, timing thresholds and full-model default are unchanged. The protected session, host, input validator and runner files match their baseline hashes. The documentation distinguishes repaired preparation from device results. The new tests cover both diagnostic roles and archive success/refusal paths without weakening existing checks.

These two pre-GPU integration blockers can be removed. Native executable linking, GPU load/residency, repeated payload equality and cross-XCD visibility, real-input capture, model correctness/import checks and paired performance gates remain open. This report does not establish hardware correctness or authorize a GPU session.

The repositories remained read-only. Only this new report was written. Prior reports were preserved; the prior independent review SHA-256 remains `38e8b286460c956a74c7a9b8cf3f58e1eb133b8a567fd4e437e6465a087a4ed8`.

结论：两项缺陷均已关闭；限定修复范围内未发现阻塞性回归。GPU 正确性与性能门槛仍待验证。
