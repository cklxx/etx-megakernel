# Tool ownership

Top-level Python modules retain the historical imports and `python -m` commands.
They only re-export implementations; new code should import the owner directly.

| Concern | Owner |
| --- | --- |
| Assembly parsing, hop state, instruction execution, CFG analysis | `isa` |
| Acquire schema, LLVM checks, source checks, orchestration | `contracts` |
| Shared-memory protocol checking | `memory` |
| Bandwidth floors and handoff cost models | `costs` |
| Trace diagnostics and layer budgets | `tracing` |
| Gap, spill and phase diagnostics | `diagnostics` |
| CPU acceptance and protocol simulation | `validation` |
| Compiler command line | `commands` |
| Session jobs | `etx.session.gpu_jobs` |
| Trace data and schedule inspection | `etx.runtime.trace` |

Each concern package is limited to 1,500 Python lines, including its exports.
`tests/test_tools_compat.py` checks the budget and legacy callable identities.
Tests that inject faults patch the implementation where the function's globals
live, rather than the compatibility re-export.
