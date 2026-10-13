"""The acquire mode DeepSeek sessions build with: the safe sc1 domain acquire, unless an arm asks for relaxed by name.

ACQUIRE (empty: the build default, `buffer_inv sc1`) is what every session arm passes. RELAXED (the old sc0 acquire,
`-DETX_ACQUIRE_DOMAIN_RELAXED`) is correct only if no consumer re-reads a protected line still in its L1
(etx/machine/arch/gfx942.yaml, `acquire_relaxed`): a session may build it only as an explicitly named comparison arm.
The 2026-09-27 DeepSeek sessions passed RELAXED on every arm, including the 3.474 ms stride-32 record; their logs
keep that meaning, the session files now default to safe (docs/DSV2-D1-D5.md, D1).
"""
ACQUIRE = ""
RELAXED = "-DETX_ACQUIRE_DOMAIN_RELAXED"
