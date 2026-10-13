"""Lever l1_loads (D2b): fleet's tiles read every payload another task wrote with AGENT-scope loads (`global_load ...
sc1`), so the megakernel drops the per-task acquire invalidate WITHOUT D2's per-task L2 write-back.

Measured basis (bench/sync/sync_bench.hip loadscope, 2026-10-13, `$ETX_CACHE/results_2026-10-13-d1d5/run5`): with no
invalidate, sc1 payload loads read 0 stale words on one XCD (xl1: they skip the consumer CU's stale L1 and read the
shared L2) and across XCDs behind a buffer_wbl2 release (cl1); plain and sc0 loads read stale (xn, xl0). So:
  same-XCD (DOMAIN) hand-off   producer drains its stores into the L2 (plain release), consumer sc1-loads them;
  cross-XCD (DEVICE) hand-off  the last-arriver flush writes the L2 back (unchanged), consumer sc1-loads them.
graph.py marks the tiles coherence="coherent_l1" under ETX_DSV2_COH=l1 (coh_check.py refuses one without the other).

The payload accessors are D2's (levers/coherent_loads.py EDITS, same sites, same audit) with the helpers at agent
scope and named ld_l1*; the in-body q_absorb hand-off needs no write-back here (FLEET_L1_LOADS, not FLEET_COH_LOADS).
When attn_prologue_live is applied first (the `all` arms), its kv_post_live / q_absorb_gather_live read the same
payloads and are patched too.
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.levers import attn_prologue_live as apl
from examples.dsv2lite.levers import coherent_loads as cl
from examples.dsv2lite.patch_fleet import _once

NAME = "l1_loads"


def _l1(text: str) -> str:
    return (text.replace("__HIP_MEMORY_SCOPE_SYSTEM", "__HIP_MEMORY_SCOPE_AGENT")
                .replace("#define FLEET_COH_LOADS 1", "#define FLEET_L1_LOADS 1")
                .replace("FLEET_COH_G", "FLEET_L1_G")
                .replace("ld_sys", "ld_l1")
                .replace("etx lever coherent_loads (D2): system-scope relaxed atomic loads, `global_load ... sc0 sc1` "
                         "(address_space(1), so not\n// flat_), past this XCD's L1 and L2. For payloads written by "
                         "another task: the producer's release wrote its L2 back.",
                         "etx lever l1_loads (D2b): agent-scope relaxed atomic loads, `global_load ... sc1` "
                         "(address_space(1), so not flat_):\n// past this CU's L1, from the XCD's L2 (same-XCD "
                         "producers drained into it; cross-XCD ones were flushed)")
                .replace("etx lever coherent_loads", "etx lever l1_loads"))


EDITS = [(rel, old, _l1(new)) for rel, old, new in cl.EDITS]

# attn_prologue_live's own payload reads (present only when that lever was applied before this one)
LIVE = [
    ("    const float raw0 = kv_a_raw[t], raw1 = kv_a_raw[t + kBlockThreads];\n",
     "    const float raw0 = ld_l1(kv_a_raw + t), raw1 = ld_l1(kv_a_raw + t + kBlockThreads);   // etx lever l1_loads\n"),
    ("    const float rope_raw = kv_a_raw[kv_lora + min(t, qk_rope - 1)];\n",
     "    const float rope_raw = ld_l1(kv_a_raw + kv_lora + min(t, qk_rope - 1));\n"),
]
GATHER_LIVE_NEW = (apl.GATHER_LIVE
                   .replace("v += partials[s * kv_lora + j];   // fixed order",
                            "v += ld_l1(partials + s * kv_lora + j);   // fixed order (etx lever l1_loads)")
                   .replace("q_pe_lds[i] = q_head[qk_nope + i];", "q_pe_lds[i] = ld_l1(q_head + qk_nope + i);"))


def apply(src: Path) -> None:
    att = src / "kernels/attention.h"
    a = att.read_text()
    live = "kv_post_live(" in a
    if live:
        for old, new in LIVE:
            a = _once(a, old, new, NAME)
        a = _once(a, apl.GATHER_LIVE, GATHER_LIVE_NEW, NAME)
        att.write_text(a)
    texts: dict[str, str] = {}
    for rel, old, new in EDITS:
        text = texts.get(rel)
        if text is None:
            text = (src / rel).read_text()
        texts[rel] = _once(text, old, new, NAME)
    for rel, text in texts.items():
        (src / rel).write_text(text)
