"""Lever kchunk352 (D3): fleet's gate_up -> down hand-off in four K-chunks of 352 rows of h instead of one piece.

Two edits to the pinned source (docs/deepseek/DSV2-D1-D5.md, D3):
  fleet_types.h     kExpertKChunk 2048 -> 352 (1408 rows of h = 4 chunks; a multiple of 8, the 16-byte weight loads,
                    and of 32, so no h cache line straddles two chunks)
  fleet_kernel.hip  after each in-body chunk wait of the expert down, `buffer_inv sc1` -- fleet's XCD-local acquire is
                    `buffer_inv sc0`, which does not invalidate the CU's L1 (sync_bench loadscope, 2026-10-13 run 5),
                    so the down could stage a stale h line; sc1 is the acquire the default path's event wait performs.
The chunk signalling itself is fleet's (gemv_gate_up_rows FLEET_CHUNKS_DONE, kept by gemv_no_terminal_all) and the
chunk waits are fleet's (TASK_EXPERT_DOWN, FLAG_CHUNK_WAIT); fleet_shim.hip's ETX_DSV2_KCHUNKS wrappers turn them on.
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.patch_fleet import _once

NAME = "kchunk352"
ROWS = 352

EDITS = [
    ("runtime/fleet_types.h",
     "constexpr int kExpertKChunk = 2048; // rows of h per gate_up→down tile (taskgraph EXPERT_K_CHUNK);\n",
     f"constexpr int kExpertKChunk = {ROWS};  // etx lever kchunk352 (D3): 4 chunks of h per gate_up→down hand-off\n"),
    ("kernels/fleet_kernel.hip",
     "                        chunk_ok = wait_event(rt, tc.epoch, t.local_event + c, SCOPE_XCD_LOCAL,\n"
     "                                              xcd, t.wait_count) ? 1 : 0;\n",
     "                        chunk_ok = wait_event(rt, tc.epoch, t.local_event + c, SCOPE_XCD_LOCAL,\n"
     "                                              xcd, t.wait_count) ? 1 : 0;\n"
     "                        if (chunk_ok) asm volatile(\"buffer_inv sc1\" ::: \"memory\");   // etx lever kchunk352: L1 too\n"),
]


def apply(src: Path) -> None:
    texts: dict[str, str] = {}
    for rel, old, new in EDITS:
        text = texts.get(rel)
        if text is None:
            text = (src / rel).read_text()
        texts[rel] = _once(text, old, new, NAME)
    for rel, text in texts.items():
        (src / rel).write_text(text)
