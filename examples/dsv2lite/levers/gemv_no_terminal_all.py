"""Lever Q-all, gemv_no_terminal_all: no pipelined GEMV issues its never-reduced terminal group.

fleet's three software-pipelined row loops (gemv.h @3fb67a6) issue the next group before reducing the current one and
check the end afterwards, so every active wave issues exactly one group past its last row (clamped to row cnt-1):
  gemv_rows_pipelined<RPI,D>   q, kv_a, merge W_UV <8,1>, router <2,4>, down <2,4>, lm_head <2,4>
  gemv_rows_pipelined_half<8>  o_proj (K = 256)
  gemv_gate_up_rows            expert and dense gate_up (one extra pair; also signals FLEET_CHUNKS_DONE per group)
This lever peels the tail of all three IN PLACE: full pairs keep issue-before-reduce in one basic block, the last one or
two groups reduce without issuing another, and gate_up's chunk signals keep their order (after reducing group g:
CHUNKS_DONE(g + stride)). It subsumes Q (gemv_no_terminal, qkv only) and composes with it.
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.patch_fleet import _once


def loops(issue: str, reduce: str, stride: str, indent: str, done: str | None = None) -> tuple[str, str]:
    """(original, peeled) text of one pipelined loop; `done` is gate_up's per-group chunk signal macro."""
    i, j = indent, indent + "    "
    sig = (lambda kk: f"{j}{done}({kk})\n") if done else (lambda kk: "")
    old = (f"{i}{issue}(bufA, k)\n"
           f"{i}while (true) {{\n"
           f"{j}const int k2 = k + {stride};\n"
           f"{j}{issue}(bufB, k2){'          // unconditional; clamped past the end' if issue == 'FLEET_ISSUE_GROUP' else ''}\n"
           f"{j}{reduce}(bufA, k)\n" + sig("k2") +
           f"{j}if (k2 >= cnt) break;\n"
           f"{j}const int k3 = k2 + {stride};\n"
           f"{j}{issue}(bufA, k3)\n"
           f"{j}{reduce}(bufB, k2)\n" + sig("k3") +
           f"{j}if (k3 >= cnt) break;\n"
           f"{j}k = k3;\n"
           f"{i}}}\n")
    new = (f"{i}{issue}(bufA, k)\n"
           f"{i}// etx lever gemv_no_terminal_all: full pairs keep issue-before-reduce; the tail issues nothing past cnt.\n"
           f"{i}while (k + 2 * {stride} < cnt) {{\n"
           f"{j}const int k2 = k + {stride};\n"
           f"{j}{issue}(bufB, k2)\n"
           f"{j}{reduce}(bufA, k)\n" + sig("k2") +
           f"{j}const int k3 = k2 + {stride};\n"
           f"{j}{issue}(bufA, k3)\n"
           f"{j}{reduce}(bufB, k2)\n" + sig("k3") +
           f"{j}k = k3;\n"
           f"{i}}}\n"
           f"{i}if (k + {stride} < cnt) {{\n"
           f"{j}const int k2 = k + {stride};\n"
           f"{j}{issue}(bufB, k2)\n"
           f"{j}{reduce}(bufA, k)\n" + sig("k2") +
           f"{j}{reduce}(bufB, k2)\n" + sig(f"k2 + {stride}") +
           f"{i}}} else {{\n"
           f"{j}{reduce}(bufA, k)\n" + sig(f"k + {stride}") +
           f"{i}}}\n")
    return old, new


LOOPS = {
    "gemv_rows_pipelined": loops("FLEET_ISSUE_GROUP", "FLEET_REDUCE_GROUP", "stride", "    "),
    "gemv_rows_pipelined_half": loops("FLEET_HISSUE", "FLEET_HREDUCE", "stride", "    "),
    "gemv_gate_up_rows": loops("FLEET_ISSUE_PAIR", "FLEET_REDUCE_PAIR", "kWaves", "        ", "FLEET_CHUNKS_DONE"),
}


def apply(src: Path) -> None:
    gemv = src / "kernels/gemv.h"
    g = gemv.read_text()
    for helper, (old, new) in LOOPS.items():
        g = _once(g, old, new, f"gemv_no_terminal_all ({helper})")
    gemv.write_text(g)
