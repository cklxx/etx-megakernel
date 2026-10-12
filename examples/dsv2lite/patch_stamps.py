"""Add measurement boundaries to a COPY of pinned fleet 3fb67a6 sources.

Run after any performance patches: python -m examples.dsv2lite.patch_stamps SRC.
The build must define ETX_DSV2_STAMPS2; the unpatched/default source is untouched.
Stamps are thread-0 source boundaries, not synchronized workgroup completion.
The baseline gamma interval includes its original apply/round/store loop.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"stamps2: expected one anchor, found {text.count(old)}: {old!r}")
    return text.replace(old, new)


def function_end(text: str, start: int) -> int:
    """Find a function's balanced closing brace (the pinned bodies have no brace literals)."""
    depth = 0
    for i in range(text.index("{", start), len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if not depth:
            return i + 1
    raise ValueError("unclosed function")


def apply(src: Path) -> None:
    kernel = src / "kernels/fleet_kernel.hip"
    attention = src / "kernels/attention.h"
    text, attn = kernel.read_text(), attention.read_text()
    text = once(text, '#include "attention.h"', '''// ETX measurement-only source boundaries: no drains or added barriers.
#if !defined(ETX_DSV2_STAMPS2) || !ETX_DSV2_STAMPS2
#error "stamped fleet copy requires -DETX_DSV2_STAMPS2"
#endif
static __device__ __forceinline__ void etx_ds_stamp(uint64_t* stamps, int index) {
    if (stamps && threadIdx.x == 0) {
        asm volatile("" ::: "memory");
        stamps[index] = __builtin_amdgcn_s_memrealtime();
        asm volatile("" ::: "memory");
    }
}
#include "attention.h"''')
    # ids_blk is used only by lm_head; target qkv/router/merge never use it.
    # Overlay instead of growing the fixed graph LDS allocation.
    text = once(text, "    int   ids_blk[kBlock];", "    union { int ids_blk[kBlock]; uint64_t stamps2[13]; };")
    matches = list(re.finditer(r"__device__ inline void rms_norm_lds\w*\(", text))
    for match in reversed(matches):
        start, end = match.start(), function_end(text, match.start())
        body = text[start:end]
        body = once(body, "float* __restrict__ v, float* smem) {",
                    "float* __restrict__ v, float* smem, uint64_t* etx_stamps = nullptr) {\n"
                    "    etx_ds_stamp(etx_stamps, 0); // norm begin")
        # Both the baseline and ordered-tree patch retain these named assignments.
        tree = re.search(r"    const float ssq = [^;]+;", body)
        if tree is None:
            raise ValueError("stamps2: missing norm reduction anchor")
        body = body[:tree.start()] + "    etx_ds_stamp(etx_stamps, 1); // tree begin\n" + tree[0] + "\n    etx_ds_stamp(etx_stamps, 2); // tree end" + body[tree.end():]
        # The final loop is gamma load + application in baseline, application
        # only when the performance patch already hoists gamma. Do not split it.
        loop = body.rfind("#pragma unroll 8")
        if loop < 0:
            raise ValueError("stamps2: missing norm apply loop")
        body = body[:loop] + "    etx_ds_stamp(etx_stamps, 3); // gamma/apply begin\n" + body[loop:]
        body = once(body, "    __syncthreads();\n}", "    etx_ds_stamp(etx_stamps, 4); // gamma/apply end\n    __syncthreads();\n    etx_ds_stamp(etx_stamps, 5); // norm end\n}")
        text = text[:start] + body + text[end:]
    # Pass scratch only for the qkv and post-attention norm sites, never lm_head.
    text, count = re.subn(r"(rms_norm_lds\w*\(d, w\.layer_norm_(?:in\[L\]|post\[layer\]), s\.vec, s\.smem)\)",
                          r"\1, tc.trace ? s.stamps2 : nullptr)", text)
    if count < 2:
        raise ValueError("stamps2: missing qkv/router norm call sites")
    text = once(text, "            const int q_rows = heads_per_xcd * d.q_head_dim;",
                "            etx_ds_stamp(tc.trace ? s.stamps2 : nullptr, 6); // q GEMV begin\n            const int q_rows = heads_per_xcd * d.q_head_dim;")
    text = once(text, "            const int kv_rows = d.kv_lora + d.qk_rope;",
                "            etx_ds_stamp(tc.trace ? s.stamps2 : nullptr, 7); // q end / kv_a begin\n            const int kv_rows = d.kv_lora + d.qk_rope;")
    text = once(text, "            break;\n        }\n\n        // Attention: kv post",
                "            etx_ds_stamp(tc.trace ? s.stamps2 : nullptr, 8); // kv_a end\n            break;\n        }\n\n        // Attention: kv post")
    text = once(text, "                         d.kv_lora, d.qk_nope, d.v_head, s.q_c,\n                         t.kv_chunk * rows, rows);",
                "                         d.kv_lora, d.qk_nope, d.v_head, s.q_c,\n                         t.kv_chunk * rows, rows, tc.trace ? s.stamps2 : nullptr);")
    attn = once(attn, "float* __restrict__ o_c_lds, int row0, int n_rows) {",
                "float* __restrict__ o_c_lds, int row0, int n_rows, uint64_t* etx_stamps = nullptr) {\n    etx_ds_stamp(etx_stamps, 9); // merge M1 begin")
    attn = once(attn, "    const float inv_l = 1.f / gl;", "    const float inv_l = 1.f / gl;\n    etx_ds_stamp(etx_stamps, 10); // M1 end / M2 begin")
    attn = once(attn, "    // W_UV[h] is [v_head, kv_lora]", "    etx_ds_stamp(etx_stamps, 11); // M2 end / M3 begin\n    // W_UV[h] is [v_head, kv_lora]")
    attn = once(attn, "              n_rows, kv_lora, /*xcd=*/0, /*n_xcds=*/1, /*worker=*/0, /*n_workers=*/1);",
                "              n_rows, kv_lora, /*xcd=*/0, /*n_xcds=*/1, /*worker=*/0, /*n_workers=*/1);\n    etx_ds_stamp(etx_stamps, 12); // M3 end")
    kernel.write_text(text)
    attention.write_text(attn)
    print(f"patched {src} (ETX_DSV2_STAMPS2; 16-word prologue format)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    apply(Path(sys.argv[1]))
