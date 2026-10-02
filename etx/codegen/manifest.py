"""The feature manifest: what the emitter actually emitted, published to every translation unit that must agree
with it.

A feature that changes device code or host behaviour and that more than one side must agree about belongs here and
nowhere else. The failure mode it exists to prevent is a `-D` on one compiler invocation and an independent
re-derivation of the same decision somewhere else; every occurrence has been silent, producing a build that looks
healthy and is wrong:

  2026-10-01 round 14  ETX_VL_NB      the tile was emitted without it while the plan batched, so extra requests
                                      would have read column 0.
  2026-10-01 round 19  ETX_DEV_RESET  build.sh appended the define to an array that a later `if` overwrote whenever
                                      ETX_VL_COH was set, so the kernel's reset tail was preprocessed out while the
                                      host believed it ran: 0/32 tokens, no diagnostic.
  2026-10-01 round 20  ETX_DEV_RESET  the same define never reaching the separate device-code compile stage.
  2026-10-01 round 30  the top-k fold model.py derived "do not fold" while build.sh's define defaulted to "fold", so
                                      the plan gave the top-k its own phase and the expert tiles recomputed it
                                      anyway: 0.63 ms, tokens still matched, nothing failed.

The rule now has teeth. `guard_block()` is emitted into both generated artefacts -- `megakernel_d0.hip` for the
device translation unit and `plan_data.h` for the host one -- ahead of anything that reads a feature macro. It
`#error`s if the macro arrived from a command line, then defines it from the plan. A reviewer who adds a feature the
old way gets a compile failure naming this file, not a wrong answer.

Each entry maps the plan's `Graph.codegen` key to the macro the sources test and how it is spelled:
  "flag"  the macro is defined only when the feature is on, so sources may use #ifdef (the tile sources do);
  "value" the macro always has an integer value, so sources may use #if or read it as a number.
"""
from __future__ import annotations

# codegen key -> (macro the sources test, spelling, default when the plan says nothing)
FEATURES: dict[str, tuple[str, str, int]] = {
    "argpf": ("ETX_EMIT_ARGPF", "value", 0),        # next-task argument prefetch (kernel.py)
    "bar2": ("ETX_EMIT_BAR2", "value", 0),          # two-barrier static loop (kernel.py)
    "dev_reset": ("ETX_EMIT_DEV_RESET", "value", 0),  # the kernel resets its own per-step state (host.h)
    "fold_topk": ("ETX_EMIT_FOLD_TOPK", "value", 0),  # the top-k folded into the expert tasks (host.hip)
    "coh": ("ETX_DSL_COH", "flag", 0),              # the DSL tiles' coherent activation access (etx/ops/coh.py)
    "nb": ("ETX_VL_NB", "value", 1),                # activation columns the tiles were emitted for (host.hip)
    "requests": ("ETX_VL_REQUESTS", "value", 1),    # concurrent decode requests the plan carries (host.hip)
    "uc": ("ETX_VL_UC", "flag", 0),                 # cross-XCD tensors in uncached memory (host.hip, lowering)
    "dsl_bigk": ("ETX_EMIT_DSL_BIGK", "value", 0),  # MoE K > 4096 chunked in the DSL tiles (gemv.py, host.hip guards)
}

# Macros that used to be passed as -D for a feature now in FEATURES. Trapped by name so an old build script, an old
# shell array, or a copied command line fails loudly instead of silently disagreeing with the plan.
RETIRED: tuple[str, ...] = ("ETX_DSL_COH", "ETX_VL_NB", "ETX_VL_REQUESTS", "ETX_VL_UC", "ETX_COHERENT_DEVICE_DATA",
                            "ETX_DEV_RESET", "ETX_VL_NO_TOPK_FOLD")


def guard_block(codegen: dict) -> list[str]:
    """The trap and the definitions, in that order. Emitted near the top of every generated artefact."""
    out = ["", "// ---- feature manifest (etx/codegen/manifest.py): the emitter's decisions, not the command line's."]
    out.append("// Passing any of these as -D is the bug class this block exists to catch; the plan decides them.")
    for mac in RETIRED:
        out.append(f"#ifdef {mac}")
        out.append(f'#error "{mac} must not be defined on the command line: the plan publishes it '
                   '(etx/codegen/manifest.py). Set the feature in examples/vllm_llm/model.py so the emitter and '
                   'every translation unit agree by construction."')
        out.append("#endif")
    for key, (mac, kind, default) in FEATURES.items():
        v = codegen.get(key, default)
        v = int(v) if not isinstance(v, bool) else int(bool(v))
        if kind == "flag":
            if v:
                out.append(f"#define {mac} 1")
        else:
            out.append(f"#define {mac} {v}")
    out.append("")
    return out
