"""Tile primitives (docs/core/PLAN-generic-decoder-2026-10-08.md section 6.4).

A tile generator calls a primitive by name. The C definition of the primitive comes from the machine model
(`primitives:` in etx/machine/arch/<arch>.yaml). This module does not test an architecture name.

| Primitive   | C function      | Result                                  |
|-------------|-----------------|-----------------------------------------|
| `wave_sum`  | `etx_wave_sum`  | the sum of `v` over the wave, every lane |
| `wave_sum0` | `dsl_wave_sum0` | the same sum, defined in lane 0 only     |
| `uniform`  | `etx_uniform`  | broadcast lane 0's integer to the wave   |

The gfx942 text of `wave_sum0` is the ETX_DSL_WSUM reduction that etx/ops/gemv.py had inline before this module.
The gfx942 reduction text is preserved (tests/test_prims.py).
"""
from __future__ import annotations

from etx.machine.model import MachineModel, load_machine

PRIMITIVES = {"wave_sum": "etx_wave_sum", "wave_sum0": "dsl_wave_sum0", "uniform": "etx_uniform",
              "wave_max": "etx_wave_max", "wave_max_u32": "etx_wave_max_u32",
              "block_sum": "etx_block_sum", "block_max": "etx_block_max"}
MARK = "@@ETX_PRIM:{}@@"


def machine_or_default(machine: MachineModel | str | None) -> MachineModel:
    if machine is None:
        raise ValueError("tile primitive lowering requires an explicit machine")
    return load_machine(machine) if isinstance(machine, str) else machine


def lower(machine: MachineModel | str | None, name: str) -> str:
    """The C definition of primitive `name` for `machine`, without a trailing newline."""
    if name not in PRIMITIVES:
        raise KeyError(f"unknown tile primitive {name!r}; known: {sorted(PRIMITIVES)}")
    m = machine_or_default(machine)
    text = m.primitives.get(name)
    if not text:
        raise KeyError(f"{m.name}: no lowering for tile primitive {name!r} (add it under primitives: in its yaml)")
    if f" {PRIMITIVES[name]}(" not in text:
        raise ValueError(f"{m.name}: the {name!r} lowering must define {PRIMITIVES[name]}()")
    return text.rstrip("\n")


def expand(src: str, machine: MachineModel | str | None) -> str:
    """Replace every primitive mark in `src` with the machine's definition. A mark that stays is an error."""
    m = machine_or_default(machine)
    for name in PRIMITIVES:
        mark = MARK.format(name)
        if mark in src:
            src = src.replace(mark, lower(m, name))
    if "@@ETX_PRIM:" in src:
        raise KeyError("unknown primitive mark in the tile source: " + src[src.index("@@ETX_PRIM:"):][:40])
    return src


def support(machine: MachineModel | str, name: str) -> str:
    """A source-support block owned by the target's primitive table."""
    m = machine_or_default(machine)
    try:
        return m.primitives[name].rstrip("\n")
    except KeyError:
        raise KeyError(f"{m.name}: no lowering for tile support {name!r}") from None


def emit_source(src: str, machine: MachineModel | str, lds_bytes: int = 16) -> str:
    """Translate the legacy DSL spelling using the target's source-support table.

    AMD's source is deliberately untouched. CUDA support lives with its machine
    primitives rather than in the shared HIP runtime headers.
    """
    m = machine_or_default(machine)
    if "source_prelude" not in m.primitives:
        return src
    from etx.ops import coh
    src = src.replace(coh.HELPERS, support(m, "coherence"))
    for header in ('#include "etx/import.h"', '#include "etx/tile_util.h"'):
        src = src.replace(header, "")
    # CUDA has generic pointers and native vector structs, not Clang vectors or
    # AMD address-space qualifiers. Keep the DSL's indexing and alignment.
    import re
    src = re.sub(r"__attribute__\(\(address_space\(\d\)\)\)", "", src)
    src = re.sub(r"typedef unsigned int dsl_u32x([24]) __attribute__\(\(ext_vector_type\([24]\)\)\);", "", src)
    src = src.replace('__hip_bfloat16', '__nv_bfloat16')
    src = src.replace('__builtin_nontemporal_load(p)', '(*p)')
    src = src.replace('_Static_assert(', 'static_assert(')
    src = src.replace('"v"(', '"r"(')  # integer keepalive operands
    # Floating-point keepalives need the PTX floating register constraint.
    src = src.replace('"r"(acc[b])', '"f"(acc[b])').replace('"r"(v)', '"f"(v)')
    src = src.replace('__shfl_xor(', 'etx_shuffle_xor(').replace('__shfl(', 'etx_shuffle(')
    # The fixed-capacity top-k tiles cover 128 experts on either wave size.
    capacity = 128 // m.wave_size
    src = src.replace('k < 2', f'k < {capacity}')
    for decl in ('float p[2]', 'bool taken[2]', 'float raw[2]'):
        src = src.replace(decl, decl.replace('[2]', f'[{capacity}]'))
    src = src.replace('raw[0] = p[0]; raw[1] = p[1];',
                      f'for (int k = 0; k < {capacity}; ++k) raw[k] = p[k];')
    # The optional paired-load shortcut only fills two slots; use the general loop.
    src = src.replace('#if defined(ETX_TOPK_PAIRED_LOADS) && !defined(ETX_ABL_TOPK_NOLOAD)', '#if 0')
    # The vector attention prologue uses 16 active lanes, so it also fits warp32.
    src = src.replace('HD == 128 && ETX_WAVE_SIZE == 64 && X',
                      'HD == 128 && ETX_WAVE_SIZE >= 16 && X')
    arena = lds_bytes
    src = re.sub(r'static_assert\(\d+ <= ETX_IMPORT_LDS',
                 f'static_assert({arena} <= ETX_IMPORT_LDS', src)
    blocks = [f'#ifndef ETX_IMPORT_LDS\n#define ETX_IMPORT_LDS {arena}\n#endif',
              support(m, "source_prelude")]
    for name in PRIMITIVES:
        definition = lower(m, name)
        src = src.replace(definition, "")
        blocks.append(definition)
    blocks.append(support(m, "topk_key32"))
    return "\n".join(blocks) + "\n" + src
