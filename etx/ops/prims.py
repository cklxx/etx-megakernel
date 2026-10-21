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

PRIMITIVES = {"wave_sum": "etx_wave_sum", "wave_sum0": "dsl_wave_sum0", "uniform": "etx_uniform"}
DEFAULT_MACHINE = "gfx942"          # the machine of every measured build; a caller passes another model explicitly
MARK = "@@ETX_PRIM:{}@@"


def machine_or_default(machine: MachineModel | str | None) -> MachineModel:
    if machine is None:
        return load_machine(DEFAULT_MACHINE)
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
