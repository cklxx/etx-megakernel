'Memory contract check for one imported producer/consumer pair: attn_reduce -> wvsplitk_y2 (o_proj).\n\ndocs/core/CONTRACT-acquire-2026-10-05.md states the contract. This tool checks that a build still matches it:\n\n  python -m etx.tools.acquire_contract --build <build dir> --cflags "<device -D flags of that build>"\n\nThe build directory holds imports.bc (or pass --ir <file.ll|file.bc>), megakernel_d0.hip, imp_tiles.hip,\netx_lowering.h and plan.json. The tool checks:\n\n  1. IR: the wrapper argument offsets and types, every specialisation body\'s parameter attributes, an allowlist of\n     opcodes and callees, and the root parameter of every global access. A global pointer must come from a\n     parameter through getelementptr, phi or select. Anything else is an unknown alias pattern.\n  2. Bindings: every operator argument that names attn_out in etx.importer.spec.\n  3. Emitted source: per layer, the producer\'s arrive and the consumer\'s wait use the same DEVICE event, and the\n     code order is wait -> ACQUIRE_DEVICE -> barrier -> body -> all-wave drain -> barrier -> thread 0 arrive.\n  4. Plan: event scope, producer count and placement, consumer inputs.\n  5. Lowering and runtime: the DEVICE release / acquire / poll / arrive macros and the arrive function text.\n  6. Compile flags: only known -D macros. Any flag that removes a fence fails.\n\nFail-closed: an unknown opcode, callee, pointer origin, attribute, source line or flag is an error.\nExit 0: contract holds. Exit 1: violations, one per line. Exit 2: refused (an input is missing).\nThis is a static check of what was emitted. It does not prove the GPU memory model.'
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from etx.tools.contracts.source_control import compact
from .schema import (
    ROOT,
    Refusal,
    Param,
    P,
    I,
    PRODUCER,
    CONSUMER,
    PRODUCER_TILE,
    CONSUMER_TILE,
    CROSS,
    CONTRACT,
    GRID_OFFSETS,
    ETX_PARAMS,
    CROSS_USERS,
    ATTN_EXPORT_PREFIX,
    OPCODES,
    CALLEES,
    WG_FENCE,
    ALLOWED_DEFINES,
    LOWERING,
    ARRIVE_FLUSH,
    ATOMIC_SUB_DEVICE,
    WAIT_SCOPE,
    PRE_WAIT,
    IMP_TOUCH,
    Function,
)
from .llvm import (
    split_top,
    parse_param,
    parse_module,
    statements,
    ll_type,
    check_wrapper,
    PTR1,
    check_body,
    check_attn_out_readnone,
    load_ir,
    check_ir,
)
from .source import (
    check_bindings,
    P1,
    P2,
    P3,
    BETWEEN,
    TRACE,
    cases,
    CONSUMER_BLOCK,
    PRODUCER_BLOCK,
    match_block,
    check_source,
    check_plan,
    top_level_defines,
    check_lowering,
    function_text,
    check_atomic_sub_device,
    check_runtime,
    check_cflags,
)
from .acquire import (
    run,
    summary,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
