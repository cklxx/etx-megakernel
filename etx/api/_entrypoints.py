"""Implementations of the supported Python entry points."""
from pathlib import Path as _Path

from etx.codegen import (
    emit_kernel as _emit_kernel,
    emit_lowering_header as _emit_lowering_header,
    emit_plan_json as _emit_plan_json,
)
from etx.codegen.plan_header import emit_plan_header as _emit_plan_header
from etx.compiler.frontend.contract import load_contract as _load_contract
from etx.frontends import (
    emit_tile_decls as _emit_tile_decls,
    emit_triton_skeleton as _emit_triton_skeleton,
)
from etx.passes import compile_graph as _compile_graph


def plan(graph, machine, bindings, runtime=None, options=None):
    """Verify and schedule a Graph, returning a Plan; machine may be an arch name."""
    return _compile_graph(graph, machine, bindings, runtime, options)


def compile(plan, output_dir):
    """Lower a Plan to source/header/JSON files and return their directory.

    This is source generation. Compile/link the emitted HIP or CUDA sources
    with the tile sources using the target's device toolchain before execution.
    """
    out = _Path(output_dir)
    # Generate everything before writing, so invalid plans do not leave partial output.
    ext = 'hip' if plan.machine.vendor == 'amd' else 'cu'
    files = {
        'etx_lowering.h': _emit_lowering_header(plan.machine),
        'etx_tiles.h': _emit_tile_decls(plan.graph),
        'plan.json': _emit_plan_json(plan),
        'plan_data.h': _emit_plan_header(plan, 0),
        'decisions.log': '\n'.join(plan.log) + '\n',
    }
    for device in range(plan.n_devices):
        files[f'megakernel_d{device}.{ext}'] = _emit_kernel(plan, device)
    if any(grid.body.kind == 'triton' for grid in plan.graph.grids):
        files['megakernel_triton.py'] = _emit_triton_skeleton(plan)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding='utf-8')
    return out


def load_model(path):
    """Load and validate a model.json contract, resolving paths and config hash.

    Returns model metadata; the caller supplies weight buffers and backend
    callbacks when creating a runtime descriptor.
    """
    return _load_contract(path)
