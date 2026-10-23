"""All-rank source emission through the existing compiler/codegen services."""
from pathlib import Path

from etx.api.artifacts import SourceArtifact
from etx.codegen import emit_kernel, emit_lowering_header, emit_plan_json
from etx.codegen.plan_header import emit_plan_header
from etx.frontends import collect_sources, emit_tile_decls, emit_triton_skeleton
from .planning import recompile


def source_snapshot(root, *, paths):
    from etx.artifacts.identity import source_snapshot as snapshot
    return snapshot(root, paths)


def verify_source_snapshot(root, record):
    from etx.artifacts.identity import verify_source_snapshot as verify
    return verify(root, record)


def build(artifact, *, output, docker, image, deadline_s=90):
    from etx.build.toolchain import build as compile_and_link
    return compile_and_link(artifact, output=output, docker=docker, image=image, deadline_s=deadline_s)


def load_build(directory):
    from etx.artifacts.builds import load_build as load
    return load(directory)


def check_device_sources(artifact, *, output, docker, image, deadline_s=90):
    from etx.build.toolchain import check_device_sources as check
    return check(artifact, output=output, docker=docker, image=image, deadline_s=deadline_s)


def check_codegen(codegen, *, toolchain, docker='docker', logs=None, deadline_s=600):
    from etx.build.codegen import check_codegen as check
    return check(codegen, toolchain=toolchain, docker=docker, logs=logs, deadline_s=deadline_s)


def save_sources(artifact, directory):
    from etx.artifacts.sources import save_sources as save
    return save(artifact, directory)


def load_sources(directory):
    from etx.artifacts.sources import load_sources as load
    return load(directory)


def emit(planned):
    return artifact_from(planned, recompile(planned))


def artifact_from(planned, compiled):
    """SourceArtifact for a plan whose compiled form the caller already holds."""
    files, sources, headers = emit_payloads(compiled)
    from etx.artifacts.native import operator_sources
    native, external = operator_sources(planned.request, collect_sources(compiled.graph))
    files.update(native)
    return SourceArtifact(planned, files, tuple(sources), tuple(headers), external)


def emit_payloads(compiled, *, all_rank_headers=True, runtime_headers=True):
    """One emission path; legacy CLI preserves its historical file inventory."""
    ext = 'hip' if compiled.machine.vendor == 'amd' else 'cu'
    files = {'etx_lowering.h': emit_lowering_header(compiled.machine),
             'etx_tiles.h': emit_tile_decls(compiled.graph),
             'plan.json': emit_plan_json(compiled),
             'decisions.log': '\n'.join(compiled.log) + '\n'}
    sources, headers = [], []
    for rank in range(compiled.n_devices):
        source, header = f'megakernel_d{rank}.{ext}', f'plan_data_d{rank}.h'
        files[source] = emit_kernel(compiled, rank)
        if all_rank_headers:
            files[header] = emit_plan_header(compiled, rank)
            headers.append(header)
        sources.append(source)
    files['plan_data.h'] = files[headers[0]] if headers else emit_plan_header(compiled, 0)
    if any(grid.body.kind == 'triton' for grid in compiled.graph.grids):
        files['megakernel_triton.py'] = emit_triton_skeleton(compiled)
    # Capture runtime headers rather than depending on cwd or an unrecorded
    # installed runtime at compile time. No model/toolchain execution here.
    if runtime_headers:
        include = Path(__file__).resolve().parents[1] / 'runtime' / 'include'
        for path in sorted(include.rglob('*.h')):
            files['include/' + path.relative_to(include).as_posix()] = path.read_text()
    return files, sources, headers
