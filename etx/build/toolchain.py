"""Offline gfx942 compilation using an already-installed, networkless container."""
import hashlib
import json
import os
from pathlib import Path
import uuid

from etx.api.artifacts import SourceArtifact
from etx.api.build import DeviceCheck, BuildArtifact
from etx.api.execution import ProcessRequest
from etx.artifacts.sources import save_sources
from etx.contracts.execution import execute


def check_device_sources(artifact, *, output, docker, image, deadline_s=90):
    return _compile_sources(artifact, output=output, docker=docker, image=image, deadline_s=deadline_s)


def build(artifact, *, output, docker, image, deadline_s=90):
    return _compile_sources(artifact, output=output, docker=docker, image=image,
                            deadline_s=deadline_s, link_host=True)


def _compile_sources(artifact, *, output, docker, image, deadline_s, link_host=False):
    if not isinstance(artifact, SourceArtifact):
        raise TypeError('device check requires SourceArtifact')
    if artifact.planned.request.target != 'gfx942' or artifact.external_sources:
        raise ValueError('offline check supports self-contained gfx942 source artifacts only')
    from etx.artifacts.native import checked_native_units
    units = checked_native_units(artifact)
    if link_host:
        from etx.artifacts.native import checked_host
        host, units = checked_host(artifact)
    if type(image) is not str or not image or image.startswith('-'):
        raise ValueError('compiler image must be explicit')
    root = Path(output).absolute()
    # Validate command settings before creating any outputs.
    ProcessRequest((docker, 'version'), str(root.parent), deadline_s)
    save_sources(artifact, root)
    environment = {'PATH': '/usr/bin:/bin'}

    def invoke(args, name, deadline=deadline_s):
        request = ProcessRequest(tuple([docker, *args]), str(root), deadline,
                                 environment=environment, silence_s=10)
        return execute(request, output=root / 'checks' / name)

    inspected = invoke(['image', 'inspect', '--format', '{{.Id}}', image], 'image')
    if inspected.status != 'complete':
        raise RuntimeError(f'installed compiler image unavailable; see {root / "checks/image"}')
    image_id = (root / 'checks/image/output.log').read_text().strip()
    if not image_id.startswith('sha256:') or len(image_id) != 71:
        raise ValueError('docker did not report an immutable image identity')

    def container(args, name):
        container_name = 'etx-check-' + uuid.uuid4().hex
        try:
            return invoke(['run', '--name', container_name, '--pull=never', '--network=none',
                           '--user', f'{os.getuid()}:{os.getgid()}',
                           '--volume', f'{root}:/work', '--workdir', '/work', image_id, *args], name)
        finally:
            # Killing the docker client does not stop the daemon's container.
            # Always remove the named container before admitting a compiler result.
            removed = invoke(['rm', '--force', container_name], name + '-cleanup', 20)
            if removed.status != 'complete':
                raise RuntimeError(f'compiler container cleanup not confirmed; see {root / "checks"}')

    version = container(['hipcc', '--version'], 'version')
    if version.status != 'complete':
        raise RuntimeError(f'compiler version check failed; see {root / "checks/version"}')
    toolchain = dict(image=image, image_id=image_id,
                     version=(root / 'checks/version/output.log').read_text(),
                     target='gfx942', network='none', gpu_access=False,
                     compiler_kind='offline-device-object')
    if units:
        toolchain['native_units'] = list(units)
    ranks, tiles, objects = [], [], {}
    def compile_unit(source, obj, name, defines):
        # HIP RDC stops at LLVM bitcode. Run the AMDGPU backend explicitly so
        # DEVICE_CHECKED means an ELF device object, not a host/offload bundle.
        # Separate objects still need the later link gate.
        command = ('source=$1; object=$2; shift 2; '
                   'hipcc -O3 -std=c++17 -fgpu-rdc --offload-device-only --offload-arch=gfx942 '
                   '-I/work/include -I/work "$@" -c "$source" -o "$object.bc" && '
                   '/opt/rocm/llvm/bin/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx942 '
                   '-nogpulib -x ir -O3 -c "$object.bc" -o "$object" -Rpass-analysis=kernel-resource-usage')
        flags = [f'-D{key}={value}' for key, value in sorted(defines.items())]
        args = ['sh', '-c', command, 'etx-device-compile', source, obj, *flags]
        print(f'compile unit={name} source={source}', flush=True)
        result = container(args, name)
        print(f'compile unit={name} status={result.status} log={root / "checks" / name}', flush=True)
        if result.status == 'complete':
            with (root / obj).open('rb') as payload:
                header = payload.read(20)
                if header[:6] != b'\x7fELF\x02\x01' or int.from_bytes(header[18:20], 'little') != 224:
                    raise ValueError(f'{name} did not produce a 64-bit AMDGPU ELF device object')
                payload.seek(0)
                objects[obj] = hashlib.file_digest(payload, 'sha256').hexdigest()
        return result

    # Inline mode includes the same captured tile bytes in rank translation units.
    rank_defines = units[0]['defines'] if units else {}
    for rank, source in enumerate(artifact.rank_sources):
        ranks.append(compile_unit(source, f'megakernel_d{rank}.o', f'rank-{rank}', rank_defines))
    for index, unit in enumerate(units):
        tiles.append(compile_unit(unit['source'], f'native_tile_{index}.o', f'tile-{index}', unit['defines']))
    status = 'DEVICE_CHECKED' if all(result.status == 'complete' for result in ranks + tiles) else 'FAILED'
    receipt = DeviceCheck(artifact.identity, toolchain, tuple(ranks), objects, status, tuple(tiles))
    with (root / 'device_check.json').open('x') as target:
        json.dump(receipt.to_dict(), target, sort_keys=True, indent=2)
        target.write('\n')
    if not link_host:
        return receipt
    linked, files = None, {}
    if receipt.status == 'DEVICE_CHECKED':
        flags = [f'-D{key}={value}' for key, value in sorted(rank_defines.items())]
        sources = list(artifact.rank_sources)
        if not artifact.planned.request.options.get('inline_tiles', False):
            sources += [unit['source'] for unit in units]
        print('link host=' + host, flush=True)
        host_flags = ['-pthread'] if artifact.planned.request.workload.config.get('provider') == 'gemm_rs' else []
        linked = container(['hipcc', '-O2', '--offload-arch=gfx942', '-fgpu-rdc',
                            '-I/work/include', '-I/work', *flags, *host_flags, *sources, host, '-o', 'run'], 'link')
        print('link status=' + linked.status, flush=True)
        if linked.status == 'complete':
            from etx.artifacts.builds import executable_digest
            files['run'] = executable_digest(root / 'run')
    result = BuildArtifact(receipt, linked, files, 'EXECUTABLE' if files else 'FAILED')
    with (root / 'build.json').open('x') as target:
        json.dump(dict(identity=result.identity, **result.to_dict()), target, sort_keys=True, indent=2)
        target.write('\n')
    return result
