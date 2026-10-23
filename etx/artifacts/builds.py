"""Relocatable linked build receipts; no executable is run by this module."""
import hashlib
import json
from pathlib import Path
import struct

from etx.api.build import BuildArtifact, DeviceCheck
from etx.api.execution import ProcessRequest, ProcessResult
from etx.artifacts.sources import load_sources


def executable_digest(path):
    """Require an x86-64 linked ELF with a nonempty HIP device-image section."""
    data = Path(path).read_bytes()
    if (len(data) < 64 or data[:6] != b'\x7fELF\x02\x01' or
            int.from_bytes(data[16:18], 'little') not in (2, 3) or
            int.from_bytes(data[18:20], 'little') != 62):
        raise ValueError('linked output is not an x86-64 ELF executable')
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, names = struct.unpack_from('<HHH', data, 58)
    if size != 64 or not 0 < names < count or offset + size * count > len(data):
        raise ValueError('invalid ELF section table')
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + index * size) for index in range(count)]
    start, length = sections[names][4:6]
    if start + length > len(data):
        raise ValueError('invalid ELF section names')
    strings = data[start:start + length]
    for section in sections:
        name, _, _, _, start, length, *_ = section
        if name >= len(strings):
            raise ValueError('invalid ELF section name')
        if strings[name:].split(b'\0', 1)[0] == b'.hip_fatbin':
            if length == 0 or start + length > len(data):
                raise ValueError('invalid HIP device image')
            return hashlib.sha256(data).hexdigest()
    raise ValueError('linked output has no HIP device image')


def load_build(directory):
    root = Path(directory).resolve()

    def path(name):
        candidate = root / name
        if any(item.is_symlink() for item in (candidate, *candidate.parents)
               if item != root and root in item.parents):
            raise ValueError('build payload must not traverse symlinks')
        return candidate

    record = json.loads(path('build.json').read_text())
    identity = record.pop('identity')
    device = record.pop('device_check')
    device['ranks'] = tuple(ProcessResult(**item) for item in device['ranks'])
    device['tiles'] = tuple(ProcessResult(**item) for item in device.get('tiles', []))
    check = DeviceCheck(**device)
    link = record.pop('link')
    artifact = BuildArtifact(check, ProcessResult(**link) if link is not None else None, **record)
    if artifact.identity != identity:
        raise ValueError('build identity mismatch')
    source = load_sources(root)
    if source.identity != check.source_identity:
        raise ValueError('build source identity mismatch')
    from etx.artifacts.native import checked_host
    _, units = checked_host(source)
    if len(check.ranks) != len(source.rank_sources) or len(check.tiles) != len(units):
        raise ValueError('build compiler inventory mismatch')
    if check.toolchain.to_dict().get('native_units') != list(units):
        raise ValueError('build native compiler configuration mismatch')
    if json.loads(path('device_check.json').read_text()) != check.to_dict():
        raise ValueError('device-check receipt mismatch')
    from etx.contracts.execution import read_process_result
    checks = [(f'rank-{i}', result) for i, result in enumerate(check.ranks)]
    checks += [(f'tile-{i}', result) for i, result in enumerate(check.tiles)]
    if artifact.link is not None:
        checks.append(('link', artifact.link))
    required = [('image', None), ('version', None), ('version-cleanup', None)]
    required += [(name + '-cleanup', None) for name, _ in checks]
    for name, expected in checks + required:
        folder = path('checks/' + name)
        for filename in ('request.json', 'result.json', 'output.log'):
            path(f'checks/{name}/{filename}')
        request = ProcessRequest(**json.loads((folder / 'request.json').read_text()))
        actual = read_process_result(folder, request=request)
        if ((expected is not None and actual != expected) or
                (expected is None and (actual.backend != 'host-process' or actual.status != 'complete'))):
            raise ValueError('build compiler evidence mismatch')
    if ((root / 'checks/version/output.log').read_text() != check.toolchain['version'] or
            (root / 'checks/image/output.log').read_text().strip() != check.toolchain['image_id']):
        raise ValueError('build toolchain evidence mismatch')
    for name, digest in check.objects.items():
        payload = path(name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('device object checksum mismatch')
    for name, digest in artifact.files.items():
        if executable_digest(path(name)) != digest:
            raise ValueError('executable checksum mismatch')
    return artifact
