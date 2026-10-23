import hashlib
import json
import struct
import pytest
from etx.testing import sealed_contract_probe


def decode_records(rows, *, batch):
    return sealed_contract_probe('routing-decode', value={'raw': rows, 'batch': batch}).to_dict()


def raw(batch=2):
    rows=[]
    for r in range(batch):
        b=bytearray()
        for t in range(32):
            for l in range(1,27):
                for x in range(8):
                    b+=struct.pack('<16I',33+t,1,*range(6),*[0x3e000000]*6,0,0)
        rows.append(bytes(b))
    return rows


def test_complete_native_record_decode():
    v=decode_records(raw(),batch=2)
    assert v['requests']==2 and len(v['effective_batch_histogram'])>0
    assert not v['performance_claim']


@pytest.mark.parametrize('fault',['truncated','duplicate','missing','wrong_epoch','expert','nan'])
def test_native_capture_faults_fail_closed(fault):
    rows=raw();b=bytearray(rows[0])
    if fault=='truncated':b=b[:-4]
    if fault=='duplicate':struct.pack_into('<I',b,4,2)
    if fault=='missing':struct.pack_into('<I',b,4,0)
    if fault=='wrong_epoch':struct.pack_into('<I',b,0,32)
    if fault=='expert':struct.pack_into('<I',b,8,64)
    if fault=='nan':struct.pack_into('<I',b,32,0x7fc00000)
    rows[0]=bytes(b)
    with pytest.raises(ValueError):decode_records(rows,batch=2)


def test_legacy_routing_exports_are_library_provider_aliases():
    assert all(sealed_contract_probe('routing-compatibility').values())


@pytest.mark.parametrize('batch', [2, 4])
def test_routing_evidence_without_example_imports_and_consumer_parity(tmp_path, batch):
    """Synthetic bytes prove CPU evidence plumbing, not device correctness."""
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    identity = tmp_path / 'CPU_SYNTHETIC_NOT_EXECUTABLE'
    identity.write_text('SYNTHETIC CPU ONLY ETX_PROGRAM_ROUTE_CAPTURE=1\n')
    roles = {role: {str(identity): sha(identity)} for role in (
        'source', 'plan', 'executable', 'compiler', 'math_flags', 'resources',
        'weights', 'inputs', 'prefixes', 'goldens', 'golden_producers')}
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(batch=batch, roles=roles,
                                     command=[str(identity)], deadline_seconds=10)))
    rows = raw(batch)
    files = {}
    for request, data in enumerate(rows):
        path = tmp_path / f'routes_r{request}.bin'
        path.write_bytes(data)
        files[str(path)] = sha(path)
    log = tmp_path / 'capture.log'
    log.write_text(f'G04 CAPTURE COMPLETE requests={batch} steps=32 routed_layers=26 replicas=8 slots=6 dense_layer=0\n'
                   + ''.join(f'BATCHREQ r={r} tokens=32/32 layers=27 warmup=32/32\n' for r in range(batch)))
    capture = tmp_path / 'capture.json'
    report = dict(decode_records(rows, batch=batch), record='G04', complete=True,
                  roles=roles, raw_files=files, log_sha256=sha(log), config_sha256=sha(config))
    capture.write_text(json.dumps(report))
    stage = {'command': ['capture', '--config', str(config)]}
    expected = dict(record='G04', complete=True, admitted=False, decision_sha256=sha(capture))
    assert sealed_contract_probe('routing-isolated-decision', value=stage, path=capture).to_dict() == expected
    assert sealed_contract_probe('decision', record='G04', value=stage, path=capture).to_dict() == expected
    assert sealed_contract_probe('routing-column-dependency', path=capture).to_dict() == dict(
        record='G04', complete=True, batch=batch, decision_sha256=sha(capture))

    # Only the session binds the registered configuration digest. Extraction
    # must not quietly strengthen the column consumer to require that binding.
    report['config_sha256'] = '0' * 64
    capture.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='configuration identity'):
        sealed_contract_probe('decision', record='G04', value=stage, path=capture)
    assert sealed_contract_probe('routing-column-dependency', path=capture)['complete'] is True

    # Only the column consumer requires an exact raw-file key set. Preserve
    # that extra predicate and its ProtocolError type rather than unifying it.
    report['config_sha256'] = sha(config)
    report['raw_files'][str(identity)] = sha(identity)
    capture.write_text(json.dumps(report))
    assert sealed_contract_probe('decision', record='G04', value=stage, path=capture)['admitted'] is False
    with pytest.raises(ValueError, match='unexpected/missing request export') as error:
        sealed_contract_probe('routing-column-dependency', path=capture)
    assert type(error.value).__name__ == 'ProtocolError'
