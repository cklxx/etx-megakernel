"""Protocol fixtures are synthetic; no device correctness or timing evidence."""
import copy
import json
from pathlib import Path

import pytest

from examples.experiment_program.protocol import (Election, paired_decision,
    pair_order, verify_files, qualify_exposure, capture_coverage)


def samples():
    return [dict(ordinal=i, order=order, control_ns=100000, candidate_ns=98000-i,
                 correct=True, instrumented=False) for i, order in enumerate(pair_order(19))]


def test_complete_pair_and_layer_rules():
    assert paired_decision(samples(), seed=19, qwen_layer=True)['admitted']
    s=samples();s[0]['candidate_ns']=100000
    assert not paired_decision(s, seed=19)['admitted']
    s=samples();s[0]['candidate_ns']=99000
    assert not paired_decision(s, seed=19)['admitted']


@pytest.mark.parametrize('fault', ['truncated','duplicate','wrongorder','instrumented','nan','negative','correctness'])
def test_bad_population_fails_closed(fault):
    s=samples()
    if fault=='truncated':s.pop()
    if fault=='duplicate':s[-1]['ordinal']=0
    if fault=='wrongorder':s[0]['order']='invalid'
    if fault=='instrumented':s[0]['instrumented']=True
    if fault=='nan':s[0]['control_ns']=float('nan')
    if fault=='negative':s[0]['candidate_ns']=-1
    if fault=='correctness':s[0]['correct']=False
    with pytest.raises(ValueError):paired_decision(s,seed=19)


def test_bound_input_hash(tmp_path):
    p=tmp_path/'input';p.write_bytes(b'native-capture-not-generated')
    import hashlib
    seal={str(p):hashlib.sha256(p.read_bytes()).hexdigest()}
    verify_files(seal)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):verify_files(seal)
    p.unlink()
    with pytest.raises(ValueError):verify_files(seal)
    with pytest.raises(ValueError):verify_files({})


def test_election_all_waves_norm_fanout_epochs():
    e=Election();e.begin(1,quiescent=True)
    for owner in range(128):
        for wave in range(16):e.drain(owner,wave,epoch=1)
        e.arrive(owner,epoch=1,norm_ready=owner==0)
    e.publish(epoch=1,replicas=range(8),acquire=True)
    for r in range(8):e.read(r,epoch=1,acquire=True)
    assert e.complete
    e.begin(2,quiescent=True)
    with pytest.raises(ValueError):e.arrive(0,epoch=1)
    with pytest.raises(ValueError):e.arrive(0,epoch=2)
    with pytest.raises(ValueError):e.begin(3,quiescent=False)


def test_election_refuses_early_norm_and_duplicate():
    e=Election();e.begin(1,quiescent=True)
    with pytest.raises(ValueError):e.publish(epoch=1,replicas=range(8),acquire=True)
    for w in range(16):e.drain(0,w,epoch=1)
    with pytest.raises(ValueError):e.arrive(0,epoch=1,norm_ready=False)
    e.arrive(0,epoch=1,norm_ready=True)
    with pytest.raises(ValueError):e.arrive(0,epoch=1,norm_ready=True)


def test_exposure_qualification_is_not_optimization():
    r=qualify_exposure(expected={(0,0),(0,1)}, observed=[(0,0),(0,1)],
                       allocations=[(80,0,0,30000)]*3, perturbation_ns=500, exact=True)
    assert r['qualified'] and r['performance_claim'] is False
    for kwargs in [dict(perturbation_ns=501),dict(observed=[(0,0)]),dict(exact=False),
                   dict(allocations=[(80,0,0,30000),(81,0,0,30000),(80,0,0,30000)]),
                   dict(observed=[(0,0),(0,0),(0,1)])]:
        args=dict(expected={(0,0),(0,1)},observed=[(0,0),(0,1)],allocations=[(80,0,0,30000)]*3,perturbation_ns=500,exact=True)
        args.update(kwargs)
        with pytest.raises(ValueError):qualify_exposure(**args)


def test_g04_capture_only_complete_indices():
    expected={(r,t,l,s) for r in range(2) for t in range(32) for l in range(1,27) for s in range(6)}
    rows=list(expected)
    assert capture_coverage(rows,batch=2)['slots']==2*32*26*6
    with pytest.raises(ValueError):capture_coverage(rows[:-1],batch=2)
    with pytest.raises(ValueError):capture_coverage(rows+[rows[0]],batch=2)
    with pytest.raises(ValueError):capture_coverage(rows,batch=4)


def test_explicit_staging_relocation_preserves_origin_seals(tmp_path,monkeypatch):
    import json
    from examples.experiment_program.protocol import resolve_path,sha,verify_files
    stage=tmp_path/'staged';stage.mkdir();data=stage/'native.hsaco';data.write_bytes(b'Explicit synthetic byte-identity relocation test, not a native module.')
    digest=sha(data);mapping=tmp_path/'paths.json'
    mapping.write_text(json.dumps({'schema':'etx.path-map.v1','paths':{'/origin/private-build':str(stage)}}))
    monkeypatch.setenv('ETX_PROGRAM_RELOCATIONS',str(mapping))
    assert resolve_path('/origin/private-build/native.hsaco')==data
    verify_files({'/origin/private-build/native.hsaco':digest})
    assert resolve_path('/origin/private-build-other/native.hsaco')==Path('/origin/private-build-other/native.hsaco')
    data.write_bytes(b'changed')
    with pytest.raises(ValueError):verify_files({'/origin/private-build/native.hsaco':digest})
    for invalid in ({'schema':'unknown','paths':{}},{'schema':'etx.path-map.v1','paths':{'relative':str(stage)}}):
        mapping.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):resolve_path('/origin/private-build/native.hsaco')
