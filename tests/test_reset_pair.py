import copy
import pytest
from examples.vllm_llm.reset_pair import check_pair


def identities():
    b = dict(overrides={'ETX_DEV_RESET': '0', 'ETX_VL_TOUCH': '1'}, hipcc_flags='same',
             files={'imports': 'i', 'import_metadata': 'm', 'device_source/dsl_tiles.hip': 'd', 'device_source/imp_tiles.hip': 'a'})
    c = copy.deepcopy(b); c['overrides']['ETX_DEV_RESET'] = '1'
    p = dict(domains=8, workers_per_domain=38, events=[], ev_inst=[], task_in=[], descs=[])
    return b, c, p


def test_reset_pair_only_registered_switch():
    b, c, p = identities(); check_pair(b, c, p, p)
    c['overrides']['ETX_VL_TOUCH'] = '0'
    with pytest.raises(ValueError, match='flags other'): check_pair(b, c, p, p)


def test_reset_pair_refuses_math_and_schedule_changes():
    b, c, p = identities(); c['files']['imports'] = 'new'
    with pytest.raises(ValueError, match='body/import'): check_pair(b, c, p, p)
    b, c, p = identities(); q = dict(p, workers_per_domain=37)
    with pytest.raises(ValueError, match='schedule'): check_pair(b, c, p, q)


def test_runtime_fallback_cannot_become_keep(tmp_path, monkeypatch):
    from examples.sessions import qwen300_hour as h
    from test_qwen300_hour import host_text
    s=h.Session(dict(manifest=str(h.MANIFEST),builds={'kept':'/unused','reset':'/unused'},model='/unused'),tmp_path)
    monkeypatch.setattr(h,'certify',lambda *args:None)
    monkeypatch.setattr(h,'host',lambda *args:host_text(cost=4.0)+'HOUR-RESET enabled=0 checked=1 valid=0\n')
    with pytest.raises(ValueError,match='reset tail not confirmed'):
        h.candidate_pairs(s,dict(id='DEV_RESET',build='reset'),'RESET')


def test_kernel_envelope_reports_without_new_timing_events():
    from examples.sessions import qwen300_hour as h
    import json
    text='\n'.join('HOUR-KERNEL '+json.dumps(dict(run=0,step=t,ms=4.1)) for t in range(1024,1055))
    assert h.kernel_mean(text)==4.1
    with pytest.raises(ValueError,match='population'):h.kernel_mean(text.rsplit('\n',1)[0])


def test_reset_measurement_precedes_s1_even_if_stamps_are_missing(tmp_path,monkeypatch):
    from examples.sessions import qwen300_hour as h
    s=h.Session(dict(manifest=str(h.MANIFEST),builds={'kept':'/unused'},model='/unused'),tmp_path)
    calls=[]
    def certify(s,arm):
        if arm=='kept': s.certified.add(arm)
        else: raise ValueError('missing stamps/stages')
    monkeypatch.setattr(h,'certify',certify)
    monkeypatch.setattr(h,'registered_reset',lambda s:calls.append('reset'))
    stop=h.first_stop
    def first_stop(s,g): calls.append('S1');return stop(s,g)
    monkeypatch.setattr(h,'first_stop',first_stop)
    assert not h.tranche_a(s)
    assert calls==['reset','S1']
    assert s.rows['A.stop']['status']=='NOT-COMPLETE'
