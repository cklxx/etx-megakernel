import json
import pytest
from examples.vllm_llm import x5


def samples(value=3.8):
    return '\n'.join('X5-SAMPLE '+json.dumps(dict(layer=l,role=r,sample=n,position=1023,context=1024,
        replays=100,bytes_equal=True,original_us=4.0,imported_us=value))
        for l in range(48) for r in ('attn','reduce') for n in range(3))


def test_x5_registered_bar_and_no_partial_population():
    p=x5.prices(samples(),candidate=True);assert p['pass_bar'];assert p['R2_ms']==pytest.approx(48*(10.645-7.6)/1000)
    assert not x5.prices(samples(4.0))['pass_bar']
    for text in (samples().rsplit('\n',1)[0], samples().replace('"bytes_equal": true','"bytes_equal": false',1),
                 samples().replace('"position": 1023','"position": 1040',1),
                 samples().replace('"context": 1024','"context": 1025',1),
                 samples().replace('"imported_us": 3.8','"imported_us": NaN',1)):
        with pytest.raises(ValueError): x5.prices(text)


@pytest.mark.parametrize('candidate,value,credit', [(False, 3.8, 0.), (True, 3.8, 48*(10.645-7.6)/1000), (True, 4., 0.)])
def test_x5_control_never_gets_candidate_credit(candidate, value, credit):
    result = x5.prices(samples(value), candidate=candidate)
    assert result['R2_ms'] == pytest.approx(credit)
    assert x5.prices(samples(value))['R2_ms'] == 0.  # missing candidate evidence fails closed


def build_identity(path, *, kernel='attention', export='attn', grid=1):
    (path/'imp').mkdir(parents=True)
    imports = {export: dict(kernel=kernel, block=[256,1,1], slots=1),
               'reduce': dict(kernel='reduce', block=[256,1,1], slots=1)}
    (path/'imp/imports.json').write_text(json.dumps(imports))
    (path/'vl_plan.txt').write_text('\n'.join(
        f'{2*l+i} {e} {l} {role} {grid if role == "attn" else 32} 1 1 0 0 0 0 0 1'
        for l in range(48) for i,(e,role) in enumerate(((export,'attn'),('reduce','reduce')))))


def test_x5_candidate_identity_is_attention_or_reduce_change(tmp_path):
    kept=tmp_path/'kept'; renamed=tmp_path/'renamed'; alternate=tmp_path/'alternate'
    build_identity(kept);build_identity(renamed,export='renamed');build_identity(alternate,kernel='alternate_attention')
    assert x5.attention_signature(kept) == x5.attention_signature(renamed)
    assert x5.attention_signature(kept) != x5.attention_signature(alternate)
    # A different legal launch also differs; a host/build label cannot do so.
    build_identity_path=tmp_path/'launch';build_identity(build_identity_path,grid=2)
    assert x5.attention_signature(kept) != x5.attention_signature(build_identity_path)
    (renamed/'vl_plan.txt').write_text('')
    with pytest.raises(ValueError,match='incomplete'): x5.attention_signature(renamed)


@pytest.mark.parametrize('builds,value,credit', [([], 3.8, 0.), (['kept'], 3.8, 0.),
    (['alias'], 3.8, 0.), (['candidate'], 3.8, 48*(10.645-7.6)/1000), (['candidate'], 4., 0.),
    (['candidate','kept'], 3.6, 48*(7.6-7.2)/1000)])
def test_x5_session_prices_only_distinct_certified_candidates(tmp_path, monkeypatch, builds, value, credit):
    from examples.sessions import qwen300_hour as h
    from examples.vllm_llm import import_certificate as cert
    paths={arm: tmp_path/arm for arm in ('kept','alias','candidate')}
    for arm,path in paths.items(): build_identity(path,kernel='alternate_attention' if arm=='candidate' else 'attention')
    s=h.Session(dict(manifest=str(h.MANIFEST),builds={a:str(p) for a,p in paths.items()},
                     model='/unused',x5_builds=builds),tmp_path/'session')
    calls=[]
    monkeypatch.setattr(h,'verify_seal',lambda *args: dict(flags={}))
    monkeypatch.setattr(h,'certify',lambda s,arm: s.certified.add(arm))
    def host(s,arm,*args):
        calls.append(arm)
        return samples(3.8 if arm=='kept' else value)+'\nCHECK OK: 0 launch(es) differ\n'
    monkeypatch.setattr(h,'host',host)
    monkeypatch.setattr(cert,'verified_build',lambda *args: {})
    monkeypatch.setattr(cert,'check_output',lambda *args: None)
    result=x5.run(s)
    assert result['R2_ms'] == pytest.approx(credit)
    assert s.rows['A2.route']['value']['attainable_cut_ms'] == pytest.approx(credit)
    if not credit: assert 'B1 stays open' in result['B1']
    if 'kept' in builds:
        assert calls[0]=='kept'
        assert result['results']['kept']['R2_ms']==0.
        assert result['baseline_us_layer']==pytest.approx(7.6)


def test_inventory_does_not_turn_missing_vendor_sources_into_no_winner(tmp_path):
    result=x5.inventory(tmp_path)
    assert not result['complete'];assert result['aiter_sources']=={}
    assert result['survivor_builds']==['kept']
    assert any(c['status'].startswith('[U]') for c in result['candidates'])


def test_session_prices_x5_before_s2_without_funding_body_pairs(tmp_path,monkeypatch):
    from examples.sessions import qwen300_hour as h
    s=h.Session(dict(manifest=str(h.MANIFEST),builds={'kept':'/unused'},model='/unused'),tmp_path)
    s.certified.add('kept');calls=[]
    monkeypatch.setattr(h,'preflight',lambda s:None)
    monkeypatch.setattr(h,'environment_guard',lambda s:True)
    monkeypatch.setattr(h,'qualify_c',lambda s:[])
    monkeypatch.setattr(h,'tranche_a',lambda s:h.first_stop(s,dict(G_ms=.100),h.stamp_perturbation(dict(perturbation=dict(sampled_bodies=48,max_abs_body_change_us=.5)),True)))
    def probe(s):
        calls.append('X5');s.rows['A2.route']=dict(value=dict(attainable_cut_ms=.236));return {}
    monkeypatch.setattr(x5,'run',probe)
    monkeypatch.setattr(h,'tranche_b',lambda s:calls.append('B') or [])
    monkeypatch.setattr(h,'tranche_c',lambda s:(calls.append('C') or [],0))
    monkeypatch.setattr(h,'tranche_d',lambda *args:calls.append('D'))
    h.execute(s)
    assert calls==['X5','B','C','D'];assert s.rows['B.stop']['status']=='PASS'
