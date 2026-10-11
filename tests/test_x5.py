import json
import pytest
from examples.vllm_llm import x5


def samples(value=3.8):
    return '\n'.join('X5-SAMPLE '+json.dumps(dict(layer=l,role=r,sample=n,position=1023,context=1024,
        replays=100,bytes_equal=True,original_us=4.0,imported_us=value))
        for l in range(48) for r in ('attn','reduce') for n in range(3))


def test_x5_registered_bar_and_no_partial_population():
    p=x5.prices(samples());assert p['pass_bar'];assert p['R2_ms']==pytest.approx(48*(10.645-7.6)/1000)
    assert not x5.prices(samples(4.0))['pass_bar']
    for text in (samples().rsplit('\n',1)[0], samples().replace('"bytes_equal": true','"bytes_equal": false',1),
                 samples().replace('"position": 1023','"position": 1040',1),
                 samples().replace('"context": 1024','"context": 1025',1),
                 samples().replace('"imported_us": 3.8','"imported_us": NaN',1)):
        with pytest.raises(ValueError): x5.prices(text)


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
    monkeypatch.setattr(h,'tranche_a',lambda s:h.first_stop(s,dict(G_ms=.100)))
    def probe(s):
        calls.append('X5');s.rows['A2.route']=dict(value=dict(attainable_cut_ms=.236));return {}
    monkeypatch.setattr(x5,'run',probe)
    monkeypatch.setattr(h,'tranche_b',lambda s:calls.append('B') or [])
    monkeypatch.setattr(h,'tranche_c',lambda s:(calls.append('C') or [],0))
    monkeypatch.setattr(h,'tranche_d',lambda *args:calls.append('D'))
    h.execute(s)
    assert calls==['X5','B','C','D'];assert s.rows['B.stop']['status']=='PASS'
