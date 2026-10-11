import json
from pathlib import Path
import pytest
from examples.vllm_llm import environment_guard as g


def medians(value=7.5):
    return {name:{'0':{'attn':dict(original_us=value)}} for name in ('old','hour')}


def test_guard_reuses_registered_thresholds_and_does_not_hide_tree_delta():
    pairs=[dict(old=4.18,hour=4.18),dict(old=4.18,hour=4.18)]
    assert g.decision(pairs,medians(),True)['status']=='PASS'
    assert g.decision(pairs,medians(8.5),True)['status']=='SLOW-VM'
    assert g.decision(pairs,medians(),False)['status']=='DATA-MISMATCH'
    assert g.decision([dict(old=4.18,hour=4.22),dict(old=4.22,hour=4.18)],medians(),True)['status']=='TREE-DRIFT'
    assert g.decision([dict(old=4.97,hour=4.98)]*2,medians(),True)['status']=='REBASE-REQUIRED'
    with pytest.raises(ValueError):g.decision(pairs[:1],medians(),True)


def check_text():
    return 'CHECK position 0 partitions 1\n'+ '\n'.join(f'  {r} L{l} kernel identical original 7.5 us imported 9.3 us' for l in range(48) for r in ('attn','reduce'))+'\nCHECK OK: 0 launch(es) differ\n'


def test_original_population_excludes_only_cold_layer_zero():
    r=g.original_medians(check_text(),0);assert r['attn']['count']==47;assert r['attn']['original_us']==7.5
    with pytest.raises(ValueError):g.original_medians(check_text().replace('attn L47','attn L46'),0)
    with pytest.raises(ValueError):g.original_medians(check_text().replace('CHECK OK','CHECK FAILED'),0)


def test_guard_runs_before_optional_qualification_and_every_tranche(tmp_path,monkeypatch):
    from examples.sessions import qwen300_hour as h
    s=h.Session(dict(manifest=str(h.MANIFEST)),tmp_path);calls=[]
    monkeypatch.setattr(h,'preflight',lambda s:calls.append('cpu'))
    monkeypatch.setattr(h,'environment_guard',lambda s:calls.append('guard') or False)
    monkeypatch.setattr(h,'qualify_c',lambda s:pytest.fail('no eligibility/GPU work before passing guard'))
    monkeypatch.setattr(h,'tranche_a',lambda s:pytest.fail('no A on slow/unknown VM'))
    h.execute(s);assert calls==['cpu','guard'];assert s.rows['GUARD']['status']=='NOT-COMPLETE'
    assert h.ENVIRONMENT_GUARD_SECONDS+sum(h.TRANCHE_SECONDS.values())+h.REPORT_SECONDS==8400
