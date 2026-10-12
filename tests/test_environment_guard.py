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


HARDWARE = '''GPU[0] : Card series: AMD Instinct MI300X
GPU[0] : sclk clock level: 0 (1000Mhz)
GPU[0] : mclk clock level: 0 (1300Mhz)
GPU[0] : Average Graphics Package Power (W): 140.0
GPU[0] : Performance Level: auto
'''


@pytest.fixture
def guard_session(tmp_path, monkeypatch):
    from examples.sessions import qwen300_hour as h
    from examples.vllm_llm import import_certificate as cert
    model=tmp_path/'model';model.mkdir();hf=model/'hf';hf.mkdir()
    (model/'vl.bin').write_bytes(b'packed');(hf/'one.safetensors').write_bytes(b'hf')
    b=tmp_path/'kept';b.mkdir();ir=b/'normalized.ll';ir.write_text('device IR')
    hashes={'test':'source'}
    monkeypatch.setattr(h,'source_hashes',lambda:hashes)
    gates=tmp_path/'gates.json'
    gates.write_text(json.dumps(dict.fromkeys(('pytest','gfx942','dry_run','shell','local_matrix_16','rocm_build_check'),True)|dict(source_hashes=hashes)))
    identity=dict(compiler='clang',hipcc_flags=h.BASE_FLAGS,check_enabled=True,overrides={'ETX_DEV_RESET':'0'})
    monkeypatch.setattr(cert,'verified_build',lambda *args:identity)
    seal=dict(source_hashes=hashes,source_commit='a'*40,
              condition=dict(precision='bf16',batch=1,context=1024,tokens=32,key32=True),
              flags={'ETX_TOPK_KEY32':'1','ETX_DEV_RESET':'0'},build_sha256=cert.digest(identity))
    (b/'hour_build.json').write_text(json.dumps(seal))
    c=dict(model=str(model),hf=str(hf),builds={'kept':str(b)},cpu_gates=str(gates),
           build_commits={'kept':'a'*40},build_seals={'kept':cert.sha(b/'hour_build.json')},
           normalized_device_ir=str(ir),normalized_device_ir_sha256=cert.sha(ir),
           weight_sha256={str(p):cert.sha(p) for p in (model/'vl.bin',hf/'one.safetensors')})
    s=h.Session(c,tmp_path/'logs');s.calls=[];s.anchor=4.18;s.attention=7.5;s.missing_anchor=False
    # Only the execution/archive boundaries are mocked; real seals, parsers and decisions run.
    monkeypatch.setattr(cert,'export_build',lambda b,m,a:a.mkdir())
    monkeypatch.setattr(cert,'validate',lambda *args,**kwargs:None)
    def step(name,argv,**kw):
        s.calls.append(name)
        if name=='GUARD_hardware': return HARDWARE
        if name=='GUARD_weights': return '\n'.join(cert.sha(p)+'  '+p for p in sorted(s.c['weight_sha256']))
        if '--check' in argv:
            pos=int(argv[argv.index('--check-pos')+1])
            text=check_text().replace('position 0 ',f'position {pos} ').replace('original 7.5 us',f'original {s.attention} us')
        else:
            text='' if s.missing_anchor else f'ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH=\nmode: ETX megakernel\nrun 0: 32/32 generated tokens match HF greedy\nCHECK-DSL OK\nRESULT model=qwen requests=1 matched=32 gen=32 median_ms=4.000 wall_ms={s.anchor:.3f} matched_each=32\n'
        tag=cert.digest(identity)
        return f'BUILD-IDENTITY phase=before sha256={tag}\n'+text+f'BUILD-IDENTITY phase=after sha256={tag}\n'
    monkeypatch.setattr(s,'step',step)
    return s


@pytest.mark.parametrize('config',['absent','entry-absent','directory-absent'])
@pytest.mark.parametrize('missing_anchor',[False,True])
def test_fresh_box_local_only_and_missing_anchor_refuses(guard_session, config, missing_anchor, capsys):
    s=guard_session
    if config!='absent':
        s.c['environment_guard']=dict(trees={} if config=='entry-absent' else {'old':dict(build=str(s.out/'missing-old')),'hour':dict(build=str(s.out/'missing-hour'))})
    s.missing_anchor=missing_anchor
    assert g.run(s) is (not missing_anchor)
    r=s.rows['GUARD'];assert r['status']==('NOT-COMPLETE' if missing_anchor else 'PASS-LOCAL-ONLY')
    assert 'unavailable:' in r['value']['pinned_trees'] and '[U]' in r['value']['pinned_trees']
    assert 'directory absent' in r['value']['pinned_trees'] if config=='directory-absent' else 'config absent' in r['value']['pinned_trees']
    if missing_anchor: assert r['value']['action'].startswith('STOP')
    else: assert r['value']['historical_need_valid'] and r['value']['wall_anchor_ms']==4.18
    s.finish()
    assert json.loads((s.out/'report.json').read_text())['thresholds']['GUARD']==r
    assert r['value']['hardware']==HARDWARE
    assert 'THRESHOLD GUARD '+r['status'] in capsys.readouterr().out


@pytest.mark.parametrize('anchor,attention,status,valid',[(4.210,7.9,'PASS-LOCAL-ONLY',True),(4.211,7.5,'REBASE-REQUIRED',False),(4.18,8.0,'SLOW-VM',True)])
def test_local_thresholds(guard_session,anchor,attention,status,valid):
    s=guard_session;s.anchor=anchor;s.attention=attention
    assert g.run(s) is (status=='PASS-LOCAL-ONLY')
    assert s.rows['GUARD']['status']==status
    assert s.rows['GUARD']['value']['historical_need_valid'] is valid


@pytest.mark.parametrize('missing',['sclk','mclk','Power','Performance'])
def test_hardware_reading_missing_refuses(missing):
    with pytest.raises(ValueError,match='reading'):
        g.hardware_readings('\n'.join(line for line in HARDWARE.splitlines() if missing not in line))


@pytest.mark.parametrize('bad',['weight','cpu','seal','ir'])
def test_local_inputs_stay_fail_closed(guard_session,bad):
    s=guard_session
    if bad=='weight': s.c['weight_sha256'][str(Path(s.c['model'])/'vl.bin')]='0'*64
    elif bad=='cpu': Path(s.c['cpu_gates']).write_text('{"pytest": 1}')
    elif bad=='seal': s.c['build_seals']['kept']='0'*64
    else: Path(s.c['normalized_device_ir']).unlink()
    assert g.run(s) is False
    assert s.rows['GUARD']['status']==('DATA-MISMATCH' if bad=='weight' else 'NOT-COMPLETE')
    assert not any(c.startswith('GUARD_r') for c in s.calls)
    assert s.rows['GUARD']['value']['action'].startswith('STOP')


@pytest.mark.parametrize('missing_anchor',[False,True])
def test_fully_pinned_guard_keeps_two_adjacent_pairs(guard_session,missing_anchor):
    from examples.sessions import qwen300_hour as h
    from examples.vllm_llm import import_certificate as cert
    s=guard_session;trees={}
    for name,commit in g.TREE_COMMITS.items():
        b=s.out/name;b.mkdir();ir=b/'normalized.ll';ir.write_text('same pinned IR')
        identity=cert.verified_build(b,Path(s.c['model']))
        seal=dict(source_commit=commit+'a'*(40-len(commit)),build_sha256=cert.digest(identity),
                  cpu_gates=dict(pytest=True,gfx942=True,rocm_build_check=True),
                  flags={'ETX_TOPK_KEY32':'1','ETX_VL_CHECK':'1','ETX_DEV_RESET':'0'},normalized_device_ir_sha256=cert.sha(ir))
        (b/'guard_build.json').write_text(json.dumps(seal))
        trees[name]=dict(build=str(b),identity_sha256=cert.digest(identity),seal_sha256=cert.sha(b/'guard_build.json'),normalized_device_ir=str(ir))
    s.c['environment_guard']=dict(trees=trees);s.missing_anchor=missing_anchor
    assert g.run(s) is (not missing_anchor)
    assert s.rows['GUARD']['status']==('NOT-COMPLETE' if missing_anchor else 'PASS')
    if not missing_anchor:
        assert [c for c in s.calls if c.startswith('GUARD_r')]==['GUARD_r0_old','GUARD_r0_hour','GUARD_r1_hour','GUARD_r1_old']
        assert len(s.rows['GUARD']['value']['pairs'])==2
