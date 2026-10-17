"""Independent local execution of the actual GLM HIP body and strict input gates."""
import ctypes
from copy import deepcopy
import json
from pathlib import Path
import struct
import pytest

from examples.glm53 import stage,model,model_build,cpu_check
from etx.passes import compile_graph,verify_plan
from etx.ir import Scope


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    pytest.importorskip('torch');pytest.importorskip('transformers')
    out=tmp_path_factory.mktemp('glm-native')/'stage'
    spec,hf,head=cpu_check.fixture(out)
    return out,spec,hf,head


@pytest.mark.parametrize('batch',[1,2,4,8])
def test_actual_model_plan_has_all_layers_rank_ownership_and_system_boundary(native,batch):
    _,s,_,_=native
    g=model.build_from(s,batch);p=compile_graph(g,'gfx942',{'B':batch},{})
    assert not verify_plan(p) and p.n_devices==2
    assert len(p.tasks)==7*batch
    assert [g.device for g in g.grids if g.name.startswith('l')]==[0,0,1,1]
    assert p.events['e_l1'].scope==p.events['e_head'].scope==Scope.SYSTEM
    assert p.tensor_placement['l1_out']==p.tensor_placement['tokens']=='fine_grained'
    assert g.grids[0].body.symbol=='glm53_embed' and g.grids[-2].body.symbol=='glm53_head'
    assert all(g.body.source=='examples/glm53/tiles.hip' for g in g.grids)


@pytest.mark.parametrize('batch,bf16,fp8',[(1,False,False),(2,False,False),(4,True,False),(8,True,True)])
def test_same_hip_body_full_hybrid_model_32_exact_tokens_and_states(tmp_path,batch,bf16,fp8):
    pytest.importorskip('torch');pytest.importorskip('transformers')
    result=cpu_check.check(tmp_path/'fixture',batch,32,bf16,fp8)
    assert result['matched']==batch*32 and result['status']=='PASS'
    assert result['fixture'] and not result['full_checkpoint']


def test_staging_rejects_fixtures_and_changed_blob(native,tmp_path):
    import shutil
    out,s,_,_=native
    with pytest.raises(ValueError,match='synthetic fixtures'):stage.validate(out)
    dest=tmp_path/'copied';shutil.copytree(out,dest)
    p=dest/s['weights']['l0']['blob'];p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError,match='changed/unsafe|out-of-bounds'):stage.validate(dest,True)


def test_descriptor_out_of_bounds_fails_even_with_updated_hash(native,tmp_path):
    import shutil
    out,s,_,_=native;dest=tmp_path/'copied';shutil.copytree(out,dest)
    r=deepcopy(s);w=r['weights']['l0'];meta=dest/w['meta'];data=bytearray(meta.read_bytes())
    struct.pack_into('<Q',data,0,2**40);meta.write_bytes(data);w['meta_sha256']=stage.file_sha(meta)
    (dest/'stage.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='inconsistent|out-of-bounds'):stage.validate(dest,True)


def test_unique_score_kpool_selection_beyond_budget_matches_independent_hf(tmp_path):
    import numpy as np
    import torch
    from transformers.cache_utils import DynamicCache
    torch.set_num_threads(1)
    out=tmp_path/'s';s,hf,_=cpu_check.fixture(out,selection=True);m=cpu_check.CPUModel(out);d=s['dimensions']
    cache=DynamicCache(config=hf.config);attention=hf.layers[1].self_attn
    state=np.zeros(d['STATE_MLA'],np.float32);tmp=np.zeros(d['SCRATCH'],np.float32);blob,meta=m.weights['l1']
    ptr=lambda x:ctypes.c_void_p(x.ctypes.data)
    rng=np.random.default_rng(14);selected=[]
    hook=attention.indexer.register_forward_hook(lambda module,args,result:selected.append(result.detach().cpu().numpy()))
    with torch.inference_mode():
        for pos in range(16):
            x=rng.uniform(.2,2,(8,)).astype(np.float32);y=np.zeros(8,np.float32)
            ref=attention(torch.from_numpy(x)[None,None],attention_mask=torch.ones(1,1,dtype=torch.bool),past_key_values=cache)[0][0,0].numpy()
            m.lib.run_mla(ptr(blob),ptr(meta),ptr(x),ptr(y),ptr(state),ptr(tmp),pos)
            assert np.allclose(y,ref,atol=2e-6,rtol=2e-5),(pos,y,ref)
            # Native chosen flags match the exact independently selected raw IDs.
            start=d['QR']+4*d['NH']*d['AD']+d['KVR']+d['IH']*d['ID']+2*d['ID']+d['IH']+d['MAXLEN']
            chosen=set(np.flatnonzero(tmp[start:start+pos+1]));wanted=set(int(i) for i in selected[-1].ravel() if i>=0)
            assert chosen==wanted
    hook.remove()
    assert len(wanted)==2 and pos+1>d['INDEX_TOPK']


def test_clipped_swiglu_matches_hf_at_saturation_boundaries(tmp_path):
    import numpy as np
    import torch
    out=tmp_path/'s';s,hf,_=cpu_check.fixture(out)
    with torch.no_grad():
        hf.layers[0].mlp.gate_proj.weight.mul_(1000);hf.layers[0].mlp.up_proj.weight.mul_(1000)
    # Re-stage that layer after deliberately driving both clamp boundaries.
    old=s['weights']['l0'];(out/old['blob']).unlink();(out/old['meta']).unlink()
    s['weights']['l0']=stage.write_blob(out,'l0',stage.shapes(s['config'],0),lambda k:(hf.layers[0].state_dict()[k],None),stage.slot_names(s['config']))
    (out/'stage.json').write_text(json.dumps(s));m=cpu_check.CPUModel(out);d=s['dimensions']
    x=np.arange(8,dtype=np.float32)-4;y=np.zeros(8,np.float32);scratch=np.zeros(d['SCRATCH'],np.float32)
    b,w=m.weights['l0'];ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
    m.lib.run_mlp(ptr(b),ptr(w),ptr(x),ptr(y),ptr(scratch))
    with torch.inference_mode():ref=hf.layers[0].mlp(torch.from_numpy(x)).numpy()
    assert np.allclose(y,ref,atol=1e-5,rtol=1e-5)


def test_model_session_has_real_model_and_two_visible_devices():
    from etx.tools.gpu_jobs import load_session
    jobs,_=load_session(Path('examples/sessions/glm53_model.py'))
    job={j.name:j for j in jobs}
    assert [j.name for j in jobs]==['apt','etx_env','model_build','model_qualify','verify']
    assert job['etx_env'].reads==['apt'] and job['model_build'].reads==['etx_env']
    assert job['model_qualify'].env['HIP_VISIBLE_DEVICES']=='0,1'
    assert 'examples.glm53.run' in job['model_qualify'].cmd and 'qualify' in job['model_qualify'].cmd
    assert job['model_qualify'].reads==['model/run']
    assert job['verify'].lane=='final' and 'tools/two_gpu_verify.py' in job['verify'].cmd
    assert job['verify'].reads==['?qualification/qualification.json']


def test_fixture_cannot_enter_hardware_qualification(native,tmp_path):
    from examples.glm53.run import qualify
    out,_,_,_=native;result=tmp_path/'result'
    with pytest.raises(ValueError,match='synthetic fixtures'):
        qualify(out,tmp_path/'build',tmp_path/'reference.json',result)
    receipt=json.loads((result/'qualification.json').read_text())
    assert receipt['status']=='NOT-COMPLETE' and not receipt['model_certified']


def test_build_identity_refuses_source_drift_and_local_binary(native,tmp_path):
    out,s,_,_=native
    build=tmp_path/'build';model_build.generate(s,build)
    model_build.finalize(build,local=True)
    with pytest.raises(ValueError,match='linked model binary'):model_build.verified_build(build)
    (build/'run').write_text('synthetic, never executes a GPU')
    model_build.finalize(build,local=False)
    assert model_build.verified_build(build)['status']=='GPU-LINKED'
    (build/'glm53_spec.h').write_text('changed')
    with pytest.raises(ValueError,match='identity'):model_build.verified_build(build)


def test_exact_production_shape_has_45_layers_and_pp24_boundary():
    s=model_build.production_compile_spec();p=compile_graph(model.build_from(s,8),'gfx942',{'B':8},{})
    assert s['fixture'] and s['compile_only']
    assert len(p.tasks)==48*8 and p.events['e_l23'].scope==Scope.SYSTEM
    assert p.tensor_placement['l23_out']=='fine_grained'
    d=s['dimensions']
    assert d['H']==4096 and d['EXPERTS']==288 and d['TOPK']==8
    assert d['INDEX_TOPK']==2048 and d['POOL']==4
    assert [(r['rank'],r['layers'][0],r['layers'][-1]) for r in s['ranks']]==[(0,0,23),(1,24,44)]
    assert [r['moe_layers'] for r in s['ranks']]==[21,21]
    assert [r['kda_layers'] for r in s['ranks']]==[18,16]
    assert all(len(r['routed_experts'])==288 for r in s['ranks'])


def test_production_schedule_numerics_cover_288_experts_topk8_and_45_layers(tmp_path):
    """The same native body at production layer/expert/top-k schedule, small widths."""
    pytest.importorskip('torch');pytest.importorskip('transformers')
    result=cpu_check.check(tmp_path/'production',2,32,production_schedule=True)
    assert result['status']=='PASS' and result['matched']==64
    assert result['layers']==45 and result['experts']==288 and result['topk']==8
    assert result['production_schedule'] and not result['full_checkpoint']
    spec=json.loads((tmp_path/'production/stage.json').read_text())
    assert [(r['rank'],r['layers'][0],r['layers'][-1]) for r in spec['ranks']]==[(0,0,23),(1,24,44)]


def test_snapshot_reader_accepts_hf_symlink_shards_and_rejects_path_traversal(tmp_path):
    import torch
    from safetensors.torch import save_file
    src=tmp_path/'snapshot';src.mkdir();blobs=tmp_path/'blobs';blobs.mkdir()
    target=blobs/'weights';save_file({'model.embed_tokens.weight':torch.ones(3,4)},target)
    (src/'model-00001.safetensors').symlink_to(target)
    index=src/'model.safetensors.index.json';index.write_text(json.dumps({'weight_map':{'model.embed_tokens.weight':'model-00001.safetensors'}}))
    r=stage.Snapshot(src);value,scale=r('model.embed_tokens.weight')
    assert value.shape==(3,4) and scale is None
    assert r.verified_inputs()['shards']
    index.write_text(json.dumps({'weight_map':{'model.embed_tokens.weight':'../blobs/weights'}}))
    with pytest.raises(ValueError,match='unsafe checkpoint'):stage.Snapshot(src)('model.embed_tokens.weight')


@pytest.mark.parametrize('bf16,fp8',[(False,False),(True,True)])
def test_hf_chunk_prefill_then_32_exact_native_greedy_tokens(tmp_path,bf16,fp8):
    result=cpu_check.check_prefill(tmp_path/'prefill',2,8,bf16,fp8)
    assert result['matched']==64 and result['status']=='PASS'


def test_logit_comparison_reports_error_that_leaves_argmax_unchanged():
    import numpy as np
    from examples.glm53.run import compare_logits
    expected=np.zeros((2,1,4),np.float32);got=expected.copy();got[0,0,0]=0.5
    assert np.array_equal(got.argmax(-1),expected.argmax(-1))     # tokens match
    report=compare_logits(expected,got)
    assert report['logit_gate']=='REPORTED' and report['logit_max_abs']==0.5
    assert report['logit_argmax_equal'] and not report['logit_bit_equal']
    assert report['logit_max_abs_per_step']==[0.5,0.0]
    # The same mismatch fails an explicit gate, even though the tokens match.
    with pytest.raises(ValueError,match='exceeds'):
        compare_logits(expected,got,0.25)
    assert compare_logits(expected,got,0.5)['logit_gate']=='ENFORCED'


def test_logit_comparison_is_exact_and_rejects_invalid_input():
    import numpy as np
    from examples.glm53.run import compare_logits
    logits=np.arange(12,dtype=np.float32).reshape(1,3,4)
    report=compare_logits(logits,logits.copy(),0.0)
    assert report['logit_bit_equal'] and report['logit_max_abs']==0.0 and report['logit_gate']=='ENFORCED'
    with pytest.raises(ValueError,match='shape'):
        compare_logits(logits,logits[:,:2])
    for atol in (-1.0,float('nan'),float('inf'),'0.1'):
        with pytest.raises(ValueError,match='finite and non-negative'):
            compare_logits(logits,logits.copy(),atol)


def test_bf16_conv_output_rounds_before_silu(tmp_path):
    # Also support BF16 conv tensors, rather than silently treating them like
    # the production FP32 conv. This caught a free-greedy mismatch at step 22.
    result=cpu_check.check_prefill(tmp_path/'bf16-conv',2,8,True,True,fp32_conv=False)
    assert result['matched']==64
