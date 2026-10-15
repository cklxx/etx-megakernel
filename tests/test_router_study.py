"""CPU guards; synthetic numerics are not model validation."""
import gzip
import numpy as np
import pytest
from examples.vllm_llm import router_study as s
from etx.ops import topk_key32 as tk


def test_bf_round_ties_even_and_reject_nonfinite():
    x=np.array([1+1/256,1+3/256,-1-1/256],dtype=np.float32)
    assert s.bf(x).tolist()==[1,1+2/128,-1]
    with pytest.raises(ValueError): s.bf([np.nan])


def test_topk_matches_existing_cpu_oracle():
    rng=np.random.default_rng(44)
    for _ in range(12):
        l=s.bf(rng.normal(size=128))
        ids,w=s.routing(l)
        ri,rw=tk.key_topk(tk.bf16_bits(l),8,True)
        assert ids.tolist()==ri
        assert np.array_equal(w,np.array(rw,dtype=np.float32))


def test_ties_and_temperature():
    ids,w=s.routing(np.zeros(128,dtype=np.float32))
    assert ids.tolist()==list(range(8))
    assert w.tolist()==[.125]*8
    l=s.bf(np.linspace(-2,2,128))
    a,aw=s.routing(l); b,bw=s.routing(l*2)
    assert np.array_equal(a,b)
    assert not np.array_equal(aw,bw)  # same IDs do not imply same mixing


def test_original_proposal_is_not_bf16_identity():
    w=s.numerical(samples=1)['counterexample']
    assert w['comparison']['reference_ids']==list(range(8))
    assert w['comparison']['candidate_ids']==list(range(7))+[8]
    assert w['reference_logits'][7]==1.515625
    assert w['candidate_logits'][7]==1.5078125


def test_residual_rounding_and_gamma():
    o=np.full(2048,1,dtype=np.float32); r=np.full(2048,1/256,dtype=np.float32)
    g=np.full(2048,2,dtype=np.float32)
    z,scale,x=s.norm(o,r,g)
    assert np.all(z==1)
    assert np.all(x==2)
    assert 0<scale<1
    with pytest.raises(ValueError): s.norm(o[:8],r,g)


def test_real_number_identity():
    rng=np.random.default_rng(6)
    w=rng.normal(size=(128,2048)); z=rng.normal(size=2048); g=rng.uniform(.5,1.5,2048)
    scale=1/np.sqrt(np.mean(z*z)+1e-6)
    np.testing.assert_allclose(w@(z*scale*g),(w@(z*g))*scale,rtol=1e-12,atol=1e-12)


def test_norm_order_and_zero():
    z,scale,x=s.norm(np.zeros(2048),np.zeros(2048),np.ones(2048))
    assert np.all(z==0) and np.all(x==0)
    assert float(scale)==pytest.approx(1000,rel=1e-7)
    a=np.zeros(256,dtype=np.float32); a[:3]=[1e8,1,-1e8]
    assert s.ordered_sum(a)!=s.ordered_sum(a[::-1]) or s.ordered_sum(a)!=s.ordered_sum(a[[0,2,1]+list(range(3,256))])


def test_trace_rejects_bad_count():
    raw=gzip.decompress((s.DEFAULT/'tr_off.bin.gz').read_bytes())
    log=(s.DEFAULT/'tr_off.log').read_text()
    with pytest.raises(ValueError): s.trace_windows(raw[:-1],log)
    with pytest.raises(ValueError): s.trace_windows(raw,log.replace('o_proj_0                  256','o_proj_0                  255'))


def test_historical_trace_partition():
    raw=gzip.decompress((s.DEFAULT/'tr_off.bin.gz').read_bytes())
    w=s.trace_windows(raw,(s.DEFAULT/'tr_off.log').read_text())
    assert w['task_count']==60362 and len(w['layers'])==48
    assert w['span_us']==4303.54
    for r in w['layers']:
        parts=sum(r[k] for k in ('o_router_gap_us','router_envelope_us','router_topk_gap_us','topk_envelope_us','topk_moe_gap_us'))
        assert parts==pytest.approx(r['window_to_first_moe_us'])
        assert max(r['o_done_relative_us'])==0 and len(r['o_done_relative_us'])==256


def test_normalized_input_is_required_on_critical_path():
    assert s.finish_times([-4,0],1,1,1,1,10,1)==13
    assert s.finish_times([-4,0],1,1,1,1,0,1)==5
    with pytest.raises(ValueError): s.finish_times([0],-1,1,1,1,1,1)


def test_current_gain_refuses_cross_build_transfer():
    with pytest.raises(ValueError,match='identity'):
        s.costs({'raw_sha256':'historical-trace','layers':[{'window_to_first_moe_us':20,'o_done_relative_us':[0]}]},timing_identity='latest-kept')
    assert s.transfer_gain(100,'same','same')==100


def test_publication_guards():
    for seed in range(20):
        assert s.publication(seed)['reads']==256
        with pytest.raises(ValueError,match='visibility'): s.publication(seed,release=False)
        with pytest.raises(ValueError,match='visibility'): s.publication(seed,acquire=False)
        with pytest.raises(ValueError,match='epoch'): s.publication(seed,epoch_match=False)


def test_serial_tail_and_retained_outputs():
    assert s.finish_times([-4,0],1,1,1,1,10,1,norm_parallel=False)==15
    a=s.audit_routing(s.DEFAULT/'routing-originals.json.gz')
    assert a['files']==384 and a['layer_position_cases']==96
    assert a['replica_mismatches']==0 and a['id_cases_different']==0 and a['weight_cases_different']==0


def test_sparse_witness_survives_rsqrt_ulp_perturbation():
    # Independent scalar construction, no long-dot or norm reduction assumptions.
    scale=np.float32(1/np.sqrt(2056/2048+1e-6))
    for ulps in (-16,0,16):
        sv=np.asarray(int(scale.view(np.uint32))+ulps,dtype=np.uint32).view(np.float32)
        base7=s.bf(np.float32(.50390625)*s.bf(np.float32(3)*sv))
        base8=s.bf(np.float32(1.515625)*s.bf(sv))
        new7=s.bf(np.float32(np.float32(.50390625*3)*sv))
        new8=s.bf(np.float32(np.float32(1.515625)*sv))
        assert base7==base8 and new7<new8
