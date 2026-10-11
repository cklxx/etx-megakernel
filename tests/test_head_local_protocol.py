import itertools

import pytest

from examples.vllm_llm.head_local_protocol import EpochModel, NET_US_PER_LAYER, check_opt_in, specification


def test_every_completion_order_elects_exactly_one_reducer():
    for order in itertools.permutations(range(5)):
        m = EpochModel()
        m.claim_prep(1,0)
        m.publish_prep(1,0,all_waves_drained=True)
        elected=[]
        for p in order:
            m.acquire_prep(1,p)
            if m.arrive(1,p,all_waves_drained=True): elected.append(p)
        assert elected == [order[-1]]
        m.acquire_partials(1,elected[0])
        m.publish_output(1,elected[0],all_waves_drained=True)
        m.reset(quiescent=True)
        assert m.epoch == 2 and m.owner is None and not m.released
        with pytest.raises(ValueError,match='stale'): m.acquire_prep(1,0)


def test_visibility_duplicates_and_reset_fail_closed():
    m=EpochModel()
    with pytest.raises(ValueError): m.acquire_prep(1,0)
    with pytest.raises(ValueError): m.claim_prep(1,1)
    m.claim_prep(1,0)
    with pytest.raises(ValueError): m.claim_prep(1,0)
    with pytest.raises(ValueError): m.publish_prep(1,0,all_waves_drained=False)
    m.publish_prep(1,0,all_waves_drained=True)
    with pytest.raises(ValueError): m.arrive(1,0,all_waves_drained=True)
    for p in range(5):
        m.acquire_prep(1,p)
        with pytest.raises(ValueError): m.arrive(1,p,all_waves_drained=False)
        m.arrive(1,p,all_waves_drained=True)
        with pytest.raises(ValueError): m.arrive(1,p,all_waves_drained=True)
    with pytest.raises(ValueError): m.publish_output(1,4,all_waves_drained=True)
    m.acquire_partials(1,4)
    with pytest.raises(ValueError): m.publish_output(1,4,all_waves_drained=False)
    m.publish_output(1,4,all_waves_drained=True)
    with pytest.raises(ValueError): m.reset(quiescent=False)
    m.epoch=(1<<64)-1
    with pytest.raises(ValueError,match='rollover'): m.reset(quiescent=True)


def test_opt_in_has_no_silent_fallback(monkeypatch):
    monkeypatch.delenv('ETX_VL_HEAD_LOCAL_PROTOCOL',raising=False)
    check_opt_in()
    monkeypatch.setenv('ETX_VL_HEAD_LOCAL_PROTOCOL','1')
    with pytest.raises(SystemExit,match='no validated GPU adapter'): check_opt_in()
    from examples.llm.model import load_dims
    from examples.vllm_llm.model import instances
    with pytest.raises(SystemExit,match='no validated GPU adapter'):
        instances(load_dims('examples/llm/configs/qwen3-30b-a3b.json'))


def test_contract_threshold_and_no_hop_double_credit():
    spec=specification()
    assert NET_US_PER_LAYER == 50/48 and spec['thresholds']['net_us_per_step']==50
    assert not spec['implemented'] and not spec['enabled_by_default']
    assert spec['budget_basis']['hop_credit_in_boundary_budget']==0
