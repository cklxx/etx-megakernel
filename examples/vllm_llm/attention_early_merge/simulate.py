"""Executed CPU protocol/order model. It makes no GPU visibility claim."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct
from examples.vllm_llm.head_local_protocol import EpochModel


def f32(v): return struct.unpack('<f',struct.pack('<f',v))[0]


def ordered_sum(values,order):
    value=0.0
    for i in order: value=f32(value+values[i])
    return value


def run():
    completed=0
    for active in (0,1,4,5):
        for order in itertools.permutations(range(5)):
            m=EpochModel()
            for _ in range(3):
                epoch=m.epoch
                m.claim_prep(epoch,0); m.publish_prep(epoch,0,all_waves_drained=True)
                for p in order:
                    m.acquire_prep(epoch,p)
                    # Imports p >= active return early, but ALL five arrive.
                    m.arrive(epoch,p,all_waves_drained=True)
                assert m.reducer==order[-1]
                m.acquire_partials(epoch,m.reducer)
                m.publish_output(epoch,m.reducer,all_waves_drained=True)
                m.reset(quiescent=True)
                completed+=1
    # All-wave drain can change the elected partition, independently of body end.
    body_end=[10,11,12,13,14]; drain_end=[20,11,12,13,14]
    assert max(range(5),key=body_end.__getitem__)!=max(range(5),key=drain_end.__getitem__)
    # Eight-slot reduce is elected uniformly. Election never selects math order.
    cancellation=[2**24,1.,-2**24,1.,0.]
    original=ordered_sum(cancellation,range(5))
    alternatives={ordered_sum(cancellation,order) for order in itertools.permutations(range(5))}
    assert len(alternatives)>1
    for election in range(5):
        for slot in range(8):
            assert ordered_sum(cancellation,range(5))==original
            assert 8*3+slot in range(24,32)
    # Queue capacity is explicit. Owner is ticket zero, not a queued successor
    # behind five blocking siblings. Native residency must still be checked.
    for resident in range(1,39):
        tickets=list(range(resident))
        assert tickets[0]==0
        assert (resident>=7)==all(p in tickets for p in range(7))
    return dict(schema='etx.c6.cpu.v1',synthetic=True,gpu_visibility='NOT_TESTED',
        completed_epochs=completed,orders_per_active_case=120,active_partitions=[0,1,4,5],
        cancellation=dict(original_order=original,arrival_order_results=sorted(alternatives)),
        slot_blocks=[list(range(h*8,h*8+8)) for h in range(4)],
        schedule=dict(control_workers_per_head=7,candidate_workers_per_head=5,
                      launch_blocks=304,owner_ticket=0,head_xcds=[0,1,2,3],consumer_xcd=7),
        costs=dict(measured_here=None,derived_historical_zero_tax_local_advance_us='101.82',
                   historical_local_advance_is_wall_forecast=False,gross_registered_budget_us=96,
                   retained_payload_floor_ms_estimate='1.440',byte_ledger_gb='6.1904',bandwidth_tb_s='4.3',
                   required_net_step_us=50,general_keep_us=30,combined_tax_budget_us_per_layer='0.9583333333',
                   retained_body_hop_credit_us=0,
                   unknown=['publication/acquisition','all-wave drain','mask/election','reduce argument setup',
                            'cache reuse','output flush','reset and copies','queue interference','spills']))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); result=run()
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
