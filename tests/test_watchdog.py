import json
import shutil
import struct
import subprocess
from pathlib import Path
import pytest
from etx.tools import inbody,watchdog


def test_arrival_lookup_rejects_indices_outside_allocated_csr(tmp_path):
    """Execute the real lookup on CPU, including empty and malformed rows."""
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    root = Path(__file__).resolve().parents[1]
    header = (root/'etx/runtime/include/etx/inbody.h').read_text()
    # Compile the lookup verbatim; the separate gfx942 test covers the timer asm.
    lookup = header[header.index('static __device__ __forceinline__ int etx_arrival_slot'):header.index('#define ETX_LEDGER_BEFORE')]
    source = tmp_path/'ledger.cpp'
    source.write_text(r'''
#define ETX_INBODY 1
#define ETX_PRODUCER_LEDGER 1
#define __device__
#define __forceinline__ inline
#include "etx/abi.h"
#define assert(condition) do { if (!(condition)) return __LINE__; } while (0)
''' + lookup + r'''
int main() {
  int32_t offsets[] = {0, 2, 2}, words[] = {7, 7};
  uint64_t stamps[] = {0, 0, 0, 0};
  etx_params p{};
  p.arrival_offsets = offsets; p.arrival_words = words; p.arrival_stamps = stamps;
  p.arrival_tasks = 2; p.arrival_slots = 2;
  assert(etx_arrival_slot(p, 0, 7) == 0);
  stamps[0] = 100;
  assert(etx_arrival_slot(p, 0, 7) == 1);
  stamps[2] = 110;
  assert(etx_arrival_slot(p, 0, 7) == -1);
  assert(etx_arrival_slot(p, 0, 8) == -1);
  assert(etx_arrival_slot(p, 1, 7) == -1); // empty row
  assert(etx_arrival_slot(p, -1, 7) == -1);
  assert(etx_arrival_slot(p, 2, 7) == -1); // offsets[3] does not exist
  stamps[0] = 0;
  offsets[0] = -1;
  assert(etx_arrival_slot(p, 0, 7) == -1);
  offsets[0] = 1; offsets[1] = 0;
  assert(etx_arrival_slot(p, 0, 7) == -1);
  offsets[0] = 0; offsets[1] = 3; // words[2] and stamps[4] do not exist
  assert(etx_arrival_slot(p, 0, 7) == -1);
  offsets[1] = 2; p.arrival_slots = 0;
  assert(etx_arrival_slot(p, 0, 7) == -1);
  p.arrival_slots = 2; p.arrival_words = nullptr;
  assert(etx_arrival_slot(p, 0, 7) == -1);
}
''')
    exe = tmp_path/'ledger'
    subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(root/'etx/runtime/include'),
                    str(source), '-o', str(exe)], check=True, capture_output=True, timeout=60)
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=60)


def fixture():
    plan=dict(types={'gemv_0':dict(id=0),'consumer_0':dict(id=1)},
        descs=[dict(id=i,type=0 if i<3 else 1,coord=[i],worker=i,domain=0) for i in range(4)],
        events=[dict(name='ready',offset=0,shape=[1])],ev_inst=[['ready',[0],[0,1,2]]],task_in=[[],[],[],[0]])
    raw=dict(version=2,tick_ns=10,waves_per_task=1,legacy=[[100,110,140,0],[105,115,170,1],[130,140,0,2],[0,0,0,0]],
        poll=[[0],[0],[0],[0]],events=[[0,0,0]],waves=[[120,130],[150,160],[0,0],[0,0]],
        producer_ledger=dict(offsets=[0,1,2,3,3],words=[0,0,0],stamps=[[150,160],[180,190],[0,0]],remaining=[1],sub=[[0]]))
    return plan,raw


def binary(raw):
    words=[inbody.MAGIC,raw['version'],len(raw['legacy']),len(raw['events']),raw['waves_per_task'],10]
    for key in ('legacy','poll','events','waves'):words.extend(v for row in raw[key] for v in row)
    if raw['version']==2:
        l=raw['producer_ledger'];words.extend([len(l['words']),len(l['remaining']),len(l['sub'])])
        words.extend(l['offsets']);words.extend(l['words']);words.extend(v for row in l['stamps'] for v in row)
        words.extend(l['remaining']);words.extend(v for row in l['sub'] for v in row)
    return struct.pack('<'+'Q'*len(words),*(v % (1<<64) for v in words))


def test_synthetic_missing_arrival_names_task_and_last_stamp(tmp_path):
    p,r=fixture();f=tmp_path/'snapshot';f.write_bytes(binary(r))
    assert inbody.read_raw(f)==r
    got=watchdog.dump(p,inbody.read_raw(f));event=got['events'][0]
    assert (event['expected'],event['remaining'],event['completed_calls_observed'])==(3,1,2)
    assert event['unobserved_producer_tasks']==[2]
    producer=event['producers'][2]
    assert producer['phase']=='gemv' and producer['observed_worker']==2
    assert producer['last_stamp']==dict(kind='body_entry',ticks=140)
    assert producer['cause'] is None and got['removability_evidence'] is False


def test_in_flight_arrival_is_not_reported_unobserved():
    p,r=fixture();r['producer_ledger']['stamps'][2]=[200,0]
    e=watchdog.dump(p,r)['events'][0]
    assert e['unobserved_producer_tasks']==[]
    assert e['producers'][2]['arrival_status']=='arrival_in_flight_observed'


def test_version_one_cannot_identify_partial_producers():
    p,r=fixture();r['version']=1;r.pop('producer_ledger')
    e=watchdog.dump(p,r)['events'][0]
    assert e['remaining'] is None and e['unobserved_producer_tasks']==[]
    assert all(t['arrival_status']=='unknown_no_producer_ledger' for t in e['producers'])


@pytest.mark.parametrize('mutation',['short','extra','offset','word','bracket','plan'])
def test_corrupt_ledger_fails_closed(tmp_path,mutation):
    p,r=fixture()
    if mutation=='offset':r['producer_ledger']['offsets'][0]=1
    if mutation=='word':r['producer_ledger']['words'][0]=5
    if mutation=='bracket':r['producer_ledger']['stamps'][0]=[160,150]
    if mutation=='plan':p['ev_inst'][0][2]=[0,1]
    data=binary(r)
    if mutation=='short':data=data[:-8]
    if mutation=='extra':data+=b'\0'*8
    f=tmp_path/'raw';f.write_bytes(data)
    with pytest.raises(ValueError):watchdog.dump(p,inbody.read_raw(f))


def test_diagnostic_header_csr_is_exact_original_arrivals():
    from etx.codegen.plan_header import emit_plan_header
    from examples.vllm_llm.bf16_300 import dispatcher
    h=emit_plan_header(dispatcher.graph('/tmp/tiles.hip'))
    assert '#define ETX_ARRIVAL_SLOTS 1' in h
    assert 'etx_arrival_offsets[3] = {0, 1, 1}' in h
    assert 'etx_arrival_words[1] = {0}' in h


def test_runtime_initial_count_is_unknown_and_not_assumed_from_mapping():
    p,r=fixture();p['events'][0].update(runtime_init=True,counts=[0])
    e=watchdog.dump(p,r)['events'][0]
    assert e['expected'] is None and e['runtime_obligation_unknown']
    assert e['unobserved_producer_tasks']==[]


def test_static_initial_count_must_match_producer_multiplicity():
    p,r=fixture();p['events'][0]['counts']=[2]
    with pytest.raises(ValueError,match='initial count'):watchdog.dump(p,r)
