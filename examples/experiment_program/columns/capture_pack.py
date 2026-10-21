"""Pack exact source-bound native rows into resident replay operands.

Consumes original native exports and original checkpoint bytes. No model, dot,
normalization, routing or precision conversion is recomputed. The producer seal
must name all raw exports, the complete original log and compiled source files.
"""
from __future__ import annotations
import argparse,json,shutil,struct
from pathlib import Path
from examples.experiment_program.protocol import resolve_path
from .protocol import require,sha256,REVISIONS,DAG,stable_groups
ROW_BYTES=18944
MAGIC=0x455458434f4c3031

def decode_rows(raw,*,family):
    require(raw and len(raw)%ROW_BYTES==0,'native row extent truncated/empty')
    result=[]
    for off in range(0,len(raw),ROW_BYTES):
        h=struct.unpack_from('<64Q',raw,off)
        require(h[0]==MAGIC and h[1]>0 and h[1]==h[23] and h[25]==0 and h[22]==(4 if family=='fleet' else 16),'native row incomplete/duplicate/stale')
        require(0<h[12]<=2048 and h[10]<=h[11] and h[31]>0 and h[21]<=256,'native shape/timing/clock invalid')
        B=1 if family=='fleet' else h[32]
        require(B in (1,2) and B*h[12]<=4096,'native staged input extent')
        require(h[20] in (2,4) and h[40]<16,'native dtype/physical placement invalid')
        result.append(dict(h=h,inputs=raw[off+512:off+512+B*h[12]*4],
                           indices=struct.unpack_from('<256I',raw,off+16896)[:h[21]],
                           outputs=struct.unpack_from('<256I',raw,off+17920)[:h[21]]))
    return result

def original_values(rows,N,columns=1):
    values={}
    for row in rows:
        for index,bits in zip(row['indices'],row['outputs']):
            require(0<=index<N*columns and index not in values,'native output overlap/outside original task slice')
            values[index]=bits
    require(set(values)==set(range(N*columns)),'native output coverage incomplete')
    element=rows[0]['h'][20]
    require(all(row['h'][20]==element for row in rows),'native output dtype differs')
    return struct.pack('<'+('I' if element==4 else 'H')*len(values),*(values[i] for i in range(len(values))))

def staged(rows,*,family,B=1):
    first=rows[0]['inputs'];require(all(row['inputs']==first for row in rows),'native staged operands differ across original tasks')
    if family=='fleet':return first
    bits=struct.unpack('<'+'I'*(len(first)//4),first)
    require(all((v&65535)==0 for v in bits),'native staged value is not exact bf16 expansion')
    return struct.pack('<'+'H'*len(bits),*(v>>16 for v in bits))

def verify_seal(seal):
    require(seal.get('schema')=='etx.columns.native-producer.v1','native producer seal schema')
    for key in ('raw_files','source_files','input_files','build_files'):
        files=seal.get(key,{});require(files,'missing producer '+key)
        for name,digest in files.items():require(resolve_path(name).is_file() and sha256(name)==digest,'changed producer '+key+': '+name)
    log=resolve_path(seal['log']);require(sha256(log)==seal.get('log_sha256'),'native producer log seal mismatch')
    text=log.read_text()
    require('CAPTURE COMPLETE' in text,'native original capture did not finish')
    require(seal.get('full_greedy_tokens')==32 and seal.get('per_request_greedy')==[32]*seal['batch'],'original per-request 32/32 gate absent')
    # Assertions in a seal are not a substitute for actual original stdout.
    if seal['family']=='fleet':
        import re
        actual=re.findall(r'BATCHREQ r=(\d+) tokens=(\d+)/(\d+) layers=(\d+) warmup=(\d+)/(\d+)',text)
        require(sorted(tuple(map(int,row)) for row in actual)==[(r,32,32,27,32,32) for r in range(seal['batch'])],'original raw per-request greedy/hidden/warmup evidence incomplete')
    else:
        require(text.count('32/32 generated tokens match HF greedy')>=seal['batch'],'historical original per-request greedy evidence absent')
    return seal

def task_identity(row,family):
    h=row['h']
    return list((h[2],h[6],h[7],h[8]) if family=='fleet' else (h[35],h[36]))

def replay_rows(rows,seal):
    if seal['record']=='G02' and seal['role']!='qkv':
        x=seal.get('shared_xcd');require(x in (6,7),'exact native shared XCD required')
        selected=[r for r in rows if r['h'][7]==x and r['h'][18]>=(1<<63)]
        require(len(selected)==37*seal['batch'],'selected shared task coverage')
        return selected
    return rows

def request_ready(rows,seal):
    """NB2 jointly stages columns; its request readiness comes from matched NB1."""
    source=rows
    if seal['family']=='dsl' and all(r['h'][32]==2 for r in rows):
        meta=seal.get('readiness_producer',{});path=resolve_path(meta.get('path','/nonexistent'))
        require(path.is_file() and sha256(path)==meta.get('sha256'),'NB2 needs unchanged matched original NB1 readiness producer')
        prior=verify_seal(json.loads(path.read_text()))
        require(all(prior.get(k)==seal.get(k) for k in ('record','family','batch','role','decode_position','input_files')),'NB1 readiness model/input context differs')
        require(bool(seal.get('session_id')) and prior.get('session_id')==seal['session_id'],'NB1 readiness must share the original capture session')
        hardware=seal.get('hardware',{});require(hardware and prior.get('hardware')==hardware,'NB1 readiness hardware identity differs')
        require(sha256(hardware['path'])==hardware['sha256'],'readiness hardware proof changed')
        source=[r for name in prior['raw_files'] for r in decode_rows(resolve_path(name).read_bytes(),family='dsl')]
        require(len(source)==512 and all(r['h'][32]==1 for r in source),'readiness requires complete original NB1 task population')
        require({r['h'][31] for r in source}=={r['h'][31] for r in rows},'readiness native clock frequency differs')
    ready=[]
    for request in range(seal['batch']):
        rr=[r for r in source if r['h'][2]==request];require(rr,'original request readiness absent')
        ready.append(min(r['h'][10] for r in rr))
    return ready

def pack(seal_path,out):
    seal=verify_seal(json.loads(resolve_path(seal_path).read_text()));family=seal['family'];record=seal['record'];B=seal['batch'];role=seal['role']
    require(record in ('G02','G03','G05','G13') and B in (2,4),'registered native target required')
    rows=[]
    for name in seal['raw_files']:rows.extend(decode_rows(resolve_path(name).read_bytes(),family=family))
    require(rows and all(r['h'][4]==seal['decode_position'] for r in rows),'native position differs')
    positions={r['h'][7]:set() for r in rows}
    for row in rows:positions[row['h'][7]].add(row['h'][40])
    require(set(positions)==set(range(8)) and all(len(v)==1 for v in positions.values()) and len({next(iter(v)) for v in positions.values()})==8,'native logical/physical XCD binding unqualified')
    if record=='G05':
        from .runner import verify_dependency
        g04=seal.get('g04',{});gp=resolve_path(g04.get('path','/nonexistent'))
        require(gp.is_file() and sha256(gp)==g04.get('sha256'),'G04 actual native observation seal missing')
        receipt=verify_dependency(gp);require(receipt.get('record')=='G04' and receipt.get('batch')==B,'G04 source batch mismatch')
        routing=json.loads(gp.read_text())
        for identity_role in ('weights','inputs','prefixes','goldens','golden_producers'):
            require(all(seal['input_files'].get(p)==h for p,h in routing['roles'][identity_role].items()),'G04 original input/prefix/golden identity differs')
        picks={(p['request'],p['step'],p['layer'],p['slot']):(p['expert'],p['mix_bits']) for p in routing['slot_ids']}
        for row in rows:
            h=row['h']
            if h[18]<6:require(picks.get((h[2],h[3],h[5],h[18]))==(h[17],h[42]),'native expert/mix does not match actual G04 consumed slot')
    identities=[(r['h'][2],r['h'][6],r['h'][7],r['h'][8]) if family=='fleet' else (r['h'][35],r['h'][36]) for r in rows]
    require(len(set(identities))==len(rows) and len(rows)==seal.get('native_task_count'),'native task coverage/duplicate mismatch')
    complete_rows=rows;selected_rows=replay_rows(rows,seal);perrequest=request_ready(rows,seal)
    out=Path(out);require(not out.exists(),'refusing existing native pack');out.mkdir(parents=True)
    blob=resolve_path(seal['weights']);require(seal['weights'] in seal['input_files'],'original checkpoint not source sealed')
    segments=[];payload=out/'payload.bin'
    # Preflight the complete payload estimate; writes never eat the free reserve.
    expected=seal.get('payload_upper_bound_bytes');require(type(expected) is int and expected>0,'native payload write bound required')
    require(shutil.disk_usage(out).free>=expected+2*(1<<30),'native payload storage reserve2GiB')
    with payload.open('wb') as sink,blob.open('rb') as source:
        def put(name,data,dtype):
            while sink.tell()%4:sink.write(b'\0')
            segments.append(dict(name=name,offset=sink.tell(),bytes=len(data),dtype=dtype));sink.write(data)
        def copy(name,offset,n,dtype='bf16'):
            require(0<=offset and offset+n<=blob.stat().st_size,'native weight extent outside original checkpoint')
            while sink.tell()%4:sink.write(b'\0')
            segments.append(dict(name=name,offset=sink.tell(),bytes=n,dtype=dtype));source.seek(offset)
            remaining=n
            while remaining:
                chunk=source.read(min(1<<20,remaining));require(chunk,'original checkpoint truncated');sink.write(chunk);remaining-=len(chunk)
        common=dict(schema=1,record=record,family=family,source_revision=REVISIONS[family],per_column_dag=DAG[family],input_provenance='native-capture',source_bound=True,placement_bound=True,capture_complete=True,precision='bf16',context=1024,positions=[1024,1040] if family=='fleet' else [0,1040],decode_position=seal['decode_position'],batch=B,role=role,workers=37 if family=='fleet' else 38,xcds=8,threads=256 if family=='fleet' else 1024,epoch=rows[0]['h'][1],source_path=seal['source_path'],source_sha256=sha256(seal['source_path']),native_producer_seal=str(Path(seal_path).resolve()),native_producer_sha256=sha256(seal_path),resource_admitted=False,relocation_seal=__import__('examples.experiment_program.protocol',fromlist=['relocation_seal']).relocation_seal())
        inputs=[];outputs=[];picks=[];mix=[];up_outputs=[];down_inputs=[]
        for request in range(1 if family=='dsl' and all(r['h'][32]==2 for r in rows) else B):
            rr=[r for r in rows if r['h'][2]==request]
            # Dense NB2 has one native task containing both request columns.
            if family=='dsl' and all(r['h'][32]==2 for r in rows):rr=rows
            require(rr,'native request absent')
            if record in ('G02','G03'):
                if family=='fleet' and role!='qkv':
                    selected=seal.get('shared_xcd');require(selected in (6,7),'choose exact native shared expert XCD6/7')
                    rr=[r for r in rr if r['h'][7]==selected];require(len(rr)==37 and all(r['h'][18]>=(1<<63) for r in rr),'native shared expert unit missing')
                inputs.append(staged(rr,family=family));outputs.append(original_values(rr,rr[0]['h'][13],rr[0]['h'][32] if family=='dsl' else 1))
            elif record=='G05':
                ids=[];wm=[]
                for slot in range(6):
                    sr=[r for r in rr if r['h'][18]==slot];require(len(sr)==37,'native routed slot task coverage')
                    require(len({(r['h'][17],r['h'][19],r['h'][30]) for r in sr})==1,'native expert/mix/weight mismatch')
                    ids.append(sr[0]['h'][17]);wm.append(sr[0]['h'][19]);inputs.append(staged(sr,family=family));outputs.append(original_values(sr,sr[0]['h'][13]))
                picks.append(ids);mix.extend(wm)
            else:
                require(role=='expert_up_down' and B==2,'G13 exact paired expert role/B2 required')
                ids=[];wm=[];request_inputs=[]
                for slot in range(8):
                    up=[r for r in rr if r['h'][6]==2 and r['h'][18]==slot]
                    down=[r for r in rr if r['h'][6]==3 and r['h'][18]==slot]
                    require(len(up)==24 and len(down)==32,'G13 original 192up/256down tasks per request required')
                    require(len({(r['h'][17],r['h'][19]) for r in up+down})==1,'G13 native expert/mix differs across endpoints')
                    ids.append(up[0]['h'][17]);wm.append(up[0]['h'][19]);request_inputs.append(staged(up,family=family))
                    ug=original_values(up,768);native_down=staged(down,family=family)
                    require(native_down==ug,'G13 native downstream operand differs from original up output')
                    up_outputs.append(ug);outputs.append(original_values(down,2048))
                require(all(v==request_inputs[0] for v in request_inputs),'G13 native up operands differ across slots')
                inputs.append(request_inputs[0]);picks.append(ids);mix.extend(wm)

        if record in ('G02','G03'):
            rows=selected_rows
            N=rows[0]['h'][13];K=rows[0]['h'][12];ld=rows[0]['h'][14]
            require(len({r['h'][30] for r in rows})==1,'dense requests do not share original weight storage')
            copy('weights',rows[0]['h'][30],N*ld*(4 if role=='gate_up' else 2))
            put('inputs',inputs[0] if family=='dsl' and rows[0]['h'][32]==2 else b''.join(inputs),'bf16' if family=='dsl' else 'fp32')
            golden=outputs[0] if family=='dsl' and rows[0]['h'][32]==2 else b''.join(outputs)
            common.update(native_xcd=seal.get('shared_xcd',6),N=N,K=K,ld=ld,kva_layout='K1_SHARED' if family=='fleet' else None)
        elif record=='G13':
            up=[r for r in rows if r['h'][6]==2];down=[r for r in rows if r['h'][6]==3]
            stride=2*768*2048;dstride=2048*768
            require(all(r['h'][41]==stride for r in up) and all(r['h'][41]==dstride for r in down),'G13 original expert weight strides changed')
            ub={r['h'][30]-r['h'][17]*stride*2 for r in up};db={r['h'][30]-r['h'][17]*dstride*2 for r in down}
            require(len(ub)==len(db)==1,'G13 original expert weight checkpoint binding differs')
            copy('weights',ub.pop(),128*stride*2);copy('down_weights',db.pop(),128*dstride*2)
            put('inputs',b''.join(inputs),'bf16');put('ids',struct.pack('<'+'i'*(B*8),*(v for row in picks for v in row)),'int32')
            put('golden_up',b''.join(up_outputs),'bf16');golden=b''.join(outputs)
            common.update(N=2048,K=2048,ld=2048,slots=8,expert_count=128,expert_stride=stride,down_stride=dstride,historical_identity='NEW_CAPTURE')
        else:
            r=next(r for r in rows if r['h'][18]<6);N=r['h'][13];K=r['h'][12];ld=r['h'][14];stride=3*1408*2048;experts=64
            roleoffset=0 if role=='gate_up' else 2*1408*2048
            bases={rr['h'][30]-(rr['h'][17]*stride+roleoffset)*2 for rr in rows if rr['h'][18]<6}
            require(len(bases)==1,'native expert checkpoint base mismatch');base=bases.pop()
            # Preserve expert IDs/strides: include role-specific prefix so the
            # driver points at native expert e without synthetic renumbering.
            copy('weights',base+roleoffset*2,experts*stride*2-roleoffset*2)
            put('inputs',b''.join(inputs),'fp32');golden=b''.join(outputs)
            put('mix',struct.pack('<'+'I'*len(mix),*mix),'fp32');put('golden_slots',golden,'fp32' if role=='gate_up' else 'bf16')
            common.update(N=N,K=K,ld=ld,picks=picks,slots=6,expert_count=experts,expert_stride=stride,input_layout='request-slot-K',real_mix=True,native_pick_coverage=True)
            require(any(not g['singleton'] for g in stable_groups(picks)),'no real duplicate group captured')
        scales=struct.pack('<'+'I'*len(mix),*mix) if record in ('G05','G13') else struct.pack('<'+'f'*B,*([1.]*B))
        put('scales',scales,'fp32');put('golden',golden,'bf16' if family!='fleet' or role=='down' else 'fp32')
        require(sink.tell()<=expected,'native payload exceeded predeclared storage bound')
    common.update(arrivals_ticks=[v-min(perrequest) for v in perrequest],segments=segments,golden_sha256=__import__('hashlib').sha256(golden).hexdigest(),payload=dict(path=str(payload.resolve()),bytes=payload.stat().st_size,sha256=sha256(payload)))
    certificate=dict(source_revision=common['source_revision'],source_sha256=common['source_sha256'],decode_position=common['decode_position'],per_column_dag=common['per_column_dag'],payload_sha256=common['payload']['sha256'],golden_sha256=common['golden_sha256'],all_task_ids_once=True,all_writing_waves_drained=True,placement_qualified=True,producer_seal_sha256=sha256(seal_path),native_task_count=len(complete_rows),selected_task_count=len(rows),selected_task_ids=[task_identity(r,family) for r in rows],selected_role=role,selected_native_xcd=common.get('native_xcd'),physical_xcds={str(k):list(v) for k,v in positions.items()})
    cert=out/'native-certificate.json';cert.write_text(json.dumps(certificate,indent=2)+'\n');common['native_capture_certificate']=str(cert.resolve())
    (out/'capture.json').write_text(json.dumps(common,indent=2)+'\n');return common

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();print(json.dumps(pack(a.receipt,a.out),indent=2))
if __name__=='__main__':main()
