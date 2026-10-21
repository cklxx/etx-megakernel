"""Create checking-host capture source and seal independently produced snapshots."""
import json
from pathlib import Path
from examples.experiment_program.protocol import resolve_path
from examples.experiment_program.protocol import sha
from .replay import sha_slice


def checking_host(original,out):
    s=resolve_path(original).read_text(); anchor='        launch_imported(x.n + 1);'
    if s.count(anchor)!=1: raise ValueError('native host source anchor drift')
    start='''        RowCapture row_capture;
        if (const char* directory=getenv("ETX_ROW_CAPTURE_DIR")) {
          const std::string requested=getenv("ETX_ROW_CAPTURE_ROLE")?getenv("ETX_ROW_CAPTURE_ROLE"):"dsl_gemv_fold";
          const std::string role=x.role=="moe_up"?"dsl_moe_up":x.role=="moe_down"?"dsl_moe_down":"dsl_gemv_fold";
          if(x.req==0 && role==requested && (x.role=="qkv" || x.role=="moe_up" || x.role=="moe_down") && x.layer==atoi((getenv("ETX_ROW_CAPTURE_LAYER")?getenv("ETX_ROW_CAPTURE_LAYER"):"-1"))) {
            auto weight=m.t.at("L"+std::to_string(x.layer)+(x.role=="qkv"?".wqkv":x.role=="moe_up"?".w13":".w2"));
            row_capture.before(directory,ab.data()+x.off,dir+"/vl.bin",weight.first,weight.second*2,x.layer,check_pos,0,x.tasks/x.rep,x.tasks,role);
          }
        }
'''
    s=s.replace(anchor,start+anchor)
    anchor='        if (!wait_stream(hst.stream, 5.0)) die("DSL tile hung: " + x.exp);'
    if s.count(anchor)!=1:raise ValueError('native original wait anchor')
    s=s.replace(anchor,anchor+'\n        row_capture.after();')
    anchor='static void die(const std::string& m) { fprintf(stderr, "error: %s\\n", m.c_str()); exit(1); }'
    if s.count(anchor)!=1:raise ValueError('original die anchor')
    s=s.replace(anchor,anchor+'\n#include "examples/experiment_program/row_ops/capture.h"')
    resolve_path(out).write_text(s)


def finalize(directory,plan,*,producer,trace_proof=None,capture_build=None):
    d=resolve_path(directory);p=json.loads((d/'capture.json').read_text())
    p['dependency_edges']=plan_edges(plan,p['role'],p['layer'])
    if len(p['dependency_edges']['tasks'])!=p['count']:raise ValueError('captured task population differs from emitted plan')
    p['producer_certificate']=dict(path=str(resolve_path(producer)),sha256=sha(producer))
    from .producer import verify
    verify(p['producer_certificate'],p)
    p['arguments']['sha256']=sha(p['arguments']['file'])
    for b in p['buffers']:b['sha256']=sha_slice(b['file'],b['offset'],b['bytes'])
    for r in p['references']:r['sha256']=sha(r['file'])
    if trace_proof is not None:
        if capture_build is None:raise ValueError('exact native capture build required for trace binding')
        from examples.experiment_program.validation_trace import verify,bind_g11
        proof=resolve_path(trace_proof);meta=dict(proof=str(proof),proof_files={str(proof):sha(proof)})
        trace=verify(meta);p.update(matching_trace=meta,g12_session_id=trace['session_id'],capture_native_build=str(resolve_path(capture_build)),capture_native_identity_sha256=sha(resolve_path(capture_build)/'build-identity.json'))
        p['trace_binding']=bind_g11(p,meta)
    (d/'pack.json').write_text(json.dumps(p,indent=2)+'\n');return p


def adapt_moe(pack,out,*,producer):
    """Reuse existing original bf16_300 capture; never regenerate model values."""
    import struct
    p=json.loads(resolve_path(pack).read_text());block=resolve_path(p['arguments']).read_bytes()
    b=p['buffers'];buffers=[dict(file=x['file'],offset=x['offset'],bytes=x['bytes'],sha256=x.get('slice_sha256',x.get('sha256'))) for x in b]
    ptrs=[dict(offset=i*8,buffer=i,add=0) for i in range(5) if struct.unpack_from('<Q',block,i*8)[0]]
    x=dict(schema='etx-row-replay-v1',baseline_revision='b573664faa4e7988dff353bc03aca4ce8561a729',position=p['position'],layer=p['layer'],role='dsl_'+p['role'],first=p['first'],count=p['count'],total_tasks=p['total_tasks'],arguments=dict(file=p['arguments'],sha256=p['arguments_sha256']),buffers=buffers,pointers=ptrs,reference_provenance='unchanged-original-host',references=[dict(buffer=2,offset=p['reference']['output_offset'],file=p['reference']['file'],bytes=p['reference']['bytes'],sha256=p['reference']['sha256'])])
    x['producer_certificate']=dict(path=str(resolve_path(producer)),sha256=sha(producer))
    from .producer import verify
    original=verify(x['producer_certificate'],x)
    if resolve_path(original.get('original_pack','/missing'))!=resolve_path(pack):raise ValueError('historical certificate names a different original pack')
    resolve_path(out).write_text(json.dumps(x,indent=2)+'\n');return x


def plan_edges(plan,role,layer):
    """Recover exact emitted task/event/queue predecessors without GPU execution."""
    p=json.loads(resolve_path(plan).read_text());base={'dsl_gemv_fold':'qkv','dsl_moe_up':'moe_up','dsl_moe_down':'moe_down'}[role]
    names=[n for n in p['types'] if n in (f'{base}_{layer}',f'{base}_{layer}_x')]
    if len(names)!=1:raise ValueError('selected native grid absent/ambiguous in actual emitted plan')
    tid=p['types'][names[0]]['id'];tasks=[d for d in p['descs'] if d['type']==tid]
    if not tasks:raise ValueError('empty native selected grid')
    previous={}
    for worker,queue in p['static_queues'].items():
        for i,t in enumerate(queue):previous[t]=dict(worker=worker,previous=queue[i-1] if i else None)
    incoming=set();rows=[]
    for d in tasks:
        ins=p['task_in'][d['id']];incoming.update(ins)
        if d['id'] not in previous:raise ValueError('selected task lacks static queue predecessor')
        rows.append(dict(task_id=d['id'],coord=d['coord'],domain=d['domain'],worker=d['worker'],queue=previous[d['id']],incoming_event_instances=ins))
    events={str(i):dict(name=p['ev_inst'][i][0],coord=p['ev_inst'][i][1],producers=p['ev_inst'][i][2]) for i in sorted(incoming)}
    if not events or any(not e['producers'] for e in events.values()):raise ValueError('missing actual native event producers')
    return dict(plan_file=str(resolve_path(plan).resolve()),plan_sha256=sha(plan),grid=names[0],tasks=rows,events=events)
