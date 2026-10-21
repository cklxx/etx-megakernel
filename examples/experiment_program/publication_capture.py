"""G11 paired original-checker capture. No model values are synthesized."""
import json,struct,subprocess,tempfile
from pathlib import Path
from .protocol import require,resolve_path,sha,verify_files,selected_artifacts
from .row_ops.replay import sha_slice
BASE='b573664faa4e7988dff353bc03aca4ce8561a729'

def checking_source(original):
    """Only wrap the original standalone DSL checks; keep all launches/checks."""
    s=original;anchor='        launch_imported(x.n + 1);'
    require(s.count(anchor)==1,'original DSL capture boundary drift')
    hook='''        std::vector<RowCapture> g11_captures;
        if(const char* root=getenv("ETX_G11_CAPTURE_DIR")) {
          int layer=atoi(getenv("ETX_G11_CAPTURE_LAYER")?getenv("ETX_G11_CAPTURE_LAYER"):"-1");
          bool down=x.role=="moe_down" && x.layer==layer-1;
          bool fold=x.role=="qkv" && x.layer==layer;
          if(x.req==0 && (down || fold)) {
            if((down && x.rep!=8) || (fold && x.rep!=1) || x.tasks!=256)die("G11 actual current capture shape");
            g11_captures.resize(x.rep);
            auto weight=m.t.at("L"+std::to_string(x.layer)+(down?".w2":".wqkv"));
            for(int rr=0;rr<x.rep;++rr) {
              std::string directory=std::string(root)+(down?"/down":"/fold")+std::to_string(rr);
              g11_captures[rr].before(directory,ab.data()+x.off+rr*x.stride,dir+"/vl.bin",weight.first,weight.second*2,x.layer,check_pos,down?rr*32:0,down?32:256,256,down?"dsl_moe_down":"dsl_gemv_fold");
            }
          }
        }
'''
    s=s.replace(anchor,hook+anchor)
    anchor='        if (!wait_stream(hst.stream, 5.0)) die("DSL tile hung: " + x.exp);';require(s.count(anchor)==1,'original DSL wait drift')
    s=s.replace(anchor,anchor+'\n        for(auto& capture:g11_captures)capture.after();')
    anchor='static void die(const std::string& m) { fprintf(stderr, "error: %s\\n", m.c_str()); exit(1); }';require(s.count(anchor)==1,'original die drift')
    return s.replace(anchor,anchor+'\n#include "examples/experiment_program/row_ops/capture.h"')

def verify_producer(c,layer,position):
    """Recheck original module, generated host, 97 checks and raw exported bytes."""
    require(c['schema']=='etx.g11.original-producer.v1' and c['baseline_revision']==BASE and c['exit']==0,'not an original G11 producer')
    for key in ('files','inputs','exports'):verify_files(c[key])
    require(resolve_path(c['exit_file']).read_text().strip()=='0' and c['files'].get(str(resolve_path(c['exit_file'])))==sha(c['exit_file']),'original checker did not finish')
    build=resolve_path(c['build']);b=json.loads((build/'build-identity.json').read_text());require(c['files'].get(str(build/'build-identity.json'))==sha(build/'build-identity.json'),'original build unbound')
    require(b['host_link_exit']==0 and b['check_enabled'] is True,'original checker must be linked/checking')
    original=subprocess.check_output(['git','show',BASE+':examples/vllm_llm/host.hip'],cwd=Path(__file__).resolve().parents[2],text=True)
    import hashlib
    require(b['host_source_sha256']==hashlib.sha256(checking_source(original).encode()).hexdigest(),'changed original capture host')
    authority=json.loads(Path(__file__).with_name('row_ops').joinpath('original-contract.json').read_text())
    for name in ('model.hsaco','dsl_tiles.hip','plan.json'):require(sha(build/name)==authority[name],'candidate original '+name)
    for key in ('original_reference_objects','dependency_files','runtime_files','reused_imports'):verify_files(b[key])
    require(set(b['reused_imports'].values())=={authority['imports.bc']},'changed original imports')
    verify_files(authority['compiler_files']);selected_artifacts(build,b['hashes'],['run','model.hsaco','dsl_tiles.hip','plan.json'])
    from examples.vllm_llm import import_certificate as cert
    log=resolve_path(c['check_log']);require(c['files'].get(str(log))==sha(log),'original check log unbound')
    fp=build/'build_identity.json';require(c['files'].get(str(fp))==sha(fp),'checker fingerprint unbound')
    expected=cert.expected_instances((build/'vl_plan.txt').read_text(),json.loads((build/'imp/imports.json').read_text()));require(len(expected)==97,'original97 check population')
    cert.check_output(log.read_text(),expected,position,cert.verified_build(build,resolve_path(c['model_dir'])))
    require('CHECK OK: 0 launch(es) differ' in log.read_text(),'incomplete original checker')
    require((c['layer'],c['position'])==(layer,position),'capture producer case differs')
    root=resolve_path(c['capture_directory'])
    for name in [f'down{i}' for i in range(8)]+['fold0']:
        raw=root/name/'capture.json';require(c['exports'].get(str(raw))==sha(raw),'raw capture metadata unbound')
        require(f'ROW-CAPTURE original folded native snapshots={root/name}' in log.read_text(),'capture not exported by checked execution')
    return b

def _assemble(cp,c):
    """Join nine original blocks by actual pointer ranges, retaining live aliases."""
    cp=resolve_path(cp);layer=c['layer'];position=c['position']
    root=resolve_path(c['capture_directory']);pieces=[]
    for i,name in enumerate([f'down{i}' for i in range(8)]+['fold0']):
        p=json.loads((root/name/'capture.json').read_text());block=resolve_path(p['arguments']['file']).read_bytes()
        require(len(block)==160 and p['position']==position and p['layer']==(layer if i==8 else layer-1),'original argument case')
        require((p['first'],p['count'],p['total_tasks'])==((0,256,256) if i==8 else (i*32,32,256)),'actual eight-domain producer/consumer coverage')
        for f in [p['arguments']['file']]+[r['file'] for r in p['references']]:require(c['exports'].get(str(resolve_path(f)))==sha(f),'capture bytes unbound')
        for b in p['buffers']:require(str(resolve_path(b['file'])) in c['inputs'] or str(resolve_path(b['file'])) in c['exports'],'operand source unbound')
        pieces.append((p,block))
    # Resolve every original pointer interval first, then select the earliest
    # pristine snapshot of a containing allocation. Shared down slots are never
    # replaced by the post-down fold input (that would hide a missing producer).
    ranges=[]
    for i,(p,block) in enumerate(pieces):
        for ptr in p['pointers']:
            b=p['buffers'][ptr['buffer']];address=struct.unpack_from('<Q',block,ptr['offset'])[0]-ptr.get('add',0)
            ranges.append((address,address+b['bytes'],i,ptr['buffer'],b))
    groups=[]
    for a,z,i,j,b in sorted(ranges,key=lambda r:(r[0],-r[1],r[2])):
        overlap=[g for g in groups if a<g[1] and z>g[0]]
        if overlap:
            require(len(overlap)==1 and overlap[0][0]<=a and z<=overlap[0][1],'unsupported partial native alias')
        else:groups.append((a,z,i,j,b))
    buffers=[]
    for a,z,_,_,_ in groups:
        candidates=[r for r in ranges if r[0]==a and r[1]==z];_,_,i,j,b=min(candidates,key=lambda r:r[2]);buffers.append(dict(b,sha256=sha_slice(b['file'],b['offset'],b['bytes'])))
    pointers=[];references={}
    for i,(p,block) in enumerate(pieces):
        for ptr in p['pointers']:
            address=struct.unpack_from('<Q',block,ptr['offset'])[0];g=next(k for k,r in enumerate(groups) if r[0]<=address<r[1]);pointers.append(dict(block=i,offset=ptr['offset'],buffer=g,add=address-groups[g][0]))
        for ref in p['references']:
            original=next(x for x in p['pointers'] if x['buffer']==ref['buffer']);address=struct.unpack_from('<Q',block,original['offset'])[0]+ref['offset'];g=next(k for k,r in enumerate(groups) if r[0]<=address and address+ref['bytes']<=r[1]);references[g,address-groups[g][0],ref['bytes']]=dict(ref,buffer=g,offset=address-groups[g][0],sha256=sha(ref['file']))
    # Output references are from the latest original check touching that range.
    args=[dict(file=p['arguments']['file'],sha256=sha(p['arguments']['file'])) for p,_ in pieces]
    p=dict(schema='etx.g11.operands.v1',layer=layer,position=position,producer_certificate=dict(path=str(cp),sha256=sha(cp)),arguments=args,buffers=buffers,pointers=pointers,references=list(references.values()),capture_build=c['build'],files={str(root/name/'capture.json'):sha(root/name/'capture.json') for name in [f'down{i}' for i in range(8)]+['fold0']})
    return p

def assemble(certificate,out):
    cp=resolve_path(certificate);c=json.loads(cp.read_text());verify_producer(c,c['layer'],c['position']);p=_assemble(cp,c);validate(p);resolve_path(out).write_text(json.dumps(p,indent=2)+'\n');return p

def validate(p):
    require(p['schema']=='etx.g11.operands.v1' and p['layer'] in (1,24,47) and p['position'] in (0,1040),'G11 operand context')
    verify_files(p['files']);cp=p['producer_certificate'];require(sha(cp['path'])==cp['sha256'],'producer certificate changed');c=json.loads(resolve_path(cp['path']).read_text());verify_producer(c,p['layer'],p['position'])
    require(p==_assemble(resolve_path(cp['path']),c),'operand pack differs from original exported metadata/bytes/aliases')
    require(len(p['arguments'])==9 and p['references'],'all original argument blocks/references required')
    for a in p['arguments']:require(sha(a['file'])==a['sha256'] and resolve_path(a['file']).stat().st_size==160,'argument bytes changed')
    for b in p['buffers']:require(sha_slice(b['file'],b['offset'],b['bytes'])==b['sha256'],'operand bytes changed')
    for r in p['references']:require(sha(r['file'])==r['sha256'] and resolve_path(r['file']).stat().st_size==r['bytes'] and 0<=r['offset'] and r['offset']+r['bytes']<=p['buffers'][r['buffer']]['bytes'],'reference bytes/bounds changed')
    mapped={(x['block'],x['offset']):x for x in p['pointers']};require(len(mapped)==len(p['pointers']),'duplicate relocation')
    for i,a in enumerate(p['arguments']):
        block=resolve_path(a['file']).read_bytes();offsets=(0,8,16,64,72,80,88) if i==8 else (0,8,16,24,32,112)
        for off in offsets:
            if struct.unpack_from('<Q',block,off)[0]:require((i,off) in mapped,'live native pointer not relocated')
    for x in mapped.values():require(x['block'] in range(9) and x['offset'] in range(0,120,8) and x['buffer'] in range(len(p['buffers'])) and 0<=x['add']<p['buffers'][x['buffer']]['bytes'],'relocation bounds')
    for i in range(8):
        d=mapped[i,16];f=mapped[8,64];require(d['buffer']==f['buffer'] and d['add']==f['add'],'down outputs must feed actual fold slots')
    return p

def job(identity,model,layer,position,out,*,execute=False,seconds=600):
    """Future original native capture with complete checking transcript, no tuning."""
    import os,sys,shutil
    from .publication_run import verify_interface
    p=resolve_path(identity);i=verify_interface(json.loads(p.read_text()));build=resolve_path(i['capture_build']);model=resolve_path(model);out=resolve_path(out)
    require(layer in (1,24,47) and position in (0,1040),'canonical capture case')
    env=dict(os.environ);native=json.loads((build/'build-identity.json').read_text());env.update(native['environment']);env.update(ETX_VL_CHECK='1',ETX_G11_CAPTURE_DIR=str(out/'operands'),ETX_G11_CAPTURE_LAYER=str(layer))
    staged=out/'checker';argv=[sys.executable,'-m','examples.vllm_llm.import_certificate','--run-build',str(staged),'--model',str(model),'--','--check','48','--check-pos',str(position)]
    result=dict(schema='etx.g11.capture-job.v1',layer=layer,position=position,command=argv,environment={k:v for k,v in env.items() if k.startswith('ETX_')},original_build=str(build),original_identity_sha256=sha(build/'build-identity.json'),gpu_executed=False)
    if not execute:return result
    require(not out.exists(),'preserve original capture evidence');require(shutil.disk_usage(out.parent).free>(2<<30)+(8<<20),'original capture budget plus2GiB floor unavailable')
    out.mkdir();shutil.copytree(build,staged,copy_function=os.link)
    # Fingerprint is new session metadata; never overwrite a hard-linked prior file.
    require(not (staged/'build_identity.json').exists(),'capture checker must not contain prior mutable session fingerprint')
    from examples.vllm_llm import import_certificate as cert
    with (out/'fingerprint.log').open('x') as f:
        fp=subprocess.run([sys.executable,'-m','examples.vllm_llm.import_certificate','--write-build',str(staged),'--model',str(model)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
    (out/'fingerprint.exit').write_text(str(fp.returncode)+'\n');require(fp.returncode==0,'original checker fingerprint failed')
    for name in [f'down{j}' for j in range(8)]+['fold0']:(out/'operands'/name).mkdir(parents=True)
    (out/'capture-command.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        with (out/'check.log').open('x') as f:r=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=seconds)
        (out/'check.exit').write_text(str(r.returncode)+'\n');require(r.returncode==0,'original capture checker failed; receipts retained')
    except subprocess.TimeoutExpired:(out/'check.exit').write_text('TIMEOUT\n');raise
    c=dict(schema='etx.g11.original-producer.v1',baseline_revision=BASE,layer=layer,position=position,build=str(staged),model_dir=str(model),capture_directory=str(out/'operands'),exit=0,exit_file=str(out/'check.exit'),check_log=str(out/'check.log'))
    c['files']={str(x):sha(x) for x in [out/'capture-command.json',out/'check.log',out/'check.exit',staged/'build-identity.json',staged/'build_identity.json']};c['inputs']={str(model/n):sha(model/n) for n in ('vl.bin','vl.manifest','golden_c1024.txt')};c['exports']={str(x):sha(x) for x in (out/'operands').rglob('*') if x.is_file()}
    verify_producer(c,layer,position);certificate=out/'producer.json';certificate.write_text(json.dumps(c,indent=2)+'\n');assemble(certificate,out/'pack.json');result.update(gpu_executed=True,pack=str(out/'pack.json'));return result
