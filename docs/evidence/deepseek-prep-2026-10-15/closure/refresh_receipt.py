import fcntl,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
base=Path(__file__).parent;root=Path('/Users/ckl/code/.wt/etx-deepseek-two-gaps');sys.path.insert(0,str(root))
from examples.vllm_llm import producer_session as ps
lock=open(base.parent/'qwen-prep-heavy.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
final=json.loads((base/'full-suite-final.json').read_text());assert final['exit']==0 and final['stop'] is None
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==final['revision']
path=ps.ART/'local-ready.json';old=json.loads(path.read_text());current=ps.local_sources();shutil.copy2(path,base/'qwen-local-ready-before.json')
changed={n:dict(old=h,new=current.get(n)) for n,h in old['sources'].items() if current.get(n)!=h};added={n:h for n,h in current.items() if n not in old['sources']}
assert set(changed)=={'etx/session/proc.py','examples/llm/vllm_bench.py','tests/test_session_proc.py'},changed
assert set(added)=={'tests/test_deepseek_open_prep.py','tests/test_dsv2_batch_attribution.py'},added
reuse=dict(base='24eabf5932a9a320f3703f2774f88d7f8b743ed8',unchanged_source_count=len(old['sources'])-len(changed),changed=changed,added=added,decision='Reuse the prior producer/attention/expert compiler evidence only for unchanged device/host generation inputs. Changed files are the process supervisor, optional batch profiler markers, and tests; no kernel, C++ host, emitter, graph, import or compiler flag input changed. Shared runtime and wrappers are covered by the final full suite and bounded archive transport tests.',previous_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_evidence='Existing Qwen compiler/artifact logs and receipts remain unchanged; no new compiler pass or GPU result is claimed.')
(base/'qwen-compile-reuse.json').write_text(json.dumps(reuse,indent=2)+'\n')
old['sources']=current
old['reused_compile_evidence']['deepseek_closure']=dict(comparison='docs/evidence/deepseek-prep-2026-10-15/closure/qwen-compile-reuse.json',scope=reuse['decision'])
old['deepseek_closure_evidence']='docs/evidence/deepseek-prep-2026-10-15/closure/'
old['deepseek_archive_acceptance']='docs/evidence/deepseek-prep-2026-10-15/closure/archive-after-receipt.json'
old['commands'].append(final|dict(log='docs/evidence/deepseek-prep-2026-10-15/closure/full-suite-final.log'))
old['limitations'].append('DeepSeek integration refresh reuses unchanged compiler inputs; shared Python runtime/optional profiler wrapper changes have local CPU coverage only. Independent final review and the bounded process-cleanup recheck are still required before publication.')
path.write_text(json.dumps(old,indent=2)+'\n');assert json.loads(path.read_text())['sources']==ps.local_sources()
fixture=base/'source-fixture';relative=path.relative_to(root);shutil.copy2(path,fixture/relative)
subprocess.run(['git','add',str(relative)],cwd=fixture,check=True)
subprocess.run(['git','-c','user.name=Local source acceptance','-c','user.email=fixture@localhost','commit','-qm','Refresh readiness binding for the final combined source'],cwd=fixture,check=True)
fpath=base/'fixture.json';f=json.loads(fpath.read_text());f['files'][str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest();f['fixture_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=fixture,text=True).strip();fpath.write_text(json.dumps(f,indent=2)+'\n')
probe=subprocess.run([root/'.venv/bin/python','-c','import json; from examples.vllm_llm import producer_session as p; r=json.loads((p.ART/"local-ready.json").read_text()); assert r["ready"] is True and r["sources"]==p.local_sources(); print("BOUNDED COMMITTED SOURCE RECEIPT MATCH",len(r["sources"]))'],cwd=fixture,env=dict(os.environ,PYTHONPATH=str(fixture)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(base/'qwen-receipt-probe.log').write_text(probe.stdout);assert probe.returncode==0,probe.stdout
print(probe.stdout,flush=True)
