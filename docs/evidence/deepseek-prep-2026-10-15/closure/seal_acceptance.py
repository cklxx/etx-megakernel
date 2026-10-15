import hashlib,json,subprocess
from pathlib import Path
base=Path(__file__).parent;root=Path('/Users/ckl/code/.wt/etx-deepseek-two-gaps')
final=json.loads((base/'full-suite-final.json').read_text());archive=json.loads((base/'archive-after-receipt.json').read_text())
assert final['exit']==0 and final['stop'] is None
assert archive['exit']==0 and archive['stop'] is None
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==final['revision']
for n,h in final['source_sha256'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
old=json.loads((base.parent/'deepseek-open-prep-evidence/archive-acceptance.json').read_text())['source_sha256']
for n,h in old.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
changed=subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines()
assert changed==['examples/vllm_llm/results/2026-10-15-producer-prep/local-ready.json'],changed
commands=json.loads((base/'commands.json').read_text())+[json.loads((base/'proc-regression-before.json').read_text()),json.loads((base/'proc-regression-after.json').read_text()),final,archive]
a=dict(status='PASS_LOCAL_ONLY',revision=final['revision'],initial_preparation_pin='1a23336e1bdf1c78c63ec35f3dab136f88acc2b0',implementation='a0c3e724dbd1a342d96812d232f04ced902d6496',process_cleanup_fix=final['revision'],commands=commands,first_full_suite=dict(exit=1,result='1 failed, 2165 passed, 14 skipped in 512.73s',log='full-suite.log'),final_full_suite=dict(exit=0,result=(base/'full-suite-final.log').read_text().strip().splitlines()[-1],log='full-suite-final.log'),compiler_evidence='Reused unchanged DeepSeek and Qwen kernel/host-generation inputs, with exact original evidence/source hash checks; changed process supervisor, optional profiler wrapper and tests have fresh CPU acceptance. No compiler rerun.',archive_adaptation='fixture.json; real committed source fixture excludes historical DeepSeek GPU data for only the two Qwen archive transport cases',qwen_source_binding='Refreshed readiness sources; exact committed-fixture source match and both archive transport tests passed',independent_review='PENDING unless separate final report and explicit bounded-recheck verdict exist; missing is not pass',gpu_spend_usd=0,gpu_execution=False)
(base/'acceptance.json').write_text(json.dumps(a,indent=2)+'\n');print(a['final_full_suite']['result'])
