from pathlib import Path
import ast, subprocess
p=Path('examples/vllm_llm/import_certificate.py'); original=p.read_text(); lines=original.splitlines(True)
log=[]
try:
 for node in ast.walk(ast.parse(original)):
  if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call) or getattr(node.exc.func,'id','')!='Refusal':continue
  changed=lines[:]; changed[node.lineno-1]=' '*node.col_offset+'pass # mutation: remove refusal\n'
  for i in range(node.lineno,node.end_lineno):changed[i]=''
  p.write_text(''.join(changed))
  r=subprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_import_certificate.py'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  label=f'line {node.lineno}: {ast.unparse(node.exc)}; exit={r.returncode}'
  print(label,flush=True); log.append(label+'\n'+r.stdout)
finally:p.write_text(original)
Path('results_chkimpl_stage2_2026-10-09/guard_mutations.txt').write_text('\n'.join(log))
