"""Execute intentional guard mutants, retaining expected pytest failures.

Only monkeypatches the module in child processes. Never edits production code.
"""
import argparse
from pathlib import Path
import subprocess
import sys

MUTANTS = {
    'cross-build-transfer': ('s.transfer_gain=lambda gain,*args:gain', 'test_current_gain_refuses_cross_build_transfer'),
    'missing-normalized-input': ('old=s.finish_times\ns.finish_times=lambda done,p,v,r,t,n,f: old(done,p,v,r,t,0,f)', 'test_normalized_input_is_required_on_critical_path'),
    'visibility-counter-only': ("s.publication=lambda *a,**kw:{'reads':256}", 'test_publication_guards'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    for name,(patch,test) in MUTANTS.items():
        code='from examples.vllm_llm import router_study as s\n'+patch+'\nimport pytest\nraise SystemExit(pytest.main(["-q", "tests/test_router_study.py::'+test+'"]))'
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=60)
        (a.out/f'mutant-{name}.log').write_text(r.stdout+r.stderr)
        print(f'{name}: pytest exit={r.returncode}, expected=1',flush=True)
        if r.returncode!=1: raise SystemExit('mutant was not killed by an assertion')
    r=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_router_study.py'],capture_output=True,text=True,timeout=60)
    (a.out/'guards-restored.log').write_text(r.stdout+r.stderr)
    print(r.stdout,end="",flush=True)
    raise SystemExit(r.returncode)

if __name__=='__main__': main()
