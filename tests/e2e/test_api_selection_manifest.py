"""The host reading the variant selection from an `etx compile` manifest, moved from tests/test_selection_manifest.py.

The MATCH configuration of the round11 build b_r8base (docs/core/PLAN-generic-decoder-2026-10-08.md section 2.2) is
compiled with the command; etx/selection.h is compiled for the CPU and run against the plan file.
"""
import os
from pathlib import Path
import re
import shutil
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
INC = ROOT / "etx/runtime/include"
# Import table of the MATCH build; regenerated from the operators by
# tests/test_selection_manifest.py::test_frozen_import_table_matches_the_operators.
IMPORTS = Path(__file__).parent / "data/selection_imports.json"
MATCH_ENV = {"ETX_VL_XCD": "1", "ETX_VL_CHAIN": "1", "ETX_VL_DSL_GEMV": "1", "ETX_VL_DSL_TOPK": "1",
             "ETX_VL_DSL_ATTN": "head", "ETX_VL_DSL_FOLD": "1", "ETX_VL_TOUCH": "1", "ETX_VL_COH": "1",
             "ETX_VL_CU": "256", "ETX_VL_FOLD_TOPK": "0"}
DRIVER = r'''
#include "plan_data.h"
#include "etx/selection.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  std::ifstream f(argv[1]); std::string line; std::vector<int> runs(ETX_SEL_N_PHASES > 0 ? ETX_SEL_N_PHASES : 1, 0);
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#' || line.rfind("total", 0) == 0) continue;
    std::istringstream s(line); int n, layer; std::string exp, role; s >> n >> exp >> layer >> role;
    if (const char* e = etx_sel_check(role.c_str(), layer, exp.c_str())) { printf("instance %d %s %d %s: %s\n", n, role.c_str(), layer, exp.c_str(), e); return 1; }
    ++runs[etx_sel_find_instance(role.c_str(), layer)];
  }
  for (int i = 0; i < ETX_SEL_N_PHASES; ++i) if (!runs[i]) { printf("phase %s has no instance\n", etx_sel_phase[i]); return 2; }
  printf("ok %d\n", ETX_SEL_N_PHASES);
  return 0;
}
'''


def compiler(tmp_path):
    llvm = Path(os.environ.get("ETX_LLVM", "/opt/homebrew/opt/llvm/bin"))
    probe = tmp_path / "probe.cpp"
    probe.write_text("#include <cstdio>\nint main() { return 0; }\n")
    for cxx in (os.environ.get("CXX"), "g++", "c++", "clang++", str(llvm / "clang++")):
        path = cxx and shutil.which(cxx)
        if path and run_command([path, "-std=c++17", str(probe), "-o", str(tmp_path / "probe")]).returncode == 0:
            return path
    pytest.skip("needs a host C++17 compiler with standard headers (set CXX)")


def test_host_reads_the_selection_from_the_manifest(tmp_path):
    """etx/selection.h accepts the plan the manifest describes and refuses a plan row that disagrees.
    The environment of the run does not matter: the host derives nothing."""
    out = tmp_path / "build"
    env = {k: v for k, v in os.environ.items()
           if k not in ("ETX_VL_IMPORTED", "ETX_VL_REQUESTS", "ETX_VL_BATCH", "ETX_VL_NB")}
    env.update(MATCH_ENV, ETX_VL_IMPORTS=str(IMPORTS), ETX_LLM_CONFIG=str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"),
               ETX_LLM_LAYERS="2", ETX_VL_PLAN=str(tmp_path / "vl_plan.txt"), ETX_VL_CHAINS=str(tmp_path / "chains.hip"))
    built = run_command([sys.executable, "-m", "etx", "compile", "examples/vllm_llm/model.py", "--arch", "gfx942",
                         "--out", str(out)], cwd=ROOT, env=env, timeout=600)
    assert built.returncode == 0, built.stdout[-2000:]
    plan_txt = tmp_path / "vl_plan.txt"
    (out / "main.cpp").write_text(DRIVER)
    exe = out / "check"
    r = run_command([compiler(tmp_path), "-std=c++17", "-O0", f"-I{INC}", f"-I{out}", str(out / "main.cpp"), "-o", str(exe)])
    assert r.returncode == 0, r.stdout
    ok = run_command([str(exe), str(plan_txt)])
    assert ok.returncode == 0 and ok.stdout.strip() == "ok 12", ok.stdout
    # the switches that select the variants today, set the other way at run time: no effect on the host
    flipped = run_command([str(exe), str(plan_txt)], env={"ETX_VL_DSL_TOPK": "0", "ETX_VL_DSL_GEMV": "0",
                                                          "ETX_VL_IMPORTED": "1", "PATH": "/usr/bin:/bin"})
    assert flipped.returncode == 0, flipped.stdout
    # a plan row that runs another export than the manifest names
    bad = tmp_path / "bad_plan.txt"
    bad.write_text(re.sub(r"^(\d+) dsl_topk ", r"\1 topk_softmax_e128 ", plan_txt.read_text(), count=1, flags=re.M))
    r = run_command([str(exe), str(bad)])
    assert r.returncode == 1 and "differs from the selection" in r.stdout and "topk_softmax_e128" in r.stdout, r.stdout
