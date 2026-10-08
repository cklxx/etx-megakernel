"""Item 6 of docs/PLAN-generic-decoder-2026-10-08.md section 7: the planner's selection goes into the feature
manifest, and the host reads it from there (docs/SEC7-interface-fix.md section 6).

The selection is published in plan_data.h only (etx/codegen/manifest.py selection_block). The host reads it through
etx/runtime/include/etx/selection.h, which these tests compile and run with the CPU compiler. The megakernel source
must not change when the selection equals the current choice.
"""
import importlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from etx.codegen import emit_kernel, manifest
from etx.codegen.plan_header import emit_plan_header
from etx.passes import compile_graph

ROOT = Path(__file__).resolve().parents[1]
INC = ROOT / "etx/runtime/include"
BUDGET = {"threads": 1024, "vgpr": 128, "lds": 65536}
# The MATCH configuration of the round11 build b_r8base (docs/PLAN-generic-decoder-2026-10-08.md section 2.2).
MATCH_ENV = {"ETX_VL_XCD": "1", "ETX_VL_CHAIN": "1", "ETX_VL_DSL_GEMV": "1", "ETX_VL_DSL_TOPK": "1",
             "ETX_VL_DSL_ATTN": "head", "ETX_VL_DSL_FOLD": "1", "ETX_VL_TOUCH": "1", "ETX_VL_COH": "1",
             "ETX_VL_CU": "256", "ETX_VL_FOLD_TOPK": "0"}

# The C++ twin of the plan loop in examples/vllm_llm/host.hip: every plan row through etx_sel_check, then every
# selected phase must have a row. Exit 0 = the plan and the manifest agree.
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


def _build(tmp_path, monkeypatch, layers=2, **env):
    from etx.ops import attnprep as _prep, gemv as _gemv, topk as _topk
    from tests.test_moe_mapping import IMPORTS
    imp = {e: {"slots": s or 8, "arg_bytes": a} for e, (s, a) in IMPORTS.items()}
    for mod in (_gemv, _topk, _prep):
        imp.update(json.loads(json.dumps(mod.import_entries())))
    (tmp_path / "imports.json").write_text(json.dumps(imp))
    base = {"ETX_VL_IMPORTS": str(tmp_path / "imports.json"),
            "ETX_LLM_CONFIG": str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"), "ETX_LLM_LAYERS": str(layers),
            "ETX_VL_PLAN": str(tmp_path / "vl_plan.txt"), "ETX_VL_CHAINS": str(tmp_path / "chains.hip")}
    for k, v in {**base, **MATCH_ENV, **env}.items():
        monkeypatch.setenv(k, v)
    for k in ("ETX_VL_IMPORTED", "ETX_VL_REQUESTS", "ETX_VL_BATCH", "ETX_VL_NB"):
        if k not in env:
            monkeypatch.delenv(k, raising=False)
    import examples.vllm_llm.model as vm
    vm = importlib.reload(vm)
    g = vm.build()
    return g, compile_graph(g, "gfx942", {}, {})


def _rows(tmp_path):
    rows = [ln.split() for ln in (tmp_path / "vl_plan.txt").read_text().splitlines()[2:]]
    return [(r[3], int(r[2]), r[1]) for r in rows]


def _published(header: str) -> dict:
    arr = lambda name: re.findall(r'"([^"]*)"', re.search(rf"{name}\[\d+\] = \{{(.*?)\}};", header).group(1))
    kinds = re.search(r"etx_sel_kind\[\d+\] = \{(.*?)\};", header).group(1).split(", ")
    return {p: (e, manifest.SELECTION_KINDS[int(k)]) for p, e, k in zip(arr("etx_sel_phase"), arr("etx_sel_export"), kinds)}


def _compiler():
    cxx = shutil.which("c++")
    assert cxx, "CPU acceptance requires a C++ compiler"
    return cxx


def _driver(tmp_path, header: str, *defines):
    d = tmp_path / "drv"
    d.mkdir(exist_ok=True)
    (d / "plan_data.h").write_text(header)
    (d / "main.cpp").write_text(DRIVER)
    exe = d / "check"
    r = subprocess.run([_compiler(), "-std=c++17", "-O0", f"-I{INC}", f"-I{d}", *defines, str(d / "main.cpp"), "-o", str(exe)],
                       capture_output=True, text=True)
    return exe, r


def _run(exe, plan_txt, env=None):
    return subprocess.run([str(exe), str(plan_txt)], capture_output=True, text=True, env=env)


def test_selection_of_splits_layer_zero_only_when_it_differs():
    rows = [("in_norm", 0, "rms_norm_2d"), ("qkv", 0, "dsl_gemv"), ("qkv", 1, "dsl_gemv_fold"), ("qkv", 2, "dsl_gemv_fold"),
            ("attn", 0, "attn_mfma16_g8"), ("attn", 1, "attn_mfma16_g8"), ("lm_head", -1, "dsl_gemv_fold")]
    sel = manifest.selection_of(rows, BUDGET)
    assert sel["budget"] == BUDGET
    assert sel["phases"] == {"attn": ("attn_mfma16_g8", "imported"), "in_norm": ("rms_norm_2d", "imported"),
                             "lm_head": ("dsl_gemv_fold", "dsl"), "qkv": ("dsl_gemv_fold", "dsl"), "qkv@0": ("dsl_gemv", "dsl")}
    assert manifest.phase_key("qkv", 0, sel["phases"]) == "qkv@0"
    assert manifest.phase_key("qkv", 3, sel["phases"]) == "qkv"
    assert manifest.phase_key("attn", 0, sel["phases"]) == "attn"


def test_selection_of_refuses_a_phase_with_two_exports():
    with pytest.raises(ValueError, match="phase topk"):
        manifest.selection_of([("topk", 1, "dsl_topk"), ("topk", 2, "topk_softmax_e128")], BUDGET)
    with pytest.raises(ValueError, match="kind"):
        manifest.selection_block({"budget": BUDGET, "phases": {"topk": ("dsl_topk", "triton")}})
    assert manifest.selection_block(None) == [] and manifest.selection_block({}) == []


def test_match_plan_publishes_todays_choice(tmp_path, monkeypatch):
    """The manifest names exactly the export that each plan instance runs. MATCH: 12 phases, qkv split at layer 0."""
    g, plan = _build(tmp_path, monkeypatch)
    pub = _published(emit_plan_header(plan))
    assert pub == g.codegen[manifest.SELECTION]["phases"]
    for role, layer, export in _rows(tmp_path):
        assert pub[manifest.phase_key(role, layer, pub)][0] == export, (role, layer, export)
    assert pub["qkv@0"] == ("dsl_gemv", "dsl") and pub["qkv"] == ("dsl_gemv_fold", "dsl")
    assert pub["topk"] == ("dsl_topk", "dsl") and pub["attn"] == ("attn_mfma16_g8", "imported")
    assert len(pub) == 12
    header = emit_plan_header(plan)
    for mac, v in (("ETX_SEL_THREADS", 1024), ("ETX_SEL_VGPR", 128), ("ETX_SEL_LDS", 65536)):
        assert f"#define {mac} {v}\n" in header


def test_megakernel_source_is_byte_identical_with_the_selection(tmp_path, monkeypatch):
    """The selection reaches plan_data.h only. With today's choice, the gfx942 megakernel source does not change,
    and plan_data.h changes by the selection block only."""
    g, plan = _build(tmp_path, monkeypatch)
    with_sel_kernel, with_sel_header = emit_kernel(plan), emit_plan_header(plan)
    sel = plan.graph.codegen.pop(manifest.SELECTION)
    assert emit_kernel(plan) == with_sel_kernel
    without = emit_plan_header(plan)
    block = "\n".join(manifest.selection_block(sel))
    assert block and block not in without and with_sel_header.replace(block + "\n", "", 1) == without
    assert "etx_sel_" not in with_sel_kernel and "ETX_SEL_" not in with_sel_kernel


def test_host_reads_the_selection_from_the_manifest(tmp_path, monkeypatch):
    """etx/selection.h, compiled for the CPU, accepts the plan that the manifest describes and refuses a plan row
    or a manifest that disagrees. The environment of the run does not matter: the host derives nothing."""
    g, plan = _build(tmp_path, monkeypatch)
    plan_txt = tmp_path / "vl_plan.txt"
    exe, r = _driver(tmp_path, emit_plan_header(plan))
    assert r.returncode == 0, r.stderr
    ok = _run(exe, plan_txt)
    assert ok.returncode == 0 and ok.stdout.strip() == "ok 12", ok.stdout
    # the switches that select the variants today, set the other way at run time: no effect on the host
    flipped = _run(exe, plan_txt, env={"ETX_VL_DSL_TOPK": "0", "ETX_VL_DSL_GEMV": "0", "ETX_VL_IMPORTED": "1", "PATH": "/usr/bin:/bin"})
    assert flipped.returncode == 0, flipped.stdout
    # a plan row that runs another export than the manifest names
    bad = tmp_path / "bad_plan.txt"
    bad.write_text(re.sub(r"^(\d+) dsl_topk ", r"\1 topk_softmax_e128 ", plan_txt.read_text(), count=1, flags=re.M))
    r = _run(exe, bad)
    assert r.returncode == 1 and "differs from the selection" in r.stdout and "topk_softmax_e128" in r.stdout, r.stdout
    # a manifest that selects another variant for one phase than the plan file runs
    sel = plan.graph.codegen[manifest.SELECTION]   # compile_graph copies the graph: the plan holds the manifest input
    sel["phases"]["topk"] = ("topk_softmax_e128", "imported")
    exe2, r = _driver(tmp_path, emit_plan_header(plan))
    assert r.returncode == 0, r.stderr
    r = _run(exe2, plan_txt)
    assert r.returncode == 1 and "dsl_topk" in r.stdout, r.stdout
    # a manifest that names a phase the plan does not run
    sel["phases"]["topk"] = ("dsl_topk", "dsl")
    sel["phases"]["moe_sum"] = ("moe_sum_k8", "imported")
    exe3, r = _driver(tmp_path, emit_plan_header(plan))
    r = _run(exe3, plan_txt)
    assert r.returncode == 2 and "moe_sum" in r.stdout, r.stdout


def test_a_command_line_selection_is_a_compile_error(tmp_path, monkeypatch):
    _, plan = _build(tmp_path, monkeypatch)
    _, r = _driver(tmp_path, emit_plan_header(plan), "-DETX_SEL_N_PHASES=3")
    assert r.returncode != 0 and "must not be defined on the command line" in r.stderr
    header = emit_plan_header(plan)
    no_sel = header.replace("\n".join(manifest.selection_block(plan.graph.codegen[manifest.SELECTION])) + "\n", "", 1)
    _, r = _driver(tmp_path, no_sel)
    assert r.returncode != 0 and "publishes no variant selection" in r.stderr


def test_vllm_host_checks_every_instance_against_the_manifest():
    """host.hip needs HIP to compile, so this pins the call sites: it includes the reader, checks each plan row, and
    reads no selection switch from the environment."""
    src = (ROOT / "examples/vllm_llm/host.hip").read_text()
    assert '#include "etx/selection.h"' in src
    assert src.index('#include "plan_data.h"') < src.index('#include "etx/selection.h"')
    loop = src[src.index("// ---- plan"):src.index("// ---- golden")]
    assert "etx_sel_check(x.role.c_str(), x.layer, x.exp.c_str())" in loop
    assert loop.index("etx_sel_check(") < loop.index("insts.push_back(x)")
    assert "etx_sel_find_instance(" in loop and "has no instance" in loop
    for name in ("ETX_VL_DSL", "ETX_VL_IMPORTED", "ETX_VL_FOLD"):
        assert f'getenv("{name}' not in src, name
