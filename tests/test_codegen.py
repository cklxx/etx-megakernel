import json
import shutil
import subprocess
from pathlib import Path

import pytest

from etx.codegen import emit_kernel, emit_lowering_header, emit_plan_json
from etx.frontends import emit_tile_decls
from etx.machine import list_archs, load_machine
from etx.passes import compile_graph
from examples import gemm_rs, moe_layer, splitk_sum

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("arch", list_archs())
def test_lowering_header_has_every_scope(arch):
    h = emit_lowering_header(load_machine(arch))
    for s in ("DOMAIN", "DEVICE", "SYSTEM"):
        for p in ("RELEASE", "ACQUIRE", "POLL", "ARRIVE"):
            assert f"#define ETX_{p}_{s}" in h
    assert "ETX_BACKOFF" in h and "ETX_DOMAIN_ID" in h


def test_kernel_text_dispatches_every_grid():
    b = moe_layer.bindings()
    plan = compile_graph(moe_layer.build(), "gfx942", b, moe_layer.runtime(b))
    src = emit_kernel(plan)
    for g in plan.graph.grids:
        assert f"case {plan.type_ids[g.name]}:" in src and g.body.symbol in src
    assert "etx_wait_DEVICE" in src or "etx_wait_DOMAIN" in src
    assert "etx_push_consumers" in src          # dynamic segments push
    assert "prologue recomputes norm" in src    # Pass 3 visible in the kernel
    decls = emit_tile_decls(plan.graph)
    assert "etx_tile_group_gemm_up" in decls
    data = json.loads(emit_plan_json(plan))
    assert data["arch"] == "gfx942" and len(data["descs"]) == len(plan.tasks)
    assert any(e["runtime_init"] for e in data["events"])


def test_multi_device_emits_one_kernel_per_device():
    b = gemm_rs.bindings()
    plan = compile_graph(gemm_rs.build(2), "gfx90a", b, {})
    k0, k1 = emit_kernel(plan, 0), emit_kernel(plan, 1)
    assert "etx_tile_reduce_scatter" in k0 and "etx_tile_reduce_scatter" in k1
    assert "etx_wait_SYSTEM" in k0


@pytest.mark.skipif(shutil.which("hipcc") is None, reason="hipcc not installed")
def test_generated_hip_compiles(tmp_path):
    b = splitk_sum.bindings()
    plan = compile_graph(splitk_sum.build(), "gfx942", b, {})
    (tmp_path / "etx_lowering.h").write_text(emit_lowering_header(plan.machine))
    src = tmp_path / "k.hip"
    src.write_text(emit_kernel(plan))
    inc = ROOT / "etx" / "runtime" / "include"
    subprocess.run(["hipcc", "--offload-arch=gfx942", "-fgpu-rdc", "-fsyntax-only", f"-I{inc}", f"-I{tmp_path}", str(src)], check=True)


def test_arg_table_does_not_survive_its_plan():
    """The argument table was cached in a module dict keyed by id(plan). CPython reuses the id of a freed plan, so a
    new plan with other tensors got the dead plan's table (KeyError 'impargs' when test_selection_manifest.py ran
    before test_feature_manifest.py). Plans with different tensors are created and freed in turn; each must get its
    own table."""
    import gc

    from etx.codegen.kernel import _arg_table
    from etx.codegen.plan_header import emit_plan_header
    from etx.frontends import hip_link
    from etx.ir import Graph, Resource

    def plan_with(tensor):
        g = Graph("arg_" + tensor)
        g.tensor(tensor, (64,), role="activation", bytes_per_elem=2)
        g.etensor("E", (1,), wait_count=1)
        g.call_device("a", (1,), hip_link("a.hip", "etx_tile_a"), resource=Resource(threads=256), args=[tensor],
                      out_edges={"E": "i->(0)"}, domain_map="i->(0)", duration_us=1.0)
        return compile_graph(g, "gfx942", {}, {})

    for n in range(40):
        name = "impargs" if n % 2 else f"t{n}"
        p = plan_with(name)
        assert _arg_table(p) == {name: 0}
        assert f'"{name}"' in emit_plan_header(p)
        del p
        gc.collect()
    p = plan_with("x")
    p.graph.tensor("y", (8,), role="activation", bytes_per_elem=2)     # tensors added after the first lookup
    _arg_table(p)
    p.graph.tensor("z", (8,), role="activation", bytes_per_elem=2)
    assert _arg_table(p) == {"x": 0, "y": 1, "z": 2}


def test_large_dispatch_preserves_every_integer_and_case():
    """Invert the factoring transformation: each original type has identical C."""
    import re
    from etx.codegen.kernel import _compact_dispatch
    original = ['  switch (t.type) {']
    for i in range(600):
        original += [f'    case {i}: {{ // request {i}',
                     f'      ctx.cst[0] = {123*i}; ctx.cst[1] = 64;',
                     f'#define ETX_TARGET(idx) etx_wait_SYSTEM(p.events + ETX_EVW({i+9} + (idx)), p.ctrl_abort)',
                     '  ETX_TARGET(t.coord[0]);', '#undef ETX_TARGET',
                     f'      sym = {i % 3};', '      break; }']
    original += ['    default: break;', '  }']
    transformed = list(original)
    _compact_dispatch(transformed, 0, 'test', 600)
    src = '\n'.join(transformed)
    assert src.count('etx_wait_SYSTEM(') == 1
    table = re.search(r'etx_test_data\[600\]\[(\d+)\] = \{(.*?)\};', src, re.S)
    rows = re.findall(r'\{([^{}]*)\}', table[2])
    for i in range(600):
        assert f'case {i}:' in src
        values = list(map(int, rows[i].split(',')))
        body = re.search(r'    \{\n(.*?)      break;', src, re.S)[1]
        reconstructed = re.sub(r'etx_test_data\[t.type\]\[(\d+)\]', lambda m: str(values[int(m[1])]), body)
        assert f'ctx.cst[0] = {123*i}; ctx.cst[1] = 64;' in reconstructed
        assert f'ETX_EVW({i+9} + (idx))' in reconstructed
        assert f'sym = {i % 3};' in reconstructed
    small = ['unchanged']
    _compact_dispatch(small, 0, 'test', 10)
    assert small == ['unchanged']


def test_compact_dispatch_keeps_different_scopes_and_hooks_separate():
    from etx.codegen.kernel import _compact_dispatch
    lines = ['  switch (t.type) {']
    for i, scope in enumerate(('DOMAIN', 'SYSTEM', 'DOMAIN')):
        lines += [f'    case {i}: {{ // {scope}',
                  f'      etx_wait_{scope}(p.events + {i}, p.ctrl_abort);',
                  '      return true; }']
    lines += ['    default: return true;', '  }']
    _compact_dispatch(lines, 0, 'ready', 600)
    src = '\n'.join(lines)
    assert src.count('etx_wait_DOMAIN(') == src.count('etx_wait_SYSTEM(') == 1
    assert 'case 0:\n    case 2:' in src
    assert 'case 1:' in src
