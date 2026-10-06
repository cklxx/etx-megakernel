"""Coverage checks for complete write extents and execution order.

Token comparisons cannot prove that every output row has a writer. The local
checks below scan complete extents and require the declared event protocol.
"""
from pathlib import Path
import copy

import pytest

from etx.tools.shared_state import check_plan, check_source, lds_inventory, sources

ROOT = Path(__file__).resolve().parents[1]
HOST = (ROOT / "examples/vllm_llm/host.hip").read_text()
PREP = (ROOT / "examples/llm/prep.py").read_text()


def test_gate_exists_and_implies_teacher():
    """--check-coverage must teacher-force the whole sequence: a free-running run's positions diverge from the
    HF prefill at the first mismatch and every later position is meaningless."""
    assert '"--check-coverage"' in HOST
    assert "check_coverage = true; teacher = true;" in HOST
    assert "CHECK-COVERAGE" in HOST
    # the gate fails the process, not just a printf
    assert "return 5" in HOST.split("CHECK-COVERAGE FAILED")[-1][:200] or "return 5" in HOST


def test_reference_file_is_derived_from_the_golden_name():
    """prefill_c1024.txt beside golden_c1024.txt, per request — not one global file."""
    assert '"prefill" + gname.substr(6)' in HOST
    assert "--prefill-only" in PREP and "def prefill_reference" in PREP
    assert '"prefill" + gname[len("golden"):]' in PREP


def test_mismatches_are_classified_by_the_hf_margin():
    """A bf16 argmax tie is not an unwritten row: mismatches with HF top-1 margin <= 0.5 are tolerated and
    counted separately; anything larger fails the gate."""
    assert "cov_marg" in HOST and "<= 0.5" in HOST
    assert "margin:" in PREP  # the reference carries per-position margins


def test_hole_injection_is_env_gated_and_not_default():
    """ETX_DEBUG_HOLE reproduces the round-42 bug shape on demand (the gate's negative test). It must be
    reachable only through getenv and must never alter a build where the env is unset."""
    assert HOST.count('getenv("ETX_DEBUG_HOLE")') == 1
    seg = HOST.split('getenv("ETX_DEBUG_HOLE")')[1][:600]
    assert "deliberately never written" in seg
    # the shadowed M feeds both the row count and rpt, so the hole is exactly [M', V)
    assert "int xM = x.M;" in HOST and ".i32(xM).i32(x.K).i32(rpt)" in HOST


def test_setup_asserts_cover_the_sibling_sites():
    """The written-output invariants the host controls fail at setup, not at the gate: MoE task counts cover
    nb*TOPK slots of MI/32 (up) and H/64 (down) rows; attention partitions (hardcoded 256 tokens each,
    round 11) cover maxlen."""
    assert "do not cover nb*TOPK*(MI/32)" in HOST
    assert "do not cover nb*TOPK*(H/64)" in HOST
    assert "do not cover maxlen" in HOST


@pytest.fixture(params=["qwen3-30b-a3b", "mixtral-8x7b", "phi-3.5-moe"])
def shared_plan(request, tmp_path):
    """Build the real task graph. No tensor values or GPU weights are needed."""
    from etx.tools.shared_state import build_local
    return build_local(request.param, tmp_path, layers=2)


def test_shared_state_acceptance(shared_plan):
    plan, contract = shared_plan
    errors, report = check_plan(plan, contract)
    assert errors == []
    assert check_source() == []
    from etx.tools.shared_state import check_arguments
    assert check_arguments(plan, contract) == []
    assert sum(s["elements"] for s in report if s["site"].endswith("/lm_head")) == contract.dims.V
    assert {s["site"].split("/")[-1] for s in report} >= {"moe_up", "moe_down", "attn", "reduce", "lm_head"}


@pytest.mark.parametrize("fault", ["write", "edge", "compiled_edge", "count", "tail", "duplicate"])
def test_missing_write_or_edge_names_the_site(shared_plan, fault):
    plan, contract = shared_plan
    plan = copy.deepcopy(plan)
    target = next(t for t in plan.tasks if t.grid == "lm_head" and t.coord == (200,))
    if fault == "write":
        plan.tasks.remove(target)
    elif fault == "duplicate":
        plan.tasks.append(copy.deepcopy(target))
    elif fault == "edge":
        plan.graph.grid("lm_head").in_edges.clear()
    elif fault == "compiled_edge":
        plan.inst.task_in[(target.grid, target.coord)].clear()
    elif fault == "count":
        event = next(iter(plan.graph.grid("lm_head").out_edges))
        plan.events[event].counts[0] -= 1
    elif fault == "tail":
        # The reference extent stays V. Shrink only the host launch extent.
        contract = copy.deepcopy(contract)
        next(x for x in contract.launches if x["role"] == "lm_head")["M"] -= 1536
    errors, _ = check_plan(plan, contract)
    assert errors
    assert any("lm_head" in e or fault == "count" and event in e for e in errors), errors
    if fault in ("write", "tail"):
        assert any("missing write" in e and "/lm_head" in e for e in errors), errors


def test_round48_barrier_negative():
    src = sources()
    # Remove the barrier from both compile branches. Leave all numerical work intact.
    import re
    src["gemv"] = re.sub(r"(if \(d.mode == 4\) \{\s*)__syncthreads\(\);", r"\1", src["gemv"])
    assert any("dsl_stage_norm_at:xs" in e for e in check_source(src))


def test_lds_inventory_comes_from_source():
    src = sources()
    original = lds_inventory(src)
    assert any(s["site"] == "gemv:dsl_stage_norm_at:xs" and s["writers"] and s["readers"] for s in original)
    src["native"] = src["native"].replace("sv", "new_shared_values")
    changed = lds_inventory(src)
    assert any(s["site"] == "native:vl_argmax:new_shared_values" for s in changed)
    assert not any(s["site"] == "native:vl_argmax:sv" for s in changed)


@pytest.mark.parametrize("model", ["qwen3-30b-a3b", "mixtral-8x7b", "phi-3.5-moe"])
def test_all_model_layers(model, tmp_path):
    from etx.tools.shared_state import build_local, check_arguments
    plan, contract = build_local(model, tmp_path)
    errors, report = check_plan(plan, contract)
    assert errors == []
    assert check_arguments(plan, contract) == []
    assert len([s for s in report if s["site"].endswith("/moe_down")]) == contract.dims.L


@pytest.mark.parametrize("unit,function,count", [
    ("gemv", "dsl_stage", 1), ("gemv", "dsl_stage_norm_at", 2),
    ("gemv", "dsl_topk_lds", 1), ("gemv", "dsl_gemv_rows_nb", 3),
    ("gemv", "dsl_gemv_split", 3), ("gemv", "dsl_gemv_bigk", 2),
    ("reduction", "etx_block_sum", 2), ("reduction", "etx_block_max", 2),
    ("native", "vl_argmax", 2),
])
def test_each_required_lds_barrier(unit, function, count):
    from etx.tools.shared_state import functions, preprocess
    src = {name: preprocess(text) for name, text in sources().items()}
    body = functions(src[unit])[function][1]
    assert body.count("__syncthreads();") == count
    for ordinal in range(count):
        parts = body.split("__syncthreads();")
        parts[ordinal:ordinal + 2] = [parts[ordinal] + parts[ordinal + 1]]
        broken = {**src, unit: src[unit].replace(body, "__syncthreads();".join(parts))}
        assert any(function in error for error in check_source(broken)), (function, ordinal)


def test_missing_queue_entry(shared_plan):
    plan, contract = shared_plan
    task = next(t for t in plan.tasks if t.grid == "moe_down_0")
    plan.static_queues[task.device, task.worker].remove(task.id)
    errors, _ = check_plan(plan, contract)
    assert any("moe_down_0" in error and "queue" in error for error in errors)


def test_wrong_argument_stride(shared_plan):
    from etx.tools.shared_state import check_arguments
    plan, contract = shared_plan
    grid = next(g for g in plan.graph.grids if "impargs" in g.args)
    grid.consts = (grid.consts[0] + 16, *grid.consts[1:])
    assert any(grid.name in error for error in check_arguments(plan, contract))


@pytest.mark.parametrize("symbol", ["dsl_run_dsl_gemv_fold", "dsl_run_dsl_moe_up_fold", "dsl_run_dsl_moe_down_fold"])
def test_missing_tile_store(symbol):
    from etx.tools.shared_state import functions, preprocess
    src = {name: preprocess(text) for name, text in sources().items()}
    body = functions(src["gemv"])[symbol][1]
    src["gemv"] = src["gemv"].replace(body, body.replace("dsl_a_stbf(", "removed_store("))
    assert any(symbol in error for error in check_source(src))


def test_dead_barriers_fail():
    src = {name: text.replace('__syncthreads();', 'if (false) __syncthreads();')
           for name, text in sources().items()}
    assert check_source(src)


def test_lane_only_barrier_fails():
    from etx.tools.shared_state import functions
    src = sources()
    body = functions(src['reduction'])['etx_block_sum'][1]
    changed = body.replace('__syncthreads();',
                           'if (etx_lane() == 0) { __syncthreads(); }', 1)
    src['reduction'] = src['reduction'].replace(body, changed)
    assert check_source(src) == [
        'etx_block_sum:red: barrier control path differs from the required protocol']


def test_missing_native_token_store_fails():
    src = sources()
    assert 'VP.token_out[t] = si[0];' in src['native']
    src['native'] = src['native'].replace('VP.token_out[t] = si[0];', '')
    assert check_source(src) == [
        'vl_argmax: native output needs one writer on its required path: VP.token_out[t] = si[0];']


def test_norm_barrier_guard_alone():
    # Text order stays valid. The barrier no longer belongs to the mean branch.
    src = sources()
    src['gemv'] = src['gemv'].replace(
        'if (d.mode == 4) {\n    __syncthreads();',
        '__syncthreads();\n  if (d.mode == 4) {')
    errors = check_source(src)
    assert len(errors) == 1, errors
    assert 'dsl_stage_norm_at:xs' in errors[0]


@pytest.mark.parametrize('coherent', [0, 1])
def test_norm_barrier_must_start_the_mean_branch(coherent):
    from etx.tools.shared_state import preprocess
    src = sources()
    import re
    src['gemv'], count = re.subn(
        r'(if \(d.mode == 4\) \{\s*)__syncthreads\(\);(\s*float s1 = 0.f;)',
        r'\1\2 __syncthreads();', preprocess(src['gemv'], coherent))
    assert count == 1
    assert check_source(src) == [
        'dsl_stage_norm_at:xs: barrier control path differs from the required protocol']


@pytest.mark.parametrize('replacement', [
    'if (false) { __syncthreads(); }',
    'if (etx_lane() == 0) __syncthreads();',
    'while (false) __syncthreads();',
    'auto unused = [&]() { __syncthreads(); };',
])
def test_barrier_control_forms(replacement):
    from etx.tools.shared_state import functions
    src = sources()
    body = functions(src['reduction'])['etx_block_sum'][1]
    src['reduction'] = src['reduction'].replace(
        body, body.replace('__syncthreads();', replacement, 1))
    assert check_source(src) == [
        'etx_block_sum:red: barrier control path differs from the required protocol']


def test_early_exit_guard_alone():
    from etx.tools.shared_state import functions
    src = sources()
    body = functions(src['reduction'])['etx_block_sum'][1]
    src['reduction'] = src['reduction'].replace(
        body, 'if (etx_lane() == 0) return 0.f;\n' + body)
    assert check_source(src) == ['etx_block_sum: unsupported control exit']


@pytest.mark.parametrize('replacement', [
    'if (false) VP.token_out[t] = si[0];',
    'VP.token_out[t + 1] = si[0];',
    '',
])
def test_native_writer_path_and_index(replacement):
    src = sources()
    src['native'] = src['native'].replace('VP.token_out[t] = si[0];', replacement)
    assert check_source(src) == [
        'vl_argmax: native output needs one writer on its required path: VP.token_out[t] = si[0];']


def test_control_parse_guard_alone():
    from etx.tools.shared_state import functions
    src = sources()
    body = functions(src['reduction'])['etx_block_sum'][1]
    src['reduction'] = src['reduction'].replace(body, 'switch (0) { default: break; }' + body)
    assert check_source(src) == [
        'etx_block_sum: control parse failed: unsupported control statement: switch']


def test_unconditional_braces_preserve_barrier():
    src = sources()
    src['reduction'] = src['reduction'].replace('__syncthreads();', '{ { __syncthreads(); } }')
    assert check_source(src) == []


@pytest.mark.parametrize('name,store', [
    ('vl_prologue', 'VP.token_time[2 * t] = wall_clock64();'),
    ('vl_prologue', 'VP.positions[0] = VP.pos;'),
    ('vl_prologue', 'VP.seq_lens[0] = VP.pos + 1;'),
    ('vl_prologue', 'VP.slot_mapping[0] = VP.pos;'),
    ('vl_prologue', 'dst[i] = src[i % n8];'),
    ('vl_argmax', 'VP.token_in[t + 1] = si[0];'),
    ('vl_argmax', 'VP.token_time[2 * t + 1] = wall_clock64();'),
])
def test_each_native_output_has_a_source_writer(name, store):
    src = sources()
    assert src['native'].count(store) == 1
    # Keep an empty statement so each unbraced loop or branch retains its body.
    src['native'] = src['native'].replace(store, ';')
    assert check_source(src) == [
        f'{name}: native output needs one writer on its required path: {store}']


def test_text_order_guard_alone():
    src = sources()
    src['reduction'] = src['reduction'].replace('t += red[w];', 't += red[0];')
    assert check_source(src) == [
        r'etx_block_sum:red: required write or barrier absent: red\[w\]']


def test_fold_output_guard_alone():
    src = sources()
    src['gemv'] = src['gemv'].replace('store && rout != nullptr', 'store && rout == nullptr')
    assert check_source(src) == [
        'dsl_stage_norm_at:rout: required fold output write absent']
