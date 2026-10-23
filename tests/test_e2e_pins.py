"""External-only E2E tests restate a few pinned constants; keep them equal to their internal sources."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def pinned(path, name):
    match = re.search(rf"^{name} = '([^']+)'$", (ROOT / path).read_text(), re.M)
    assert match, (path, name)
    return match.group(1)


def test_qwen_staging_pins_match_the_preflight():
    from examples.vllm_llm.vm_preflight import GOLDEN_SHA256, VLLM_TAG
    assert pinned('tests/e2e/test_api_qwen_staging.py', 'VLLM_TAG') == VLLM_TAG
    assert pinned('tests/e2e/test_api_qwen_staging.py', 'GOLDEN_SHA256') == GOLDEN_SHA256


def test_glm_seam_label_matches_the_harness():
    from etx.models.glm53 import seam
    assert pinned('tests/e2e/test_api_glm53_seam.py', 'LABEL') == seam.LABEL


def test_decode_known_models_match_the_entry():
    import ast
    from etx.models.vllm_llm import decode
    source = (ROOT / 'tests/e2e/test_api_decode_entry.py').read_text()
    known = next(n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'KNOWN')
    assert tuple(ast.literal_eval(known)) == tuple(decode.KNOWN)


def test_inbody_magic_matches_the_trace_reader():
    from etx.tools.inbody import MAGIC
    assert int(pinned('tests/e2e/test_api_qwen300_hour.py', 'MAGIC'), 16) == MAGIC


def test_g6_build_inputs_match_the_session():
    import ast
    from examples.vllm_llm.g6_session import BUILD_INPUTS
    source = (ROOT / 'tests/e2e/test_api_qwen_g6.py').read_text()
    pinned = next(n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'BUILD_INPUTS')
    assert tuple(ast.literal_eval(pinned)) == BUILD_INPUTS


def test_qwen300_registered_count_matches_the_session():
    import ast
    from examples.sessions.qwen300_hour import REGISTERED
    source = (ROOT / 'tests/e2e/test_api_qwen300_hour.py').read_text()
    pinned = next(n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'REGISTERED_COUNT')
    assert ast.literal_eval(pinned) == len(REGISTERED)


def test_stage_table_location_matches_the_tool():
    from examples.dsv2lite.stage_table import OUT
    source = (ROOT / 'tests/e2e/test_api_dsv2_stage_table.py').read_text()
    assert "COMMITTED = ROOT / '" + str(OUT.relative_to(ROOT)) + "'" in source


def test_golden_files_and_rotation_match_the_tool():
    import ast
    from examples.dsv2lite.batch_goldens import FILES, ROTATE
    tree = ast.parse((ROOT / 'tests/e2e/test_api_dsv2_goldens.py').read_text())
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('FILES', 'ROTATE')}
    assert values == dict(FILES=tuple(FILES), ROTATE=ROTATE)


def test_attribution_batches_match_the_session():
    import ast
    from examples.dsv2lite.batch_attribution import BATCHES
    tree = ast.parse((ROOT / 'tests/e2e/test_api_dsv2_attribution_final.py').read_text())
    pinned = next(n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'BATCHES')
    assert tuple(ast.literal_eval(pinned)) == tuple(BATCHES)


def test_batch_session_batches_match_the_arms():
    import ast
    from examples.dsv2lite.arms import BATCHES
    tree = ast.parse((ROOT / 'tests/e2e/test_api_dsv2_batch_report.py').read_text())
    pinned = next(n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'BATCHES')
    assert tuple(ast.literal_eval(pinned)) == tuple(BATCHES)


def test_glm_module_gate_identity_matches_the_harness():
    import ast
    from examples.glm53 import module_gate
    from examples.glm53.prepare import MODULE_MODEL
    tree = ast.parse((ROOT / 'tests/e2e/test_api_glm53_module_gate.py').read_text())
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('MODULE_MODEL', 'MODULE_BATCH')}
    assert values == dict(MODULE_MODEL=MODULE_MODEL, MODULE_BATCH=module_gate.BATCH)


def test_selection_match_environment_matches_the_conformance_build():
    import ast
    from tests.test_selection_manifest import MATCH_ENV
    tree = ast.parse((ROOT / 'tests/e2e/test_api_selection_manifest.py').read_text())
    pinned = next(n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MATCH_ENV')
    assert ast.literal_eval(pinned) == MATCH_ENV
