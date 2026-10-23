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
