"""Shared kept Qwen3-30B-A3B recipe and source attribution; no session runner."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEPT_ENV = dict(ETX_VL_CHECK='1', ETX_TOPK_KEY32='1', ETX_DEV_RESET='0',
                ETX_VL_REQUESTS='1', ETX_VL_CU='256', ETX_VL_TOUCH='1',
                ETX_PREFETCH_ISOLATE_POLLER='0',
                # Absorbed 2026-10-12: router narrowing (paired median 0.0626 ms,
                # three pairs all faster) and the norm hoist (0.0556 ms) both passed
                # the 0.030 ms KEEP bar against the previous kept build.
                ETX_VL_ROUTER_NARROW='1')

SOURCE_FILES = ('examples/sessions/qwen300_hour.py', 'examples/sessions/qwen300_hour_arms.json',
                'examples/vllm_llm/qwen300_config.py',
                'examples/vllm_llm/host.hip', 'etx/runtime/include/etx/host.h')
BASE_FLAGS = ('-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH -DETX_INBODY_REPORT'
              ' -DETX_DSL_NORM_HOIST')   # absorbed 2026-10-12 with the router narrowing


def source_hashes():
    from examples.vllm_llm.import_certificate import sha
    paths = {ROOT/p for p in SOURCE_FILES}
    paths.update(ROOT/p for p in ('examples/sessions/qwen300.py', 'examples/llm/vllm_bench.py',
                                  'examples/llm/configs/qwen3-30b-a3b.json'))
    for directory in ('etx', 'examples/vllm_llm'):
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file()
                     and p.suffix in ('.py', '.hip', '.h', '.sh') and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
