"""Run one registered harness command with the historical argument parser, against an explicit repository."""
import importlib
import sys

from etx.api.harness import HarnessCommandRequest


PACKAGES = {'qwen-expert-merge': 'etx.models.vllm_llm.expert_merge',
            'qwen-attention-early-merge': 'etx.models.vllm_llm.attention_early_merge',
            'qwen-bf16-probe': 'etx.models.vllm_llm.bf16_300',
            'glm53-seam': 'etx.models.glm53',
            'glm53-boundary': 'etx.models.glm53',
            'glm53-model-build': 'etx.models.glm53',
            'glm53-run-all': 'etx.models.glm53',
            'qwen-vllm': 'etx.models.vllm_llm',
            'deepseek-tools': 'etx.models.dsv2lite'}


def run_harness_command(request):
    if not isinstance(request, HarnessCommandRequest):
        raise TypeError('run_harness_command requires HarnessCommandRequest')
    package = PACKAGES[request.harness]
    importlib.import_module(package + '.paths').configure(request.root)
    module = importlib.import_module(package + '.' + request.module)
    saved = sys.argv
    # The program name stays the caller's (the historical `python -m examples...` path) for usage and help text.
    sys.argv = [saved[0] if saved else request.module, *request.argv]
    try:
        # Some commands return their status, others raise SystemExit; both are the command's exit code.
        code = module.main()
    except SystemExit as exit:
        code = exit.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    finally:
        sys.argv = saved
    return code if isinstance(code, int) else 0
