"""python -m examples.vllm_llm.plan_imports <config.json>: the imported exports that model.py instances() will use.

build.sh lists its imports from the ETX_VL_DSL_* switches. The planner (etx.select) can select an imported kernel
that those switches do not list: the fallback of a GEMV phase outside every DSL variant (Qwen3-8B down, K = 12288,
host.hip:358), or the norm phases of a dense model. build.sh adds each export that this command prints and its list
lacks, so the import list follows the plan. Run it with the same ETX_VL_* environment as the model.py step.
"""
import sys

from examples.llm.model import load_dims
from examples.vllm_llm import model as M


def planned_imports(config: str) -> list[str]:
    return sorted({i["export"] for i in M.instances(load_dims(config)) if not i["export"].startswith("dsl_")})


if __name__ == "__main__":
    print(",".join(planned_imports(sys.argv[1])))
