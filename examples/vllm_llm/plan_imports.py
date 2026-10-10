"""Compatibility entry point; both build scripts use plan_exports directly."""
import sys
from examples.vllm_llm.plan_exports import exports


def planned_imports(config: str) -> list[str]:
    _, names, _ = exports(config)
    return sorted(n for n, kind in names.items() if kind == "imported")


if __name__ == "__main__":
    print(",".join(planned_imports(sys.argv[1])))
