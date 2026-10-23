"""Compatibility entry point; both build scripts use plan_exports directly."""
import sys
from etx.models.declarations import declaration as _declaration
from . import paths
exports = _declaration(paths.ROOT, "vllm_llm.plan_exports").exports


def planned_imports(config: str) -> list[str]:
    _, names, _ = exports(config)
    return sorted(n for n, kind in names.items() if kind == "imported")


def main():
    print(",".join(planned_imports(sys.argv[1])))


if __name__ == "__main__":
    main()
