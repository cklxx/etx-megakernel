"""Compatibility exports; implementation lives in etx.tools.tracing.layer_budget."""
from etx.tools.tracing.layer_budget import (
    argparse,
    heapq,
    json,
    re,
    Path,
    tl,
    METRIC_LIMITS,
    dependency_order,
    layer_of,
    analyze,
    compare,
    report,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
