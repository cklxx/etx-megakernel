"""Compatibility exports; implementation lives in etx.tools.tracing.ktrace."""
from etx.tools.tracing.ktrace import (
    argparse,
    bisect_left,
    bisect_right,
    csv,
    json,
    defaultdict,
    Path,
    median,
    read_vl_plan,
    bursts,
    PHASES,
    read_csv,
    duration,
    summarize,
    etx,
    vllm,
    _vendor_ranges,
    _boundary_crossing,
    _boundary_summary,
    vendor,
    _vendor_measure,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
