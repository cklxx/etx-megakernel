"""Compatibility exports; implementation lives in etx.tools.costs.stage_sim."""
from etx.tools.costs.stage_sim import (
    argparse,
    asdict,
    dataclass,
    json,
    math,
    Path,
    ROOT,
    LAT,
    source,
    Constant,
    check_constants,
    constants,
    simulate,
    sweep_lds_used,
    report,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
