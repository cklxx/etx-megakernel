"""Compatibility exports; implementation lives in etx.tools.tracing.inbody."""
from etx.tools.tracing.inbody import (
    argparse,
    defaultdict,
    json,
    Path,
    re,
    struct,
    tl,
    MAGIC,
    read_raw,
    event_words,
    analyze,
    LOAD_ROLES,
    union_ticks,
    load_phases,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
