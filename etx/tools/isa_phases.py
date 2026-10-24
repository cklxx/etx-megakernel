"""Compatibility exports; implementation lives in etx.tools.diagnostics.isa_phases."""
from etx.tools.diagnostics.isa_phases import (
    argparse,
    json,
    Path,
    re,
    subprocess,
    ih,
    FUNC,
    PHASES,
    PIN_DEFS,
    STREAM_LOCS,
    RTTS,
    build_map,
    load_map,
    find_pins,
    phases,
    summary,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
