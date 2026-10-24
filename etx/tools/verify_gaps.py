"""Compatibility exports; implementation lives in etx.tools.diagnostics.verify_gaps."""
from etx.tools.diagnostics.verify_gaps import (
    argparse,
    difflib,
    hashlib,
    json,
    Path,
    re,
    isa,
    digest,
    parse_isa,
    kernels,
    function_ir,
    type_layout,
    lds_inventory,
    dependencies,
    memory_order,
    annotate_sources,
    report,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
