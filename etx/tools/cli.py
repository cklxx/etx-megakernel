"""Compatibility exports; implementation lives in etx.tools.commands.cli."""
from etx.tools.commands.cli import (
    argparse,
    importlib,
    json,
    sys,
    Path,
    emit_kernel,
    emit_lowering_header,
    emit_plan_json,
    collect_sources,
    emit_tile_decls,
    emit_triton_skeleton,
    list_archs,
    load_machine,
    PassOptions,
    compile_graph,
    simulate,
    _load_example,
    _plan_from_args,
    cmd_compile,
    cmd_explain,
    cmd_archs,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
