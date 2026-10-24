"""Compatibility exports; implementation lives in etx.tools.validation.cpu_check."""
from etx.tools.validation.cpu_check import (
    argparse,
    hashlib,
    importlib,
    json,
    os,
    re,
    Path,
    subprocess,
    sys,
    time,
    ET,
    ROOT,
    provenance,
    write_json,
    digest,
    proc,
    EXTERNAL_TESTS,
    file_hash,
    ARCHIVE_INDEX_SHA256,
    verify_archive,
    dependencies,
    test_results,
    command_step,
    pytest_environment,
    run,
    fixtures,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
