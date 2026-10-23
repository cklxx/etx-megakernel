"""Reviewed file-level test classification, including legacy E2E exceptions."""
import json
from pathlib import Path


def check_test_inventory(root: Path) -> tuple[str, ...]:
    """Require every root Python/shell test to have an explicit classification.

    External tests are governed by the E2E import/fixture policy instead. Mixed
    files remain exceptions until their lifecycle cases migrate; this inventory
    never certifies their internal imports as public APIs.
    """
    root = Path(root)
    manifest = json.loads((root / 'api_inventory.json').read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {'schema_version', 'legacy_baseline', 'tests'} or
            type(manifest['schema_version']) is not int or manifest['schema_version'] != 1):
        raise ValueError('unsupported test inventory schema')
    rows = manifest['tests']
    historical = manifest['legacy_baseline']
    if (not isinstance(rows, dict) or not isinstance(historical, list) or
            any(type(name) is not str for name in historical) or len(set(historical)) != len(historical)):
        raise ValueError('invalid inventory rows or baseline')
    baseline = set(historical)
    paths = {path.name for path in root.glob('test_*') if path.suffix in ('.py', '.sh')}
    violations = []
    if set(rows) != paths:
        violations.append(f'unclassified tests: {sorted(paths - set(rows))}; stale entries: {sorted(set(rows) - paths)}')
    for path, row in rows.items():
        if not isinstance(row, dict) or set(row) != {'classification', 'owners', 'reason'}:
            violations.append(f'{path}: invalid inventory fields')
            continue
        classification = row['classification']
        if classification not in ('conformance', 'legacy-e2e', 'mixed'):
            violations.append(f'{path}: unknown classification {classification!r}')
        if not isinstance(row['reason'], str) or not row['reason'].strip():
            violations.append(f'{path}: missing classification rationale')
        owners = row['owners']
        if not isinstance(owners, list) or any(type(owner) is not str or owner not in {f'API-{n:02}' for n in range(1, 28)} for owner in owners):
            violations.append(f'{path}: invalid migration owners')
            continue
        if classification in ('legacy-e2e', 'mixed'):
            if path not in baseline:
                violations.append(f'{path}: new legacy E2E exceptions are forbidden')
            if 'API-24' not in owners or not (set(owners) - {'API-24'}):
                violations.append(f'{path}: legacy exception needs migration and removal owners')
    return tuple(violations)
