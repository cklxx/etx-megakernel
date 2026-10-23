"""Migration ratchet: legacy lifecycle coverage stays visible until migrated."""
import json
from pathlib import Path
import pytest
from etx.testing import check_test_inventory


def test_all_legacy_files_have_a_reviewed_classification():
    assert check_test_inventory(Path(__file__).parent) == ()


@pytest.mark.parametrize('case', ['unclassified', 'new-exception', 'no-owner', 'removal-only', 'bad-owners', 'stale'])
def test_inventory_refuses_silent_exceptions(tmp_path, case):
    name = 'test_example.py'
    row = dict(classification='conformance', owners=[], reason='isolated contract probe')
    manifest = dict(schema_version=1, legacy_baseline=[], tests={name: row})
    if case != 'stale':
        (tmp_path / name).write_text('def test_example(): pass\n')
    if case == 'unclassified':
        manifest['tests'] = {}
    elif case == 'new-exception':
        row.update(classification='mixed', owners=['API-24'])
    elif case in ('no-owner', 'removal-only', 'bad-owners'):
        manifest['legacy_baseline'] = [name]
        row['classification'] = 'legacy-e2e'
        if case == 'removal-only':
            row['owners'] = ['API-24']
        elif case == 'bad-owners':
            row['owners'] = 'API-24'
    (tmp_path / 'api_inventory.json').write_text(json.dumps(manifest))
    assert check_test_inventory(tmp_path)
