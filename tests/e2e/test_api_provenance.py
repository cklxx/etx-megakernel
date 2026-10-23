"""Copied-source evidence is scoped content, never fabricated Git provenance."""
import pytest
from etx.external import source_snapshot, verify_source_snapshot


def test_snapshot_has_explicit_scope_and_unknown_git_identity(tmp_path):
    (tmp_path / 'code').mkdir()
    (tmp_path / 'code' / 'model.py').write_text('value = 1\n')
    (tmp_path / 'outside.txt').write_text('not in scope')
    receipt = source_snapshot(tmp_path, paths=('code',))
    assert receipt['commit'] is None
    assert receipt['dirty'] is None
    assert receipt['kind'] == 'scoped-filesystem-snapshot'
    assert set(receipt['source_sha256']) == {'code/model.py'}
    assert verify_source_snapshot(tmp_path, receipt) == receipt
    (tmp_path / 'outside.txt').write_text('still not in scope')
    assert verify_source_snapshot(tmp_path, receipt) == receipt


@pytest.mark.parametrize('mutation', ['change', 'new', 'remove', 'symlink'])
def test_snapshot_rejects_content_or_inventory_changes(tmp_path, mutation):
    (tmp_path / 'code').mkdir()
    path = tmp_path / 'code' / 'model.py'
    path.write_text('old')
    receipt = source_snapshot(tmp_path, paths=('code',))
    if mutation == 'change':
        path.write_text('new')
    elif mutation == 'new':
        (path.parent / 'new.py').write_text('new')
    elif mutation == 'remove':
        path.unlink()
    else:
        path.unlink()
        path.symlink_to(tmp_path / 'outside')
    with pytest.raises(ValueError):
        verify_source_snapshot(tmp_path, receipt)


@pytest.mark.parametrize('paths', [(), ('missing',), ('../outside',), 'code'])
def test_snapshot_refuses_implicit_or_missing_scope(tmp_path, paths):
    with pytest.raises(ValueError):
        source_snapshot(tmp_path, paths=paths)
