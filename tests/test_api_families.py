"""The scaffold census is executable: the scoreboard's N/14 is this test's result."""
from pathlib import Path

from etx.testing.families import FAMILIES, census, entry_violations


# Raise only with the change that moves a set onto the scaffold; lowering it is a regression.
COUNTED = {'S1', 'S2', 'S2a', 'S2b', 'S3', 'S4', 'S4a', 'S4b', 'S4c', 'S5', 'S6', 'S7', 'S8', 'S9'}


def test_fourteen_sets_nine_top_level_and_five_nested():
    assert len(FAMILIES) == 14 and len({f.id for f in FAMILIES}) == 14
    assert sum(f.kind == 'top' for f in FAMILIES) == 9
    assert sum(f.kind == 'nested' for f in FAMILIES) == 5


def test_counted_sets_are_exactly_the_recorded_ones():
    assert {row['id'] for row in census() if row['counted']} == COUNTED


def test_counted_sets_are_exercised_by_the_installed_probe():
    from etx.testing import packaging
    for family in FAMILIES:
        if family.id in COUNTED:
            assert family.installed
            assert all(key in packaging._PROBE for key in family.installed), family.id


def test_census_rejects_private_imports_mechanisms_and_shell_bodies(tmp_path):
    family = FAMILIES[0].__class__('X', 'top', 'probe', ('entry.py', 'entry.sh'), installed=('k',))
    (tmp_path / 'entry.py').write_text(
        'from etx.external import plan\n'
        'def f():\n    from etx.passes import compile_graph\n'
        'import subprocess\nfrom examples.glm53 import model\nfrom . import sibling\n')
    (tmp_path / 'entry.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                                       'exec python -m etx.tools.cli "$@"\nhipcc -o run x.hip\n')
    reasons = entry_violations(family, tmp_path)
    assert reasons == ['entry.py:3: private import etx.passes', 'entry.py:4: private mechanism subprocess',
                       'entry.py:5: example implementation import examples.glm53',
                       'entry.py:6: relative import .', 'entry.sh:4: shell implementation']
    (tmp_path / 'entry.py').write_text('from etx.external import plan\nimport argparse\n')
    (tmp_path / 'entry.sh').write_text('set -euo pipefail\nexec python -m etx.tools.cli "$@"\n')
    assert entry_violations(family, tmp_path) == []
    (tmp_path / 'entry.sh').unlink()
    assert entry_violations(family, tmp_path) == ['entry.sh: missing']


def test_every_entry_point_named_by_the_review_exists():
    root = Path(__file__).resolve().parents[1]
    assert [e for f in FAMILIES for e in f.entries if not (root / e).is_file()] == []
