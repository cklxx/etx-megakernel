"""The GLM parallel-variants generator command, moved from tests/test_glm53_parallel_variants.py, through external APIs only."""
import hashlib
import json
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def variants(*argv):
    return run_command([sys.executable, '-m', 'examples.glm53.parallel_variants', *map(str, argv)], cwd=ROOT)


def test_command_generates_options_and_manifest_only(tmp_path):
    out = tmp_path/'new-options'
    run = variants('--out', out)
    assert run.returncode == 0, run.stdout
    assert f'GLM-VARIANTS options=30 cases=120 launches=0 status=uncompiled-untested out={out}' in run.stdout
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['launches_performed'] == 0 and manifest['status'] == 'uncompiled-untested'
    assert {p.name for p in out.iterdir()} == {'manifest.json'} | {e['options_file'] for e in manifest['variants']}
    for entry in manifest['variants']:
        canonical = json.dumps(json.loads((out/entry['options_file']).read_text()), sort_keys=True, separators=(',', ':'))
        assert hashlib.sha256(canonical.encode()).hexdigest() == entry['options_sha256']


def test_cli_requires_an_explicit_new_output_directory(tmp_path):
    run = variants()
    assert run.returncode == 2 and not list(tmp_path.iterdir())
    run = variants('--out', tmp_path)
    assert run.returncode == 2 and 'output already exists' in run.stdout
    assert not list(tmp_path.iterdir())
