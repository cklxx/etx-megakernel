import fcntl, hashlib, json, shutil, subprocess
from pathlib import Path
b=Path(__file__).parent
lock=open(b.parent/'qwen-prep-heavy.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for ext in ('log','json','exit'):
    (b/f'full-suite.{ext}').rename(b/f'full-suite-reserve-stop.{ext}')
(b/'full-driver.log').rename(b/'full-reserve-stop-driver.log')
old=json.loads((b/'full-suite-reserve-stop.json').read_text()); assert old['exit']==-15 and old['stop']=='deadline/reserve'
project=b/'full-pytest/test_wheel_contains_importer_s0/project'
assert project.is_dir() and not project.is_symlink()
size=sum(p.stat().st_size for p in project.rglob('*') if p.is_file() and not p.is_symlink())
shutil.rmtree(project)
plugin=b/'bounded_archive_plugin.py'
prior=plugin.read_text(); (b/'bounded_archive_plugin-before-reserve.py').write_text(prior)
plugin.write_text('''import os
from pathlib import Path
import pytest
@pytest.fixture(autouse=True)
def bounded_archive(request,monkeypatch):
    module, name = request.module.__name__, request.node.name
    archive_case = module.endswith("test_producer_readiness") and name in (
        "test_fresh_archive_interpreter_preflight_and_failure_report",
        "test_documented_push_stage_attach_start_with_local_transport")
    packaging_case = module.endswith("test_packaging") and name == "test_wheel_contains_importer_shim_headers"
    if archive_case or packaging_case:
        monkeypatch.setattr(request.module,"ROOT",Path(os.environ["DEEPSEEK_ARCHIVE_FIXTURE"]))
''')
fixture=Path(json.loads((b/'fixture.json').read_text())['fixture_path'])
root=Path('/Users/ckl/code/.wt/etx-deepseek-two-gaps')
needed=['pyproject.toml','tests/test_packaging.py']+[str(p.relative_to(root)) for p in (root/'etx').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
for rel in needed: assert (root/rel).read_bytes()==(fixture/rel).read_bytes(),rel
record=dict(stopped_attempt='full-suite-reserve-stop.json',removed_only=str(project),reproducible_temporary_copy_bytes=size,free_bytes=shutil.disk_usage(b).free,fixture_packaging_files_verified=len(needed),plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),reason='Packaging copies the entire checkout including historical DeepSeek results. Use the same committed source fixture already used by two archive transport tests; unchanged packaging metadata, all package source/header bytes and test logic verified. No test deselection or repository source change.')
(b/'reserve-recovery.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
