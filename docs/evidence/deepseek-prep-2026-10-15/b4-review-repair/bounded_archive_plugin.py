import os
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
