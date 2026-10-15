import os
from pathlib import Path
import pytest
@pytest.fixture(autouse=True)
def bounded_archive(request,monkeypatch):
    if request.module.__name__.endswith("test_producer_readiness") and request.node.name in ("test_fresh_archive_interpreter_preflight_and_failure_report","test_documented_push_stage_attach_start_with_local_transport"):
        monkeypatch.setattr(request.module,"ROOT",Path(os.environ["DEEPSEEK_ARCHIVE_FIXTURE"]))
