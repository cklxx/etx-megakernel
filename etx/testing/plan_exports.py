"""Independent compile-instance oracle for historical plan-export conformance."""
import json
from pathlib import Path
import sys

from etx.external import ProcessRequest, execute


ROOT = Path(__file__).resolve().parents[2]


def plan_export_instances(model, *, output, environment=None):
    """Compute exports from load_dims/instances, not the CLI output under test.

    A fresh CPU child preserves import-time configuration isolation. Full output
    and process receipts are retained; this does not compile or run device code.
    """
    clean = {'PATH': '/usr/bin:/bin'}  # No inherited settings or credentials in process records.
    script = (
        'import json,sys; from examples.llm.model import load_dims; '
        'from examples.vllm_llm.model import instances; '
        "print(json.dumps(sorted({p['export'] for p in instances(load_dims(sys.argv[1]))})))"
    )
    result = execute(ProcessRequest(
        (sys.executable, '-u', '-c', script, f'examples/llm/configs/{model}.json'),
        str(ROOT), 30, environment={**clean, **(environment or {})}), output=output)
    text = (Path(output) / 'output.log').read_text()
    if result.status != 'complete' or result.returncode != 0:
        raise AssertionError(f'compile-instance conformance failed: {result}\n{text}')
    return tuple(json.loads(text))
