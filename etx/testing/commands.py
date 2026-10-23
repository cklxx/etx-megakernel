"""Run a command for an E2E test through etx.external.execute: the subset of subprocess.run tests need."""
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from etx.api.execution import ProcessRequest


def run_command(argv, cwd=None, env=None, check=False, timeout=120, **_):
    """Execute `argv` with explicit cwd/environment; return returncode and the merged stdout+stderr log.

    The tool resolves on the given environment's PATH (so test stubs win), the full log is kept on disk,
    and a deadline maps to returncode 124 like timeout(1).
    """
    from etx.contracts.execution import execute
    environment = dict(os.environ if env is None else env)
    argv = [str(arg) for arg in argv]
    tool = argv[0] if os.path.isabs(argv[0]) else shutil.which(argv[0], path=environment.get('PATH'))
    if tool is None:
        raise FileNotFoundError(argv[0])
    root = Path(__file__).resolve().parents[2]
    output = Path(tempfile.mkdtemp(prefix='etx-e2e-command-')) / 'process'
    result = execute(ProcessRequest((tool, *argv[1:]), str(cwd or root), float(timeout), environment=environment,
                                    silence_s=float(timeout)), output=output)
    text = (output / 'output.log').read_text(errors='replace')
    code = result.returncode if result.returncode is not None else 124
    if check:
        assert code == 0, text
    # One merged log: both names carry it, so stdout/stderr substring checks keep their meaning.
    return SimpleNamespace(returncode=code, stdout=text, stderr=text, log=str(output / 'output.log'))
