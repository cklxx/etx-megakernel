"""Run the internal session transport with inherited terminal and environment."""
import os
from pathlib import Path
import subprocess

from etx.api.transport import SessionTransportRequest


TRANSPORT = Path(__file__).resolve().parents[1] / 'session/transport.sh'


def run_session_transport(request):
    if not isinstance(request, SessionTransportRequest):
        raise TypeError('run_session_transport requires SessionTransportRequest')
    if not Path(request.root).is_dir():
        raise ValueError(f'repository root is not a directory: {request.root}')
    # Interactive transport: ssh/rsync progress and remote output stream straight to the caller.
    completed = subprocess.run(['bash', str(TRANSPORT), *request.argv],
                               env=dict(os.environ, ETX_SESSION_ROOT=request.root))
    return completed.returncode if completed.returncode >= 0 else 128 - completed.returncode
