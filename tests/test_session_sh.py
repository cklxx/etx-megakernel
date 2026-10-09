from pathlib import Path
import subprocess


def test_session_sh():
    script = Path(__file__).resolve().with_suffix('.sh')
    subprocess.run(['bash', str(script)], cwd=script.parent.parent, check=True)
