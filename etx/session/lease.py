"""Optional persistent session budget and nonblocking runner lock."""
import fcntl
import json
import math
from pathlib import Path
import time


class Lease:
    def __init__(self, path, seconds, now=None):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('positive finite lease seconds required')
        path=Path(path).expanduser()
        path.parent.mkdir(parents=True,exist_ok=True)
        self.file=path.open('a+')
        try:
            fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise ValueError(f'runner already active: {path}; use session.sh status/fetch') from None
        try:
            self.file.seek(0);raw=self.file.read()
            now=time.time() if now is None else now
            if raw:
                data=json.loads(raw)
                if set(data)!= {'started','deadline','seconds'} or data['seconds'] != seconds:
                    raise ValueError('session lease changed; cannot extend or reset budget')
            else:
                data=dict(started=now,deadline=now+seconds,seconds=seconds)
                self.file.write(json.dumps(data)+'\n');self.file.flush()
                import os
                os.fsync(self.file.fileno())
            self.remaining=data['deadline']-now
            if not 0 < self.remaining <= seconds:
                raise ValueError(f'session budget expired or clock changed: {path}; fetch evidence, stop GPU work')
        except BaseException:
            self.close();raise

    def close(self):
        self.file.close()
