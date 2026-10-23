"""Explicit substitution backends; neither produces a build or device certificate."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

from etx.api.execution import ProcessRequest, ProcessResult
from etx.contracts.execution import read_process_result


@dataclass
class _ScriptedExecutor:
    request: ProcessRequest
    output: bytes
    returncode: int
    used: bool = False

    def execute(self, request, directory):
        if self.used or request != self.request:
            raise ValueError('scripted executor request mismatch or exhausted response')
        self.used = True
        (directory / 'output.log').write_bytes(self.output)
        status = 'complete' if self.returncode == 0 else 'deadline' if self.returncode == 124 else 'failed'
        return ProcessResult(request.identity, 'scripted', status, self.returncode,
                             hashlib.sha256(self.output).hexdigest())


@dataclass(frozen=True)
class _RecordedExecutor:
    source: Path

    def execute(self, request, directory):
        original = read_process_result(self.source, request=request)
        payload = (self.source / 'output.log').read_bytes()
        if hashlib.sha256(payload).hexdigest() != original.output_sha256:
            raise ValueError('recorded output changed during replay')
        (directory / 'output.log').write_bytes(payload)
        return ProcessResult(request.identity, 'recorded', original.status, original.returncode,
                             original.output_sha256, original.events)


def scripted_executor(request, *, output=b'', returncode=0):
    """One exact expected request, one explicit response; never executes argv."""
    if not isinstance(request, ProcessRequest) or type(output) is not bytes or type(returncode) is not int:
        raise ValueError('scripted executor requires a request, byte output and integer return code')
    return _ScriptedExecutor(request, output, returncode)


@dataclass(frozen=True)
class _ProtocolFault:
    fault: str

    def simulate(self, planned, *, seed):
        if self.fault == 'exception':
            raise RuntimeError('injected simulator failure')
        from etx.sim import simulate
        result = simulate(planned, seed=seed)
        result.makespan_us = float('inf')
        return result


def protocol_fault_executor(fault):
    """Supported fault injection inside the simulator seam, not private patches."""
    if fault not in ('exception', 'nonfinite'):
        raise ValueError('unknown protocol executor fault')
    return _ProtocolFault(fault)


def recorded_executor(directory):
    """Replay only checksum-checked evidence bound to the exact request."""
    return _RecordedExecutor(Path(directory).absolute())
