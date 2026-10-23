"""Synchronous request serving over the runtime decode interface."""
from etx.runtime.execution import run


def serve(runtime, requests, count=1, *, eos=-1):
    """Yield one token list per Descriptor in a synchronous request iterable.

    Requests run in input order, using the same runtime. Backend errors propagate
    to the caller. This iterator provides no network transport or batching policy.
    """
    for state in requests:
        yield run(runtime, state, count, eos=eos)
