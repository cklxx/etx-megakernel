"""Synchronous decoding through a caller-owned runtime descriptor."""
def run(runtime, state, count=1, *, eos=-1):
    """Decode up to count tokens using a Runtime and caller-owned Descriptor.

    Returns the token list. Runtime errors retain any completed tokens on the
    exception, as in Runtime.decode. Keep state and its callbacks alive.
    """
    return runtime.decode(state, count, eos=eos)
