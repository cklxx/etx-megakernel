"""ctypes interface to libetx_rt; importing this module does not load a GPU SDK.

Descriptors and callbacks are caller-owned. Keep the descriptor, backend,
callbacks and their state alive until the synchronous call returns. A backend
can be native or a CPU stub; Python exceptions must be caught in callbacks and
reported as a nonzero return code.
"""
import ctypes as C


class Descriptor(C.Structure):
    pass


StepHook = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(Descriptor))
RankHook = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint32, C.c_double)


class Backend(C.Structure):
    _fields_ = [(name, RankHook) for name in ('reset', 'launch', 'wait')]


Descriptor._fields_ = [
    ('abi_version', C.c_uint32), ('struct_size', C.c_uint32),
    ('backend', C.POINTER(Backend)), ('context', C.c_void_p),
    ('ranks', C.c_uint32), ('reserved', C.c_uint32),
    ('timeout_s', C.c_double), ('position', C.c_int64),
    ('token', C.c_int32), ('next_token', C.c_int32),
    ('prepare', StepHook), ('finish', StepHook),
]


def descriptor(backend, *, context=None, ranks=1, timeout_s=5.0,
               position=0, token=0, prepare=None, finish=None):
    """Build ABI v1 state, retaining the callback owners through ctypes."""
    return Descriptor(1, C.sizeof(Descriptor), C.pointer(backend), context,
                      ranks, 0, timeout_s, position, token, token,
                      prepare or StepHook(), finish or StepHook())


class Runtime:
    def __init__(self, library):
        self.library = C.CDLL(str(library))
        self.library.etx_step.argtypes = [C.POINTER(Descriptor)]
        self.library.etx_step.restype = C.c_int
        self.library.etx_decode.argtypes = [C.POINTER(Descriptor), C.c_uint64,
                                          C.c_int32, C.POINTER(C.c_int32),
                                          C.POINTER(C.c_uint64)]
        self.library.etx_decode.restype = C.c_int
        self.library.etx_status_string.argtypes = [C.c_int]
        self.library.etx_status_string.restype = C.c_char_p

    def step(self, state):
        status = self.library.etx_step(C.byref(state))
        if status:
            raise RuntimeError(self.library.etx_status_string(status).decode())
        return state.token

    def decode(self, state, count, eos=-1):
        if count < 0:
            raise ValueError('count must be nonnegative')
        tokens = (C.c_int32 * count)()
        completed = C.c_uint64()
        status = self.library.etx_decode(C.byref(state), count, eos,
                                         tokens, C.byref(completed))
        result = list(tokens[:completed.value])
        if status:
            error = RuntimeError(self.library.etx_status_string(status).decode())
            error.tokens = result
            raise error
        return result
