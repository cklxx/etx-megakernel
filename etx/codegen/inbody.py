"""Opt-in diagnostic source overlay; no changes to ordinary emitter output.

Producer atomics have no exact architectural completion timestamp: retain a
bracket, not a fabricated cycle. Never delay a consumer for diagnostic stores.
"""
import re


def instrument(source):
    source = source.replace('#include "etx/primitives.h"',
        '#include "etx/primitives.h"\n#if defined(ETX_INBODY) && ETX_INBODY\n#include "etx/inbody.h"\n#endif', 1)
    start = source.index('static __device__ __forceinline__ void etx_run_task(')
    end = source.index('#ifndef ETX_MIN_WG_PER_CU', start)
    prefix, suffix, source = source[:start], source[end:], source[start:end]
    def insert(anchor, addition, before=False):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError(f'inbody anchor count: {anchor}')
        guarded = '\n#if defined(ETX_INBODY) && ETX_INBODY\n' + addition + '\n#endif\n'
        source = source.replace(anchor, guarded + anchor if before else anchor + guarded)

    # Each generated wait target already uses the original abort/backoff loop.
    # Capture before acquire. For multiple waits retain the LAST observed
    # success (the final poll that makes this task able to proceed).
    source = re.sub(r'(?m)^#define ETX_TARGET\(idx\) (etx_lok &= etx_wait_DEVICE_mirror\(.*|etx_wait_\w+\(.*)$',
        lambda m: '#if defined(ETX_INBODY) && ETX_INBODY\n#define ETX_TARGET(idx) do { ' +
        m[1] + '; if (p.inbody_poll) p.inbody_poll[tid] = etx_inbody_tick(); } while (0)\n#else\n' +
        m[0] + '\n#endif', source)
    # Stamp only the producer that actually takes an event to zero, preserving
    # dynamic pushes and the existing per-domain last-arriver protocol.
    def arrival(m):
        line = m[0]
        begin = line.index('etx_arrive_')
        end = line.index(' == 0', begin) if ' == 0' in line else len(line)
        expr = line[begin:end]
        word = re.search(r'(\d+) \+ \(idx\)', expr)[1]
        new = line[:begin] + f'ETX_INBODY_ARRIVE(p, {word} + (idx), tid, {expr})' + line[end:]
        return '#if defined(ETX_INBODY) && ETX_INBODY\n' + new + '\n#else\n' + line + '\n#endif'
    source = re.sub(r'(?m)^#define ETX_TARGET\(idx\) .*etx_arrive.*$', arrival, source)
    insert('  int sym = -1;',
           '  if (threadIdx.x == 0 && p.inbody_poll) p.inbody_poll[tid] = 0;', before=True)
    insert('  etx_call_body(sym, &ctx);   // phase 2: the only call site of each body',
           '  uint64_t ib_return = p.inbody_wave ? etx_inbody_tick() : 0;')
    insert('  if (tr && threadIdx.x == 0) { tr[2] = (uint64_t)ETX_TIMER(); tr[3] = worker; }',
           '  if (p.inbody_wave && (threadIdx.x % ETX_WAVE_SIZE) == 0) {\n'
           '    size_t iw = ((size_t)tid * (ETX_THREADS / ETX_WAVE_SIZE) + threadIdx.x / ETX_WAVE_SIZE) * 2;\n'
           '    p.inbody_wave[iw] = ib_return; p.inbody_wave[iw + 1] = ib_drained;\n  }')
    # No barrier added; the per-wave stamp brackets the RETAINED release.
    insert('  ETX_RELEASE_DOMAIN();   // every wave drains its own stores before the barrier: thread 0\'s release only covers wave 0',
           '  uint64_t ib_drained = p.inbody_wave ? etx_inbody_tick() : 0;')
    return prefix + source + suffix
