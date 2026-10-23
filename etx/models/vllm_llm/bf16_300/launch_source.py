"""Runnable diagnostic kernels; original control source remains unchanged."""
from .source import probe_source


def candidate_source(pair, task, wave, row):
    s = probe_source(pair, task, wave, row)
    # Keep the original sparse 32-byte record, plus one completion record per
    # launched task/wave. Store after the retained publication barrier.
    anchor = '  ETX_RELEASE_DOMAIN(); __syncthreads();'
    s = s.replace(anchor,
        '  unsigned long long complete_begin = Bf16Probe::tick();\n' + anchor +
        '\n  unsigned long long complete_end = Bf16Probe::tick();\n'
        '  if (threadIdx.x % 64 == 0) records[1 + blockIdx.x * 16 + etx_wave()] = '
        '{complete_begin, complete_end, (unsigned)blk, (unsigned)etx_wave(), ETX_DOMAIN_ID(), BF16_PROBE_ROW};')
    return s.replace('probe.save(records + blockIdx.x, blk);', 'probe.save(records, blk);')
