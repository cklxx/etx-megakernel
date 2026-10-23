"""Public declaration fixtures; no compiler objects or implicit execution."""
from etx.api import KernelSpec, kernel


def fixture(name='handoff', *, devices=1, batch=1):
    if name != 'handoff':
        raise ValueError(f'unknown fixture {name!r}')
    if type(devices) is not int or devices not in (1, 2, 8):
        raise ValueError('handoff fixture supports devices 1, 2 or 8')
    if type(batch) is not int or batch not in (1, 2, 4, 8):
        raise ValueError('handoff fixture supports batch 1, 2, 4 or 8')
    tensors, events, grids = [], [], []
    for rank in range(devices):
        tensor, event = f'payload_{rank}', f'ready_{rank}'
        tensors.append(dict(name=tensor, shape=[batch], dtype='f32', bytes_per_elem=4))
        events.append(dict(name=event, shape=[batch]))
        grids.append(dict(name=f'producer_{rank}', grid=[batch], device=rank,
                          body=dict(kind='builtin', symbol='noop'), writes=[tensor],
                          out_edges={event: 'i->i'}, duration_us=1.0))
    for rank in range(devices):
        source = (rank - 1) % devices
        grids.append(dict(name=f'consumer_{rank}', grid=[batch], device=rank,
                          body=dict(kind='builtin', symbol='noop'), reads=[f'payload_{source}'],
                          in_edges={f'ready_{source}': 'i->i'}, duration_us=1.0))
    return kernel(KernelSpec('handoff', 'graph',
                            dict(tensors=tensors, events=events, grids=grids)))
