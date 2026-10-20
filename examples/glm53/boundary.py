"""Two-GPU ETX transport probe at the proposed GLM mHC residual shape.

Bit-copy and token-return test only: no GLM layers, weights or numerical claims.
"""
from etx.frontends import hip_link
from etx.ir import Graph, Resource

TILES = 'examples/glm53/boundary_tiles.hip'


def build():
    g = Graph('glm53_boundary_transport')
    g.symbols = ['B']
    n = 'B*16384'  # H=4096, four residual streams; uint16 payload
    chunks = 'B*64'
    for name in ('input', 'mailbox', 'received'):
        g.tensor(name, (n,), role='activation', bytes_per_elem=2)
    for name in ('token', 'returned'):
        g.tensor(name, ('B',), role='activation', bytes_per_elem=4)
    g.etensor('ready', (chunks,), wait_count=1)
    g.etensor('copied', (chunks,), wait_count=1)
    g.etensor('feedback', ('B',), wait_count=1)
    resource = Resource(threads=256, vgpr=32)
    g.call_device('send_state', (chunks,), hip_link(TILES, 'glm53_send_state'), device=0,
                  resource=resource, args=['input', 'mailbox'], reads=['input'], writes=['mailbox'],
                  out_edges={'ready': 'i->i'})
    g.call_device('receive_state', (chunks,), hip_link(TILES, 'glm53_receive_state'), device=1,
                  resource=resource, args=['mailbox', 'received'], reads=['mailbox'], writes=['received'],
                  in_edges={'ready': 'i->i'}, out_edges={'copied': 'i->i'})
    g.call_device('send_token', ('B',), hip_link(TILES, 'glm53_send_token'), device=1,
                  resource=resource, args=['received', 'token'], reads=['received'], writes=['token'],
                  in_edges={'copied': '*'}, out_edges={'feedback': 'i->i'})
    g.call_device('receive_token', ('B',), hip_link(TILES, 'glm53_receive_token'), device=0,
                  resource=resource, args=['token', 'returned'], reads=['token'], writes=['returned'],
                  in_edges={'feedback': 'i->i'})
    return g


def bindings(**overrides):
    result = dict(B=1)
    result.update(overrides)
    if result['B'] not in (1, 2, 4, 8):
        raise ValueError('boundary batch must be 1, 2, 4 or 8')
    return result


def runtime(bindings, rng=None):
    return {}


def oracle(batch, step):
    """Exact bit oracle, includes request offset and step to expose alias/stale data."""
    bindings(B=batch)
    if step < 0:
        raise ValueError('negative step')
    values = [(i + 17 * step) % 65535 for i in range(batch * 16384)]
    feedback = [values[r * 16384] + values[(r+1) * 16384 - 1] for r in range(batch)]
    return values, feedback
