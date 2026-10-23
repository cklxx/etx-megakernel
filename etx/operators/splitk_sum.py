"""Paper figure 3: split-K row sum. The smallest graph with a real event.

  partial_sum (n, 4) --"ij->i"--> E(n, wait 4) --"i->i"--> final_sum (n,)
"""
from etx.frontends import hip_link
from etx.ir import Graph, Resource


def build() -> Graph:
    g = Graph("splitk_sum")
    g.tensor("A", ("n*32", 128), role="activation")
    g.tensor("B", ("n*32", 4), role="scratch", bytes_per_elem=4)
    g.tensor("C", ("n*32",), role="activation", bytes_per_elem=4)
    g.etensor("E", ("n",), wait_count=4)
    g.call_device("partial_sum", ("n", 4), hip_link("examples/tiles/splitk.hip", "etx_tile_partial_sum"),
                  resource=Resource(threads=256, lds_bytes=0, vgpr=64),
                  args=["A", "B"], reads=["A"], writes=["B"], out_edges={"E": "ij->i"},
                  duration_us=2.0, duration_cv=0.1, bytes_per_tile=32 * 32 * 2)
    g.call_device("final_sum", ("n",), hip_link("examples/tiles/splitk.hip", "etx_tile_final_sum"),
                  resource=Resource(threads=256, lds_bytes=0, vgpr=32),
                  args=["B", "C"], reads=["B"], writes=["C"], in_edges={"E": "i->i"},
                  duration_us=0.8, duration_cv=0.05)
    return g


def bindings(**overrides) -> dict:
    b = {"n": 64}
    b.update(overrides)
    return b


def runtime(bindings, rng=None):
    return {}
