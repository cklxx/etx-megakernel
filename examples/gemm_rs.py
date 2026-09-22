"""GEMM + Reduce-Scatter across WORLD devices (paper figure 6).

Every device computes its full C tile grid and notifies the shared event
E[i, j] (wait_count WORLD, system scope); device d's reduce-scatter tiles wait
on the rows it owns.  On MI250X the two GCDs of one card are exactly this
graph with WORLD=2; on MI300X it is two cards over xGMI.  No multimem is
assumed: the reduce tile reads the peers' C slices over P2P and reduces
locally (design §10.4 option 1).
"""
from etx.frontends import hip_link
from etx.ir import Graph, Resource

T = "examples/tiles/gemm_rs.hip"


def build(world: int = 2) -> Graph:
    g = Graph("gemm_rs")
    g.tensor("A", ("M", "K"), role="activation")
    g.tensor("Bw", ("K", "N"), role="weight")
    g.tensor("C", ("M", "N"), role="scratch")
    g.tensor("D", ("M/%d" % world, "N"), role="activation")
    g.etensor("E", ("M/128", "N/128"), wait_count=world)
    rows_local = f"(M/128)/{world}"
    for d in range(world):
        g.call_device(f"matmul_d{d}", ("M/128", "N/128"), hip_link(T, "etx_tile_matmul"),
                      resource=Resource(threads=256, vgpr=128, lds_bytes=32768), device=d,
                      args=["A", "Bw", "C"], reads=["A", "Bw"], writes=[] , weight_args=["Bw"],
                      out_edges={"E": "ij->ij"}, duration_us=12.0, duration_cv=0.05)
    for d in range(world):
        g.call_device(f"reduce_scatter_d{d}", (rows_local, "N/128"), hip_link(T, "etx_tile_reduce_scatter"),
                      resource=Resource(threads=256, vgpr=96, lds_bytes=16384), device=d,
                      args=["C", "D"], reads=["C"], writes=[],
                      in_edges={"E": f"ij->(i+{d}*{rows_local})j"}, duration_us=6.0, duration_cv=0.3)
    return g


def bindings(**overrides) -> dict:
    b = {"M": 2048, "K": 4096, "N": 1024}
    b.update(overrides)
    return b


def runtime(bindings, rng=None):
    return {}
