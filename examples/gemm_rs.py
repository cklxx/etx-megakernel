"""GEMM + Reduce-Scatter across WORLD devices (paper figure 6), with real tiles
(examples/tiles/gemm_rs.hip) and a two-GPU host (examples/hosts/gemm_rs.hip).

Rank r holds A_r (M x K) and the full weight Bw (K x N), computes C_r = A_r Bw
(M x N) in 128x128 tiles, and reduce-scatter gives rank r the rows
[r*M/WORLD, (r+1)*M/WORLD) of sum_r C_r.  Every tile (i, j) on every rank
notifies the shared event E[i, j] (wait_count WORLD, system scope, fine-grained
memory); rank r's reduce-scatter tile (i, j) waits on E[i + r*rows_local, j] and
reads the corresponding C tiles of all ranks over peer access (no multimem on
AMD: design §10.4 option 1, owner-side reduction).

On MI250X the two GCDs of one card are exactly this graph with WORLD=2.
"""
from etx.frontends import hip_link
from etx.ir import Graph, Resource

T = "examples/tiles/gemm_rs.hip"
BM = BN = 128


def build(world: int = 2) -> Graph:
    g = Graph("gemm_rs")
    g.symbols = ["M", "K", "N"]          # explicit shape-scalar order: tiles read c->shape[0..2] as M, K, N
    g.tensor("Bw", ("K", "N"), role="weight")
    for r in range(world):
        g.tensor(f"A{r}", ("M", "K"), role="activation")
        g.tensor(f"C{r}", ("M", "N"), role="scratch", bytes_per_elem=4)
        g.tensor(f"D{r}", (f"M/{world}", "N"), role="activation", bytes_per_elem=4)
    g.etensor("E", (f"M/{BM}", f"N/{BN}"), wait_count=world)
    rows_local = f"(M/{BM})/{world}"
    for r in range(world):
        g.call_device(f"matmul_d{r}", (f"M/{BM}", f"N/{BN}"), hip_link(T, "etx_tile_matmul"),
                      resource=Resource(threads=256, vgpr=128, lds_bytes=2 * BM * 32 * 4), device=r,   # fp32 staging of A and B panels
                      args=[f"A{r}", "Bw", f"C{r}"], reads=[f"A{r}", "Bw"], writes=[f"C{r}"], weight_args=["Bw"],
                      out_edges={"E": "ij->ij"}, duration_us=12.0, duration_cv=0.05)
    for r in range(world):
        # reads every rank's C; the tile knows its rank from coord and the row offset symbolically
        g.call_device(f"reduce_scatter_d{r}", (rows_local, f"N/{BN}"), hip_link(T, f"etx_tile_reduce_scatter_r{r}"),
                      resource=Resource(threads=256, vgpr=64), device=r,
                      args=[f"C0", f"C1", f"D{r}"], reads=["C0", "C1"], writes=[f"D{r}"],
                      in_edges={"E": f"ij->(i+{r}*{rows_local})j"}, duration_us=6.0, duration_cv=0.3)
    return g


def bindings(**overrides) -> dict:
    b = {"M": 1024, "K": 512, "N": 512}
    b.update(overrides)
    return b


def runtime(bindings, rng=None):
    return {}
