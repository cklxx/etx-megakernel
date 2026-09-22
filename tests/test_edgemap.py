import pytest

from etx.ir import EdgeMap, eval_dim


def test_affine_reduce():
    m = EdgeMap.parse("ij->i")
    assert m.targets((3, 2), (8,)) == [(3,)]


def test_affine_identity_and_broadcast():
    assert EdgeMap.parse("ij->ij").targets((1, 2), (4, 4)) == [(1, 2)]
    assert EdgeMap.parse("h->bh").targets((1,), (3, 4)) == [(0, 1), (1, 1), (2, 1)]


def test_index_arithmetic():
    m = EdgeMap.parse("ij->(i/2)j")
    assert m.targets((5, 1), (4, 4)) == [(2, 1)]
    assert EdgeMap.parse("bj->(0)").targets((7, 2), (1,)) == [(0,)]


def test_runtime_range_and_gather():
    rt = {"indptr": [0, 2, 5], "topk": [[3, 1], [0, 2]]}
    assert EdgeMap.parse("i->range(indptr[i], indptr[i+1])").targets((1,), (5,), rt) == [(2,), (3,), (4,)]
    assert EdgeMap.parse("i->topk[i,:]").targets((0,), (4,), rt) == [(3,), (1,)]


def test_all_map():
    assert EdgeMap.parse("*").targets((0,), (2, 2)) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_bad_maps():
    with pytest.raises(ValueError):
        EdgeMap.parse("ij->(k)j")
    with pytest.raises(ValueError):
        EdgeMap.parse("ij i")


def test_symbols_in_index_expressions():
    m = EdgeMap.parse("ij->(i+R*(M/128)/2)j")   # R and M are symbols (single lowercase letters are coordinates)
    assert m.targets((1, 0), (16, 8), None, {"M": 2048, "R": 1}) == [(9, 0)]
    c = m.to_c(["c0", "c1"], "shp", "CB", symbols={"M": "p.shape[0]", "R": "p.shape[1]"})
    assert "p.shape[0]" in c and "p.shape[1]" in c


def test_to_c_shapes():
    c = EdgeMap.parse("ij->(i/2)j").to_c(["t.coord[0]", "t.coord[1]"], "shp", "CB")
    assert "CB(" in c and "shp[1]" in c
    c = EdgeMap.parse("h->bh").to_c(["t.coord[0]"], "shp", "CB")
    assert "for (int _b0" in c
    c = EdgeMap.parse("i->range(indptr[i], indptr[i+1])").to_c(["c0"], "shp", "CB", {"indptr": "ip"})
    assert "ip[(c0)]" in c and "ip[(c0)+1]" in c


def test_eval_dim():
    assert eval_dim("cdiv(S, 128)", {"S": 300}) == 3
    assert eval_dim("B*H", {"B": 2, "H": 3}) == 6
    with pytest.raises(KeyError):
        eval_dim("B*H", {"B": 2})
