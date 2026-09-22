import pytest

from etx.frontends import hip_link
from etx.ir import Graph, VerifyError, verify
from examples import moe_layer, splitk_sum


def test_examples_verify():
    g = splitk_sum.build()
    assert verify(g, splitk_sum.bindings()) == []
    b = moe_layer.bindings()
    assert [i for i in verify(moe_layer.build(), b, moe_layer.runtime(b)) if i.severity == "error"] == []


def test_count_mismatch_is_caught():
    g = splitk_sum.build()
    g.events["E"].wait_count = 3          # four producers notify each row
    with pytest.raises(VerifyError) as e:
        verify(g, splitk_sum.bindings())
    assert any(i.check == "count" for i in e.value.issues)


def test_forward_reference_is_a_cycle():
    g = Graph("cyc")
    g.etensor("E1", (4,), 1)
    g.etensor("E2", (4,), 1)
    g.call_device("a", (4,), hip_link("x.hip", "a"), in_edges={"E2": "i->i"}, out_edges={"E1": "i->i"})
    g.call_device("b", (4,), hip_link("x.hip", "b"), in_edges={"E1": "i->i"}, out_edges={"E2": "i->i"})
    with pytest.raises(VerifyError) as e:
        verify(g, {})
    assert any(i.check == "acyclic" for i in e.value.issues)


def test_runtime_tensor_must_be_ordered():
    g = Graph("rt")
    g.tensor("topk", (4, 2), role="runtime")
    g.etensor("E", (8,), wait_count="runtime", runtime_init_by="router")
    g.etensor("E_r", (1,), 4)
    g.call_device("router", (4,), hip_link("x.hip", "r"), writes=["topk"], out_edges={"E_r": "i->(0)"})
    # consumer uses topk in its out map but has no event path after router
    g.call_device("use", (4,), hip_link("x.hip", "u"), out_edges={"E": "i->topk[i,:]"})
    g.call_device("sink", (8,), hip_link("x.hip", "s"), in_edges={"E": "i->i"})
    issues = verify(g, {}, {"topk": [[0, 1], [2, 3], [4, 5], [6, 7]]}, strict=False)
    assert any(i.check == "runtime-first" for i in issues)


def test_double_writer_is_caught():
    g = splitk_sum.build()
    g.grid("final_sum").writes.append("B")
    with pytest.raises(VerifyError) as e:
        verify(g, splitk_sum.bindings())
    assert any(i.check == "coverage" for i in e.value.issues)
