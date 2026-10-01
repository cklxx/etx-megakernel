"""examples.vllm_llm.folds: the fold decision is derived from the model's shapes and reproduces what was measured.

A fold deletes a one-block producer phase and pays for it with rematerialised work in every consumer task. Both
outcomes were measured on Qwen3-30B-A3B on 2026-10-01: the input-norm fold wins (3.75 us/layer against 5.9 for the
separate phase, rounds 25 and 27) and the MoE top-k fold loses badly (14.1 against about 8.6, rounds 7 and 26). If
the cost model stops reproducing those, it has drifted from the measurements and must be re-fitted, not overridden.
"""
from pathlib import Path

from examples.llm.model import load_dims
from examples.vllm_llm import folds

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "examples/llm/configs"
CU, NX = 256, 8


def _by_name(cfg: str):
    return {f.name: f for f in folds.candidates(load_dims(str(CFG / cfg)), CU, NX)}


def test_reproduces_the_two_measured_outcomes():
    f = _by_name("qwen3-30b-a3b.json")
    assert f["in_norm"].decide() is True, "the input-norm fold was measured to win by 2.15 us/layer"
    assert f["topk"].decide() is False, "the top-k fold was measured to lose by 5.5 us/layer"


def test_predicted_costs_are_close_to_the_measurements():
    f = _by_name("qwen3-30b-a3b.json")
    # in_norm: measured fold 3.75, keep 5.9. The bandwidth term carries this one, so 20% is the tolerance the
    # norm row itself showed; a wider miss means the rate constant or the remat byte count is wrong.
    assert abs(f["in_norm"].fold_us() - 3.75) / 3.75 < 0.25
    assert abs(f["in_norm"].keep_us() - 5.9) / 5.9 < 0.10
    # topk: measured fold 14.1, keep about 8.6. fold_us is where the hot-spot coefficient was fitted, so this
    # asserts the fit still holds rather than testing an independent prediction.
    assert abs(f["topk"].fold_us() - 14.1) / 14.1 < 0.10
    assert abs(f["topk"].keep_us() - 8.6) / 8.6 < 0.10


def test_dense_models_fold_both_norms_and_have_no_topk_candidate():
    # A dense in_norm re-reads one residual row instead of TOPK expert rows, so the fold is far cheaper than for the
    # MoE and the decision is not close. No experts means no top-k candidate at all.
    for cfg in ("qwen2.5-1.5b.json", "qwen3-8b.json"):
        f = _by_name(cfg)
        assert set(f) == {"in_norm", "post_norm"}, cfg
        assert all(c.decide() for c in f.values()), cfg
        assert f["in_norm"].fold_us() < f["in_norm"].keep_us() / 2, cfg


def test_the_topk_decision_turns_on_the_shapes_not_on_the_phase():
    """Fewer consuming tasks must be able to flip the top-k fold, or the model is just hardcoding the answer."""
    d = load_dims(str(CFG / "qwen3-30b-a3b.json"))
    big = {f.name: f for f in folds.candidates(d, CU, NX)}["topk"]
    small = folds.Fold("topk", producer_tasks=NX, producer_body_us=big.producer_body_us,
                       consumer_tasks=16, workers=CU, remat_bytes=32, shared_bytes=32, serial_steps=2 * 6)
    assert big.decide() is False and small.decide() is True
