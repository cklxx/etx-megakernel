"""Synthetic CPU MoE acceptance fixtures; not model or GPU validation.

The vectorized float32 implementation and independent scalar float64 reference
share only generated weights/inputs and explicit semantic parameters. No launch
metadata is used by the reference. These tiny fixtures test the acceptance gate.
"""
from __future__ import annotations

import math
import numpy as np


def output_errors(actual, expected, written, *, atol=2e-5, rtol=2e-5):
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")
    actual, expected, written = np.asarray(actual), np.asarray(expected), np.asarray(written)
    errors = []
    if actual.shape != expected.shape or written.shape != expected.shape or not expected.size:
        return ["output/reference/write-mask shape mismatch or empty output"]
    if written.dtype != np.bool_ or not written.all():
        errors.append("output not completely written")
    if not np.isfinite(expected).all():
        errors.append("invalid nonfinite reference")
    if not np.isfinite(actual).all():
        errors.append("nonfinite output")
    if not errors and not np.allclose(actual, expected, atol=atol, rtol=rtol):
        errors.append("numerical mismatch")
    return errors


def fixture(*, experts=8, top_k=2, hidden=8, intermediate=12, seed=17, renormalize=True):
    if not 1 <= top_k <= experts or hidden <= 0 or intermediate <= 0:
        raise ValueError("invalid MoE dimensions or top-k")
    rng = np.random.default_rng(seed)
    def weight(shape):
        return rng.normal(0, 0.15, shape).astype(np.float32)
    return {"x": weight(hidden), "norm": np.ones(hidden, np.float32),
            "router": weight((experts, hidden)), "gate": weight((experts, intermediate, hidden)),
            "up": weight((experts, intermediate, hidden)), "down": weight((experts, hidden, intermediate)),
            "top_k": top_k, "renormalize": renormalize, "eps": 1e-5}


def vectorized(data):
    x = data["x"]
    norm = x * (1 / np.sqrt(np.mean(x * x) + data["eps"])) * data["norm"]
    logits = data["router"] @ norm
    scores = np.exp(logits - logits.max())
    scores /= scores.sum()
    selected = np.lexsort((np.arange(len(scores)), -scores))[:data["top_k"]]
    weights = scores[selected]
    if data["renormalize"]:
        weights = weights / weights.sum()
    gates = data["gate"][selected] @ norm
    up = data["up"][selected] @ norm
    hidden = gates / (1 + np.exp(-gates)) * up
    output = np.full(x.shape, np.nan, dtype=np.float32)
    written = np.zeros(x.shape, dtype=bool)
    value = x + np.sum((data["down"][selected] @ hidden[..., None])[..., 0] * weights[:, None], axis=0)
    output[:] = value
    written[:] = True
    return output, written, selected.tolist()


def scalar_reference(data):
    """Full MoE block reference: scalar reductions, own routing, no candidate helpers."""
    x = list(map(float, data["x"]))
    h = len(x)
    rms = math.sqrt(math.fsum(v * v for v in x) / h + data["eps"])
    normalized = [v * float(data["norm"][i]) / rms for i, v in enumerate(x)]
    logits = [math.fsum(float(a) * b for a, b in zip(row, normalized)) for row in data["router"]]
    exps = [math.exp(v - max(logits)) for v in logits]
    probs = [v / math.fsum(exps) for v in exps]
    selected = sorted(range(len(probs)), key=lambda i: (-probs[i], i))[:data["top_k"]]
    denom = math.fsum(probs[i] for i in selected) if data["renormalize"] else 1.0
    residual = [[] for _ in x]
    for expert in selected:
        middle = []
        for gate_row, up_row in zip(data["gate"][expert], data["up"][expert]):
            gate = math.fsum(float(a) * b for a, b in zip(gate_row, normalized))
            up = math.fsum(float(a) * b for a, b in zip(up_row, normalized))
            middle.append(gate / (1 + math.exp(-gate)) * up)
        for i, row in enumerate(data["down"][expert]):
            residual[i].append(probs[expert] / denom * math.fsum(float(a) * b for a, b in zip(row, middle)))
    return np.asarray([v + math.fsum(r) for v, r in zip(x, residual)]), selected


def acceptance_suite():
    """All failures are expected to be caught; no timings are generated."""
    records = []
    for experts, top_k in ((8, 2), (128, 8)):
        for renormalize in (False, True):
            data = fixture(experts=experts, top_k=top_k, renormalize=renormalize)
            reference, reference_ids = scalar_reference(data)
            actual, written, selected = vectorized(data)
            base = output_errors(actual, reference, written)
            if selected != reference_ids:
                base.append("routing mismatch")
            records.append({"case": f"E{experts}-K{top_k}-renorm{renormalize}",
                            "expected_valid": True, "errors": base,
                            "output": actual.tolist(), "reference": reference.tolist(),
                            "written": written.tolist(), "expert_ids": selected})
            for fault in ("nan", "inf", "hole", "wrong_value", "wrong_topk"):
                bad, mask = actual.copy(), written.copy()
                if fault in ("nan", "inf"):
                    bad[-1] = float(fault)
                elif fault == "hole":
                    # A finite stale value is as invalid as a NaN sentinel.
                    bad[-1], mask[-1] = reference[-1], False
                elif fault == "wrong_value":
                    bad[-1] += 1
                else:
                    bad, mask, _ = vectorized({**data, "top_k": 1})
                errors = output_errors(bad, reference, mask)
                records.append({"case": f"E{experts}-renorm{renormalize}-{fault}",
                                "expected_valid": False, "errors": errors})
    return {"kind": "synthetic_cpu_numerical", "dtype": "float32_vs_scalar_float64",
            "gpu_measurement": False, "model_validation": False,
            "passed": all(bool(r["errors"]) != r["expected_valid"] for r in records), "cases": records}
