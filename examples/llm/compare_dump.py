#!/usr/bin/env python3
"""Compare a GPU per-layer dump (run --dump) with the numpy reference's (ref_numpy.py --dump).
  python examples/llm/compare_dump.py gpu.bin ref.npz
GPU rows: residual entering layer L (0..L-1) and the final residual; the reference stores the residual after
each layer, so GPU row L+1 corresponds to reference row L.
"""
import sys
import numpy as np

gpu_path, ref_path = sys.argv[1], sys.argv[2]
with open(gpu_path, "rb") as f:
    n, h = np.frombuffer(f.read(8), np.int32)
    g = np.frombuffer(f.read(), np.float32).reshape(n, h)
r = np.load(ref_path)["layers"]
print(f"gpu {g.shape} ref {r.shape}")
worst = None
for L in range(min(n - 1, r.shape[0])):
    a, b = g[L + 1], r[L]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rel = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))
    exact = int(np.sum(a == b))
    flag = "" if rel < 1e-3 else "  <-- first divergence" if worst is None else ""
    if rel >= 1e-3 and worst is None:
        worst = L
    print(f"after layer {L:2d}: cosine {cos:.6f}  max rel {rel:.2e}  exact {exact}/{h}{flag}")
print("all layers agree" if worst is None else f"first layer that differs: {worst}")
