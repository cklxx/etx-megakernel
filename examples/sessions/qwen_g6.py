"""G6 on Qwen3-30B-A3B: the compiler's `coherent_l1` mode (ETX_COHERENCE=l1) against the kept build, under G6's rule
(docs/INVENTORY-SYSTEM-2026-10-13.md section 4.1, written before the code). DeepSeek's half was measured in the
2026-10-13 paid hour (examples/dsv2lite/results/2026-10-13-hour, pairs_l1: +330 us vs default, spread 13 us).

Order: preflight, both builds and the hw_bench build (CPU, in parallel) -> moerows, the Q2 decision microbenchmark
(GPU minutes, no model input: row ownership contiguous vs interleaved on Qwen's moe_up phase, cold, checksummed; plus
moegather as its contiguous cross-check) -> certify kept -> certify kept_l1 -> pairs_kept_l1 (10 invocations). Every GPU job reads what it runs, so a failed build or certificate skips what depends on it, never
runs it. The runner seals the record on the box before any fetch.

The box needs the prepared Qwen inputs (model with golden_c1024.txt, HF config) named by the JSON at
$ETX_QWEN_G6_CONFIG: {"model": ..., "hf": ..., "golden_sha256": ...}; the builds (examples/vllm_llm/build.sh) also need
the vLLM source tree at $VLLM (default ~/vllm) and, on an image without HIP headers, ETX_ROCM_HEADERS.
Run: tools/session.sh stage <ip> && ETX_QWEN_G6_CONFIG=... tools/session.sh start <ip> examples/sessions/qwen_g6.py
"""
MAX_SESSION_SECONDS = 3600
JOBS = []
OP = ".venv/bin/python -u -m examples.vllm_llm.g6_session"
SUMMARY = [r"^(PREFLIGHT|BUILD|CERTIFIED|VERDICT|PAIRS) "]


def job(name, operation, *, lane="cpu", reads=(), writes=(), timeout=900, cmd=None, summary=SUMMARY):
    JOBS.append(dict(name=name, lane=lane, gpu=0 if lane == "gpu" else None, reads=list(reads), writes=list(writes),
                     timeout=timeout, quiet_s=180, cmd=cmd or f"{OP} {operation}", summary=summary))


job("preflight", "preflight", writes=["inputs.json"], timeout=300)
job("build_hw_bench", None, writes=["b_hw/hw_bench"], timeout=600,
    cmd='mkdir -p "$ETX_SESS_LOGDIR/b_hw" && hipcc -O3 -std=c++17 --offload-arch=gfx942 bench/hw/hw_bench.hip '
        '-o "$ETX_SESS_LOGDIR/b_hw/hw_bench"')
# exit 4 = INCOMPLETE (a checksum mismatch: no decision), 3 = a hang; the DECISION line carries PASS / NO-GO
job("moerows", None, lane="gpu", reads=["b_hw/hw_bench"], writes=["moerows.txt"], timeout=600,
    summary=[r"^(RESULT|DECISION) moe"],
    cmd='set -o pipefail; "$ETX_SESS_LOGDIR/b_hw/hw_bench" moerows | tee "$ETX_SESS_LOGDIR/moerows.txt" && '
        '"$ETX_SESS_LOGDIR/b_hw/hw_bench" moegather')
for arm in ("kept", "kept_l1"):
    job(f"build_{arm}", f"build {arm}", reads=["inputs.json"], writes=[f"b_{arm}/arm.json"], timeout=1500)   # never on an unstaged box
for arm in ("kept", "kept_l1"):
    job(f"certify_{arm}", f"certify {arm}", lane="gpu", reads=["inputs.json", f"b_{arm}/arm.json"],
        writes=[f"certificate_{arm}/gate.log"], timeout=1200)
job("pairs_kept_l1", "pairs kept_l1", lane="gpu", writes=["pairs_kept_l1.json"], timeout=3000,
    reads=["inputs.json", "certificate_kept/gate.log", "certificate_kept_l1/gate.log"])
