"""The Qwen3-30B-A3B arms: ONE table (etx/session/experiment.py, component D of docs/core/SESSION-ARCH-2026-10-13.md).

The generic-path counterpart of examples/dsv2lite/arms.py. The G6 session (examples/sessions/qwen_g6.py), the local
device build check and every re-derivation (`python -m etx.session.experiment verify <dir> examples.vllm_llm.arms`)
read this table; nothing else names an arm's build or rule.

`kept` is the kept recipe (kept_build.py: KEPT_ENV + BASE_FLAGS). `kept_l1` is the same build with the compiler's
coherence mode switched to `l1` (ETX_COHERENCE, docs/core/INVENTORY-SYSTEM-2026-10-13.md section 4.1): one compiler switch,
no per-model change. Its rule is G6's, copied verbatim from section 4.1, where it was written before any Qwen code.

  python -m examples.vllm_llm.arms shell <arm>     the arm's build as shell assignments
  python -m examples.vllm_llm.arms env <arm>       its plan/selection switches, space-separated (rocm_build_check.sh E)
  python -m examples.vllm_llm.arms cflags <arm>    its compile flags (rocm_build_check.sh ETX_CFLAGS)
"""
from __future__ import annotations

from pathlib import Path
import re
import shlex
import sys

from etx.session.experiment import Arm, Rule, ns
from examples.vllm_llm.kept_build import BASE_FLAGS, KEPT_ENV

CONDITION = "one MI300X, Qwen3-30B-A3B, bf16, batch 1, context 1024, 32 tokens, device time"
G6_TEXT = ("G6 (docs/core/INVENTORY-SYSTEM-2026-10-13.md 4.1): KEEP if every pair is faster than kept, median >= 30 us and "
           "median > 2 x spread; otherwise NO-GO")

ARMS = {a.name: a for a in [
    Arm("kept", references=()),
    Arm("kept_l1", env=dict(ETX_COHERENCE="l1"), order=("kept", "kept_l1"), references=("kept",),
        rule=Rule("device_ms", "kept", 30, text=G6_TEXT, labels=("KEEP", "NO-GO"))),
]}


def build_env(arm: Arm) -> dict:
    """The environment examples/vllm_llm/build.sh is run with for this arm (the kept recipe plus the arm's switches)."""
    if arm.builder != "etx":
        raise ValueError(f"{arm.name} uses {arm.builder}, not the ETX build")
    return {**KEPT_ENV, **arm.env, "ETX_HIPCC_FLAGS": " ".join((BASE_FLAGS, *arm.flags))}


def full_result(log: str) -> dict:
    """What every timed invocation must show: one fused run, 32/32 HF-greedy tokens in every repeat, no failed check,
    no debug synchronisation; device and wall time kept as the host's decimal text (the rule works in integer ns)."""
    from examples.vllm_llm.nvref_report import correctness
    from examples.vllm_llm.pairs import parse as parse_runs
    fields = correctness(log, "qwen-arm")
    runs = parse_runs(log.splitlines()).runs
    if len(runs) != 1 or runs[0].arm != "fused":
        raise ValueError(f"a timed invocation must hold exactly one fused run, found {[r.arm for r in runs]}")
    result = dict(device_ms=fields["median_ms"], wall_ms=fields["wall_ms"], tokens=32)
    if not all(ns(result[k]) > 0 for k in ("device_ms", "wall_ms")):
        raise ValueError("invalid timing")
    return result


parse = full_result


def checks(d: Path):
    """Besides its pairs, a Qwen record carries one import certificate per build (certificate_<arm>/): re-validated."""
    from examples.vllm_llm import import_certificate as cert
    for c in sorted(d.glob("certificate_*")):
        def check(c=c):
            try:
                cert.validate(c, timing=False)
            except Exception as exc:                       # the validator refuses with several exception types
                return False, f"DIFF certificate {c.name}: {exc}"
            return True, f"OK   certificate {c.name}"
        yield check


def archives(d: Path) -> list[Path]:
    """No Qwen result keeps files outside its record."""
    return []


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 2 or a[0] not in ("shell", "env", "cflags") or a[1] not in ARMS:
        raise SystemExit(f"usage: python -m examples.vllm_llm.arms shell|env|cflags <{'|'.join(ARMS)}>")
    env = build_env(ARMS[a[1]])
    flags = env.pop("ETX_HIPCC_FLAGS")
    if a[0] == "cflags":
        print(flags)
    elif a[0] == "env":
        if any(re.search(r"\s", v) for v in env.values()):
            raise SystemExit(f"{a[1]}: a switch value holds whitespace; rocm_build_check.sh splits E on it")
        print(" ".join(f"{k}={v}" for k, v in sorted(env.items())))
    else:
        for k, v in sorted({**env, "ETX_HIPCC_FLAGS": flags}.items()):
            print(f"{k}={shlex.quote(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
