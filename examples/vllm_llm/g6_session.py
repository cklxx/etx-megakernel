"""The operations of the Qwen G6 session (examples/sessions/qwen_g6.py): one per runner job, run on the box.

  python -m examples.vllm_llm.g6_session preflight          frozen model + golden checks, writes inputs.json
  python -m examples.vllm_llm.g6_session build <arm>        examples/vllm_llm/build.sh with the registry's environment
  python -m examples.vllm_llm.g6_session certify <arm>      the same-build import certificate (positions 0 and 1040)
                                                            and one 32/32 + CHECK-DSL gate run, into certificate_<arm>/
  python -m examples.vllm_llm.g6_session pairs <arm>        five interleaved rounds (etx.session.experiment.measure),
                                                            decided by the registry's rule, into pairs_<arm>.json

Everything an arm is (build switches, order, references, rule) comes from examples/vllm_llm/arms.py. The record is the
runner's log directory ($ETX_SESS_LOGDIR); the box's inputs come from the JSON at $ETX_QWEN_G6_CONFIG
(model, hf, golden_sha256). bf16 only: a quantized or non-bf16 checkpoint is refused before anything runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from etx.session import experiment, proc
from examples.vllm_llm import arms as registry

ROOT = Path(__file__).resolve().parents[2]
QWEN3_30B_A3B = dict(model_type="qwen3_moe", num_hidden_layers=48, hidden_size=2048, num_key_value_heads=4,
                     num_experts=128, num_experts_per_tok=8)
INVOCATION_S = 240          # one host invocation: weight load (1-2 min for 16 GB+) plus 32 tokens, with margin


def record() -> Path:
    return Path(os.environ.get("ETX_SESS_LOGDIR", ".")).resolve()


def config() -> dict:
    return json.loads(Path(os.environ["ETX_QWEN_G6_CONFIG"]).read_text())


def build_dir(arm: str) -> Path:
    return record() / f"b_{arm}"


def run(name: str, argv: list, env=None, deadline_s=INVOCATION_S) -> str:
    """One logged command: full output to <record>/<name>.raw.log, a deadline, refusal on any nonzero exit."""
    path = record() / f"{name}.raw.log"
    with path.open("x") as out:
        out.write("# command " + json.dumps([str(a) for a in argv]) + "\n")
        out.flush()
        rc = proc.run([str(a) for a in argv], sink=out, deadline_s=deadline_s, poll_s=5, log=out,
                      env=dict(os.environ, PYTHONUNBUFFERED="1", **(env or {})), cwd=ROOT)
    text = path.read_text(errors="replace")
    if rc:
        raise RuntimeError(f"{name}: NOT COMPLETE rc={rc} ({'deadline' if rc == proc.DEADLINE_RC else 'failed'}), "
                           f"full log {path}")
    return text


def preflight() -> None:
    from examples.vllm_llm import import_certificate as cert
    c = config()
    hf = json.loads((Path(c["hf"]) / "config.json").read_text())
    wrong = {k: hf.get(k) for k, v in QWEN3_30B_A3B.items() if hf.get(k) != v}
    if wrong:
        raise ValueError(f"not the frozen Qwen3-30B-A3B shape: {wrong}")
    if hf.get("torch_dtype", hf.get("dtype")) != "bfloat16" or hf.get("quantization_config") not in (None, {}):
        raise ValueError("bf16 weights required: a quantized or non-bf16 checkpoint is refused")
    golden = Path(c["model"]) / "golden_c1024.txt"
    if cert.sha(golden) != c["golden_sha256"]:
        raise ValueError("the independent HF golden changed")
    (record() / "inputs.json").write_text(json.dumps(dict(c, golden_sha256_verified=True), indent=2) + "\n")
    print(f"PREFLIGHT OK Qwen3-30B-A3B bf16, golden {c['golden_sha256'][:16]}", flush=True)


def build(arm: str) -> None:
    from examples.vllm_llm import import_certificate as cert
    a, m, b = registry.ARMS[arm], Path(config()["model"]), build_dir(arm)
    b.mkdir()
    run(f"build_{arm}", ["bash", ROOT / "examples/vllm_llm/build.sh", m],
        env=dict(registry.build_env(a), ETX_OUT=str(b)), deadline_s=1200)
    identity = cert.verified_build(b, m)
    if not identity["check_enabled"]:
        raise ValueError(f"{arm}: a CHECK build is required (the certificate runs --check)")
    (b / "arm.json").write_text(json.dumps(dict(arm=arm, env=registry.build_env(a)), indent=2) + "\n")
    print(f"BUILD {arm} OK {b}", flush=True)


def host(arm: str, name: str, args: list) -> str:
    """The host through the certificate's --run-build, which refuses a binary that is not the built one."""
    from examples.vllm_llm import import_certificate as cert
    b, m = build_dir(arm), Path(config()["model"])
    text = run(name, [sys.executable, "-u", "-m", "examples.vllm_llm.import_certificate", "--run-build", b,
                      "--model", m, "--", *args])
    cert.stamped(text, cert.verified_build(b, m))
    return text


def certify(arm: str) -> None:
    from examples.vllm_llm import import_certificate as cert
    archive = record() / f"certificate_{arm}"
    archive.mkdir()
    cert.export_build(build_dir(arm), Path(config()["model"]), archive)
    for pos in (0, 1040):
        (archive / f"check_p{pos}.txt").write_text(host(arm, f"{arm}_check_p{pos}",
                                                        ["--check", "48", "--check-pos", str(pos)]))
    cert.validate(archive, timing=False)
    gate = host(arm, f"{arm}_gate", ["--repeat", "1", "--check-dsl"])
    if "CHECK-DSL OK" not in gate:
        raise ValueError(f"{arm}: CHECK-DSL missing")
    registry.parse(gate)                                   # 32/32 in every repeat, one fused run
    (archive / "gate.log").write_text(gate)
    print(f"CERTIFIED {arm}: import certificate at positions 0 and 1040, 32/32, CHECK-DSL OK", flush=True)


def pairs(arm: str) -> None:
    a = registry.ARMS[arm]
    for b in a.order:
        if not (record() / f"certificate_{b}").is_dir():
            raise ValueError(f"{arm}: build {b} has no certificate in this record")
    rounds = experiment.measure(a, lambda b, name: registry.parse(host(b, name, ["--repeat", "1"])))
    out = experiment.report(a, rounds, registry.CONDITION)
    (record() / f"pairs_{arm}.json").write_text(json.dumps(out, indent=2) + "\n")
    v, d = out["verdict"], out["decision"]
    print(f"VERDICT arm={arm} {v['outcome']} rule={v['rule']!r}", flush=True)
    print(f"PAIRS arm={arm} vs {d['ref']} median {d['median_us']} us spread {d['spread_us']} us "
          f"every_positive={int(d['every_positive'])}", flush=True)


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    ops = dict(preflight=(preflight, 0), build=(build, 1), certify=(certify, 1), pairs=(pairs, 1))
    if not a or a[0] not in ops or len(a) - 1 != ops[a[0]][1] or (len(a) == 2 and a[1] not in registry.ARMS):
        raise SystemExit(__doc__)
    ops[a[0]][0](*a[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
