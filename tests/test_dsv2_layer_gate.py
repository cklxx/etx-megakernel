"""The host's real layer gate, run as portable C++ with only the HIP copy stubbed.

This is not a GPU build. Extracting the existing loop keeps the diagnostic and
its consecutive-prefix verdict under the same synthetic request boundaries.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from etx.session import batch
from etx.models.dsv2lite import session as ins
from examples.dsv2lite import inplace_session as cli

ROOT = Path(__file__).resolve().parents[1]
SWITCH = "ETX_DS_LAYER_METRICS"
ARM = "alld4r0l1kvsmw"


@pytest.fixture(scope="module")
def host_gate(tmp_path_factory):
    source = (ROOT / "examples/dsv2lite/host.hip").read_text()
    start = source.index("  for (int r = 0; r < R; ++r) {\n    Req& q = reqs[r];\n    if (q.golden_hidden.empty()) continue;")
    gate = source[start:source.index("  hst.report_phases", start)]
    directory = tmp_path_factory.mktemp("host_gate")
    cpp = directory / "gate.cpp"
    cpp.write_text(r'''
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define HIP_CHECK(expr) expr
constexpr int hipMemcpyDeviceToHost = 0;
void hipMemcpy(void* out, const void* in, size_t n, int) { std::memcpy(out, in, n); }
int main(int argc, char** argv) {
  const int R = std::atoi(argv[1]);
  struct { int layers = 4, hidden = 2, first_k_dense = 1; } d;
  struct Req {
    struct { float* layer_dump = nullptr; } act;
    std::vector<float> golden_hidden;
    int layers_ok = -1;
  };
  std::vector<Req> reqs(R);
  std::vector<std::vector<float>> actual(R, {1, 2, 1, 2, 1, 2, 1, 2});
  for (int r = 0; r < R; ++r) {
    reqs[r].golden_hidden = actual[r];
    reqs[r].act.layer_dump = actual[r].data();
  }
  if (std::atoi(argv[2])) actual[0][3] = -2; // layer 1 fails; layers 2 and 3 pass again
''' + gate + r'''
  for (int r = 0; r < R; ++r)
    std::printf("BATCHREQ r=%d tokens=32/32 layers=%d warmup=32/32 interval_median_ms=3.50000 device_median_ms=3.400000\n",
                r, reqs[r].layers_ok);
  std::printf("BATCH requests=%d tokens=32 step_median_ms=3.600000\n", R);
}
''')
    compiler = Path("/opt/homebrew/opt/llvm/bin/clang++")
    binary = directory / "gate"
    subprocess.run([str(compiler) if compiler.exists() else "c++", "-std=c++17", str(cpp), "-o", str(binary)],
                   check=True, capture_output=True, text=True)

    def run(*, requests=2, fail=False, switch=None):
        env = {k: v for k, v in os.environ.items() if k != SWITCH}
        if switch is not None:
            env[SWITCH] = switch
        return subprocess.run([str(binary), str(requests), str(int(fail))], env=env,
                              check=True, capture_output=True, text=True).stdout

    return run


def summary(log):
    """Preserve the exact bytes of the lines consumed by the existing batch parser."""
    return "".join(x + "\n" for x in log.splitlines() if x.startswith(("BATCHREQ ", "BATCH "))).encode()


def classification(log, requests=2):
    try:
        return json.dumps(batch.etx_result(log, requests, layers=4), sort_keys=True).encode()
    except batch.Incomplete as exc:
        return str(exc).encode()


@pytest.mark.parametrize("fail", [False, True])
def test_switch_off_keeps_legacy_output_and_parsed_summary_bytes(host_gate, fail):
    old_count = 1 if fail else 4
    expected_summary = (
        f"BATCHREQ r=0 tokens=32/32 layers={old_count} warmup=32/32 interval_median_ms=3.50000 device_median_ms=3.400000\n"
        "BATCHREQ r=1 tokens=32/32 layers=4 warmup=32/32 interval_median_ms=3.50000 device_median_ms=3.400000\n"
        "BATCH requests=2 tokens=32 step_median_ms=3.600000\n"
    ).encode()
    expected = (f"  r0 consecutive layers within the gate: {old_count} of 4\n"
                "  r1 consecutive layers within the gate: 4 of 4\n").encode() + expected_summary
    baseline = host_gate(fail=fail)
    assert baseline.encode() == expected
    assert summary(baseline) == expected_summary
    for disabled in ("0", "true", ""):
        off = host_gate(fail=fail, switch=disabled)
        assert off.encode() == baseline.encode()
        assert summary(off) == summary(baseline)
        assert classification(off) == classification(baseline)


@pytest.mark.parametrize("fail", [False, True])
def test_switch_on_prints_every_boundary_without_changing_batch_verdict(host_gate, fail):
    before = host_gate(fail=fail)
    visible = host_gate(fail=fail, switch="1")
    lines = re.findall(r"^LAYERGATE r=(\d+) layer=(\d+) rel=(\S+) cosine=(\S+) pass=([01])$", visible, re.M)
    assert [(int(r), int(layer)) for r, layer, *_ in lines] == [(r, layer) for r in range(2) for layer in range(4)]
    for r, layer, rel, cosine, passed in lines:
        bad = fail and (r, layer) == ("0", "1")
        assert float(rel) == pytest.approx(2 if bad else 0)
        assert float(cosine) == pytest.approx(-0.6 if bad else 1)
        assert passed == ("0" if bad else "1")
    assert summary(visible) == summary(before)
    assert classification(visible) == classification(before)
    if fail:
        assert classification(visible) == b"request 0: 1 layers within the gate (need 4)"


def test_single_request_retains_legacy_lines_and_adds_all_layer_metrics(host_gate):
    before = host_gate(requests=1, fail=True)
    visible = host_gate(requests=1, fail=True, switch="1")
    metrics = [x for x in visible.splitlines(keepends=True) if x.startswith("LAYERGATE ")]
    assert len(metrics) == 4
    assert [int(re.search(r"layer=(\d+)", line)[1]) for line in metrics] == list(range(4))
    assert "pass=0" in metrics[1] and "pass=1" in metrics[2]
    assert "".join(x for x in visible.splitlines(keepends=True) if not x.startswith("LAYERGATE ")) == before
    assert summary(visible) == summary(before)
    assert classification(visible, 1) == classification(before, 1)


@pytest.mark.parametrize("request_id,fail", [(5, False), (7, True)])
def test_layers_command_uses_own_reference_and_passes_switch_through_clean_env(tmp_path, monkeypatch, capsys, request_id, fail):
    logs, data, reference = tmp_path / "logs", tmp_path / "data", tmp_path / f"r{request_id}"
    for name, value in (("ETX_SESS_LOGDIR", logs), ("ETX_DSV2_DATA", data), ("ETX_DSV2_FLEET", tmp_path / "fleet")):
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv("ETX_UNRELATED_OVERRIDE", "must not leak")
    seen = []
    host_log = ("run 0 (warm-up): 32/32 tokens match\n"
                f"  consecutive layers within the gate: {3 if fail else 27} of 27\n"
                "per-token latency (embed -> argmax on device): median 3.000 ms  mean 3.000 ms; "
                "step wall (launch to end) 3.100 ms/token\n"
                "tokens: 32/32 match HF greedy\n")
    host_log += "".join(f"LAYERGATE r=0 layer={layer} rel=0 cosine=1 pass={int(not (fail and layer == 3))}\n"
                        for layer in range(27))

    def fake_process(command, *, sink, env, **kwargs):
        seen.append((list(map(str, command)), env))
        sink(host_log)
        return 0

    monkeypatch.setattr(ins.proc, "run", fake_process)
    command = ["layers", ARM, "--reference-dir", str(reference)]
    if fail:
        with pytest.raises(ValueError, match="27/27 layer gate"):
            cli.main(command)
    else:
        cli.main(command)
    assert len(seen) == 1
    args, env = seen[0]
    assert str(logs / f"b_{ARM}" / "run") in args
    assert args[args.index("--weights") + 1] == str(data / "weights.bin")
    for option, filename in (("--cache", "fleet_cache.bin"), ("--golden", "golden_tokens.txt"), ("--hidden", "golden_hidden.bin")):
        assert args[args.index(option) + 1] == str(reference / filename)
    assert "--req-dir" not in args and env[SWITCH] == "1"
    assert "ETX_UNRELATED_OVERRIDE" not in env
    raw_logs = list(logs.glob("*.raw.log"))
    assert len(raw_logs) == 1 and raw_logs[0].read_text().endswith(host_log)
    assert capsys.readouterr().out.count("LAYERGATE ") == 27


@pytest.mark.parametrize("args", [["layers", ARM], ["layers", "x8", "--reference-dir", "r5"],
                                  ["pairs", ARM, "--reference-dir", "r5"]])
def test_layers_cli_refuses_missing_reference_multi_request_and_ignored_reference(monkeypatch, args):
    monkeypatch.setattr(ins, "Session", lambda: pytest.fail("invalid diagnostic reached session construction"))
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("switch", [None, "0", "1"])
def test_batch_gate_passes_only_opted_in_metrics_through_clean_env(tmp_path, monkeypatch, switch):
    from etx.models.dsv2lite import batch_session as bs
    for name in ("ETX_SESS_LOGDIR", "ETX_DSV2_DATA", "ETX_DSV2_FLEET"):
        monkeypatch.setenv(name, str(tmp_path))
    if switch is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, switch)
    session = bs.BatchSession()
    original_run = session.run
    invocation_options, process_envs = [], []

    def spy_run(name, command, **kwargs):
        invocation_options.append(kwargs)
        return original_run(name, command, **kwargs)

    def fake_process(command, *, sink, env, **kwargs):
        process_envs.append(env)
        sink("".join(f"BATCHREQ r={r} tokens=32/32 layers=27 warmup=32/32 interval_median_ms=3.50000 device_median_ms=3.400000\n"
                     for r in range(2)) + "BATCH requests=2 tokens=32 step_median_ms=3.600000\n")
        return 0

    monkeypatch.setattr(session, "run", spy_run)
    monkeypatch.setattr(ins.proc, "run", fake_process)
    session.gate("x2")
    assert len(invocation_options) == len(process_envs) == 1
    if switch == "1":
        assert invocation_options == [dict(env={SWITCH: "1"})]
        assert process_envs[0][SWITCH] == "1"
    else:
        assert invocation_options == [{}]
        assert SWITCH not in process_envs[0]
    assert json.loads((tmp_path / "bgate_x2.json").read_text())["passed"] is True
