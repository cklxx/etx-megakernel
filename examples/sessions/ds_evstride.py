"""A/B of ETX_EV_STRIDE (event counters one cache line each) on DeepSeek-V2-Lite, one and two requests, alternated on GPU 0;
a traced one-request run with the stride on GPU 1 for the ledger. Prediction (docs/deepseek/MODEL-dsv2.md H6, 2026-09-27): about
5 XCD-local hand-offs per MoE layer lose ~2.3 us each to the shared line, so up to ~11 us per layer, ~0.3 ms per token.
Assumes the environment, references and weights of ds_model.py are in place (same VM)."""
from examples.dsv2lite.acquire import ACQUIRE, RELAXED   # D1: safe sc1 unless an arm names RELAXED
FLEET = "~/fleet-mi300x"
JOBS = []
X2 = "ETX_DSV2_REQUESTS=2 ETX_DSV2_COUPLE=none ETX_RELAY_PLAN=off ETX_FLEET_PATCH=gemv_noinline"
X2F = f"{ACQUIRE} -DETX_DS_REQUESTS=2 -DETX_SLOT_BY_CU"
B = {"x1": ("", f"{ACQUIRE}"), "x1s": ("", f"{ACQUIRE} -DETX_EV_STRIDE=32"),
     "x2": (X2, X2F), "x2s": (X2, X2F + " -DETX_EV_STRIDE=32")}
for b, (env, flags) in B.items():
    JOBS.append(dict(name=f"build_{b}", lane="cpu", timeout=1800,
                     cmd=f"{env} ETX_OUT=$HOME/sess2/b_{b} bash examples/dsv2lite/build.sh ~/fleet-mi300x {flags} > $HOME/sess2/b_{b}.buildlog 2>&1; test -x $HOME/sess2/b_{b}/run"))
prev = None
for rnd in (1, 2):
    for b in ("x1", "x1s", "x2", "x2s"):
        args = " --r1-dir build_r1" if b.startswith("x2") else ""
        JOBS.append(dict(name=f"t{rnd}_{b}", gpu=0, reads=[f"build_{b}"], timeout=600, cwd=FLEET,
                         cmd=f"$HOME/sess2/b_{b}/run --tokens 32 --context 1024 --repeat 2{args}",
                         summary=[r"per-token", r"tokens:", r"consecutive", r"^RESULT"]))
        prev = f"t{rnd}_{b}"
JOBS.append(dict(name="trace_x1s", gpu=1, reads=["build_x1s"], timeout=600, cwd=FLEET,
                 cmd="ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess2/x1s_trace.bin $HOME/sess2/b_x1s/run --tokens 8 --context 1024 && "
                     "cp $HOME/sess2/b_x1s/plan.json $HOME/sess2/x1s_plan.json", summary=[r"per-token", r"tokens:"]))
JOBS.append(dict(name="ledger_x1s", lane="cpu", reads=["trace_x1s"], timeout=300,
                 cmd="PYTHONPATH=. .venv/bin/python -m examples.dsv2lite.ledger --etx $HOME/sess2/x1s_plan.json $HOME/sess2/x1s_trace.bin "
                     "--label 'x1 stride 32' --fleet-src ~/fleet-mi300x/src", summary=[r"^== |handoff|layer "]))
