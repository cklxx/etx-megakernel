"""Six serial vLLM default rounds on the running MI300X; about 13 minutes.

Uses the existing image and model. Run with --logdir ~/sess for the round logs.
"""
JOBS = []


def job(**kw):
    JOBS.append(kw)


LLM = "$HOME/llm"
M = "qwen3-30b-a3b"
S = "${ETX_SESS_LOGDIR:-$HOME/sess}"
IMAGE = "vllm/vllm-openai-rocm:v0.27.1"


def vllm(name, aiter=False):
    return ("sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g "
            "--security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$GPU "
            + ("-e VLLM_ROCM_USE_AITER=1 " if aiter else "")
            + f"-v $HOME:$HOME -w $HOME/etx --entrypoint python3 {IMAGE} examples/llm/vllm_bench.py "
            f"--model {LLM}/{M}/hf --golden {LLM}/{M}/golden_c1024.txt --label default --batch 1 --tokens 32 "
            f"--repeats 1 --json {S}/{name}.json")


for rnd in range(1, 7):
    name = f"vs_r{rnd}"
    job(name=name, gpu=0, after=[f"vs_r{rnd - 1}"] if rnd > 1 else [],
        timeout=420, quiet_s=240, cmd=vllm(name), summary=[r"^VLLM|^RESULT"])

job(name="vs_summary", lane="cpu", after="vs_r6", timeout=120,
    cmd="""python3 - "${ETX_SESS_LOGDIR:-$HOME/sess}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

paths = sorted(Path(sys.argv[1]).glob("vs_r*.json"))
assert [p.name for p in paths] == [f"vs_r{i}.json" for i in range(1, 7)], paths
values = []
for path in paths:
    value = json.loads(path.read_text())["median_ms"]
    values.append(value)
    print(f"{path.stem} median_ms={value:.6f}")
lo, hi, med = min(values), max(values), statistics.median(values)
spread = hi - lo
print(f"min_ms={lo:.6f} max_ms={hi:.6f} median_ms={med:.6f} spread_ms={spread:.6f}")
print(f"VSTAB {lo:.6f} {hi:.6f} {spread:.6f} {med:.6f}")
PY""", summary=[r"."])
