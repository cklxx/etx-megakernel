"""python -m examples.vllm_llm.decode <model> [--hf DIR] [--work DIR] [--machine gfx942] [--only STEPS] [--dry-run]

One entry point for the GQA models (docs/core/PLAN-generic-decoder-2026-10-08.md section 7, item 8). <model> is a known
name (qwen3-8b, qwen3-30b-a3b, mixtral-8x7b) or the path of a Hugging Face config.json. Every model goes through the
same code: the config, then the GQA check, then the variant selection (etx.select), then three steps:

  prep   examples/vllm_llm/prep.py --src <hf> --out <work> --maxlen 1072   (packs the weights; needs torch)
  build  examples/vllm_llm/build.sh <work>                                  (needs hipcc and VLLM=<vLLM sources>)
  run    build/vl/<name>/run <work> build/vl/<name>/vl_plan.txt --repeat 2  (needs an MI300X)

This file has no branch on a model name. The known names only select a bundled config. The planner and model.py
read the same config, so the printed selection is the selection that model.py builds (tests/test_decode_entry.py).
DeepSeek-V2-Lite uses MLA, not GQA. It stays on examples/dsv2lite, and this command refuses its config.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/llm/configs"
KNOWN = ("qwen3-8b", "qwen3-30b-a3b", "mixtral-8x7b")      # the three GQA models of section 7
MAXLEN = 1072                                              # as examples/vllm_llm/vm_run.sh
MLA_KEYS = ("kv_lora_rank", "q_lora_rank")                 # DeepSeek-V2 attention (MLA)


class NotGQA(ValueError):
    pass


def resolve(model: str, hf: str | None = None) -> tuple[str, Path]:
    """(name, config path). A known name uses <hf>/config.json when it exists, else the bundled config."""
    if model in KNOWN:
        if hf and (Path(hf) / "config.json").is_file():
            return model, Path(hf) / "config.json"
        return model, CONFIGS / f"{model}.json"
    p = Path(model)
    if p.is_dir():
        p = p / "config.json"
    if not p.is_file():
        raise FileNotFoundError(f"{model}: not a known model ({', '.join(KNOWN)}) and not a config.json")
    if p.name != "config.json":
        return p.stem, p
    d = p.parent                                       # vm_run.sh layout: ~/llm/<name>/hf/config.json
    return (d.parent.name if d.name == "hf" and d.parent.name else d.name), p


def check_gqa(config: Path):
    """The Dims of a GQA decoder. Raises NotGQA for an MLA config or for NH % NKV != 0."""
    raw = json.loads(Path(config).read_text())
    if any(raw.get(k) for k in MLA_KEYS):
        raise NotGQA(f"{config}: MLA attention ({'/'.join(MLA_KEYS)}); use examples/dsv2lite for DeepSeek-V2-Lite")
    from examples.llm.model import load_dims
    d = load_dims(str(config))
    if d.NH % d.NKV:
        raise NotGQA(f"{config}: {d.NH} query heads do not divide into {d.NKV} kv heads")
    return d


def select(config: Path, machine: str = "gfx942"):
    from etx.select import plan
    return plan(str(config), machine)


def steps(name: str, hf: Path, work: Path) -> list[tuple[str, list[str]]]:
    """The commands of the three steps, in order. build.sh names its output after the work directory."""
    out = ROOT / "build/vl" / work.name
    py = str(ROOT / ".venv/bin/python") if (ROOT / ".venv/bin/python").exists() else sys.executable
    return [("prep", [py, str(ROOT / "examples/vllm_llm/prep.py"), "--src", str(hf), "--out", str(work),
                      "--maxlen", str(MAXLEN)]),
            ("build", ["bash", str(ROOT / "examples/vllm_llm/build.sh"), str(work)]),
            ("run", [str(out / "run"), str(work), str(out / "vl_plan.txt"), "--repeat", "2"])]


def golden(name: str) -> Path | None:
    """The bundled golden token file of the model, if the repository has one (examples/llm/results)."""
    g = ROOT / f"examples/llm/results/golden_c1024_{name}.txt"
    return g if g.is_file() else None


def describe(d, p, name: str | None = None) -> str:
    lines = [f"model {name or d.name}: H {d.H}, NH {d.NH}, NKV {d.NKV} (G {d.G}), HD {d.HD}, layers {d.L}, "
             + (f"MoE E {d.E} top-{d.TOPK} MI {d.MI}" if d.moe else f"dense INTER {d.INTER}")
             + f", q/k norm {int(d.qk_norm)}",
             f"selection: budget {p.budget}, cost {p.cost:.2f} us per layer (placeholder costs until GPU data)"]
    for k, v in p.phases.items():
        lines.append(f"  {k:<10} {v.export:<20} {v.kind}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m examples.vllm_llm.decode", description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--hf", help="Hugging Face model directory (default ~/llm/<name>/hf)")
    ap.add_argument("--work", help="packed model directory (default ~/llm/<name>)")
    ap.add_argument("--machine", default="gfx942")
    ap.add_argument("--only", default="prep,build,run", help="comma-separated steps to run")
    ap.add_argument("--dry-run", action="store_true", help="print the selection and the commands, run nothing")
    a = ap.parse_args(argv)
    home = Path(os.path.expanduser("~"))
    stem = a.model if a.model in KNOWN else None
    hf = Path(a.hf) if a.hf else (home / "llm" / stem / "hf" if stem else None)
    name, config = resolve(a.model, str(hf) if hf else None)
    hf = hf or config.parent
    work = Path(a.work) if a.work else home / "llm" / name
    try:
        d = check_gqa(config)
    except NotGQA as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    p = select(config, a.machine)
    print(describe(d, p, name))
    want = [s for s in a.only.split(",") if s]
    todo = [(k, cmd) for k, cmd in steps(name, hf, work) if k in want]
    g = golden(name)
    for k, cmd in todo:
        print(f"[{k}] " + " ".join(shlex.quote(c) for c in cmd))
        if k == "run" and g is None:
            print(f"[{k}] note: the repository has no golden file for {name}. The host reads "
                  f"{work / 'golden_c1024.txt'} and stops without it. Put the HF greedy tokens there first.")
    if a.dry_run:
        return 0
    for k, cmd in todo:
        if k == "run":
            if g is not None:
                shutil.copyfile(g, work / "golden_c1024.txt")    # as vm_run.sh
            elif not (work / "golden_c1024.txt").is_file():
                print(f"error: no {work / 'golden_c1024.txt'} (host.hip reads it)", file=sys.stderr)
                return 2
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            print(f"error: step {k} failed with exit code {r.returncode}", file=sys.stderr)
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
