#!/usr/bin/env bash
# Entire local GLM acceptance, no checkpoint downloads, network or GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?new acceptance directory under ~/code}
mkdir -p "$OUT"
for b in 1 2 4 8; do
  flags=(); [ "$b" -lt 4 ] || flags+=(--bf16); [ "$b" -ne 8 ] || flags+=(--fp8)
  .venv/bin/python -u -m examples.glm53.cpu_check --out "$OUT/cpu-b$b" --batch "$b" ${flags[@]+"${flags[@]}"}
done
.venv/bin/python - "$OUT/full-shape" <<'PY'
from pathlib import Path
import sys
from examples.glm53.model_build import generate,production_compile_spec
p=generate(production_compile_spec(),Path(sys.argv[1]),8)
assert p.n_devices==2 and len(p.tasks)==384
print('FULL-SHAPE PLAN OK (no checkpoint weights; compiler check only)',flush=True)
PY
R=${ETX_ROCM_HEADERS:-$HOME/code/rocm-headers}
export ETX_DEVLIBS=${ETX_DEVLIBS:-$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode}
for d in 0 1; do
  bash etx/importer/hip_ir_local.sh "$OUT/full-shape/megakernel_d$d.hip" "$OUT/full-shape/mk$d.ll" -Ietx/runtime/include -I "$OUT/full-shape" -I .
  "${ETX_LLVM:-/opt/homebrew/opt/llvm/bin}/llc" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 "$OUT/full-shape/mk$d.ll" -o "$OUT/full-shape/mk$d.s"
done
"${ETX_LLVM:-/opt/homebrew/opt/llvm/bin}/clang++" -x hip --cuda-host-only -fsyntax-only -nogpuinc -nogpulib \
  --offload-arch=gfx942 -std=c++17 -isystem "$R/clang" -isystem "$R/inc" -include __clang_hip_runtime_wrapper.h \
  -isystem etx/importer/shim/mac -D__HIP_PLATFORM_AMD__ -Ietx/runtime/include -I "$OUT/full-shape" -I . examples/glm53/model_host.hip
bash examples/glm53/local_model_rocm.sh "$OUT/full-shape"
.venv/bin/python -m etx.tools.gpu_jobs examples/sessions/glm53_model.py --gpus 0,1 --logdir "$OUT/dryrun" --dry-run
.venv/bin/python -m pytest -q tests/test_glm53_model.py tests/test_glm53_preparation.py
