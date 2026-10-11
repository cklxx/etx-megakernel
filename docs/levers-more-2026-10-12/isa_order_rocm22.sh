#!/usr/bin/env bash
# isa_order.py on the VM's compiler: ROCm 7.2.4 clang 22 in the etx-rocm22 container (see rocm_build_check.sh).
#   bash docs/levers-more-2026-10-12/isa_order_rocm22.sh > docs/levers-more-2026-10-12/isa_order_rocm22.txt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; B=${ETX_BUILD:-$HOME/code/etx/build/qwen300-rocm-on}
W=$HOME/code/etx/build/lv_isa22; mkdir -p "$W"
case "$ROOT$B" in *) [[ $ROOT == $HOME/code/* && $B == $HOME/code/* ]] || { echo "paths must be under ~/code"; exit 1; };; esac
PY=${PY:-$HOME/code/etx/.venv/bin/python}
"$PY" - "$ROOT" "$W" <<'EOF'
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('isa', pathlib.Path(sys.argv[1]) / 'docs/master-table-2026-10-11/isa.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
(pathlib.Path(sys.argv[2]) / 'wrap.hip').write_text(m.WRAP)
EOF
c() { echo "/code/${1#$HOME/code/}"; }
docker run --rm --pull=never --network=none --platform linux/amd64 -v "$HOME/code:/code" etx-rocm22 bash -c "
set -e; R=/code/rocm-headers; RL=\$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm; DL=\$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
cd $(c "$ROOT")
\$RL/bin/clang++ -x hip --cuda-device-only --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path=\$DL \
  -isystem \$R/inc -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -gline-tables-only \
  -DETX_DSL_COH -DETX_DSL_NT -DETX_DSL_VECNORM -DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH -DETX_TOPK_KEY32 \
  -I $(c "$B") -I etx/runtime/include -I etx/runtime -S $(c "$W")/wrap.hip -o $(c "$W")/wrap.s >&2
\$RL/bin/clang++ --version | head -1"
cd "$ROOT" && ETX_BUILD="$B" ISA_S="$W/wrap.s" "$PY" docs/levers-more-2026-10-12/isa_order.py
