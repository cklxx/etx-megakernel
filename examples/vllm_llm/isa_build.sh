#!/usr/bin/env bash
# Assembly of every kernel of one build, both forms, with ROCm 7.2.4's clang (the VM's compiler), for the kernel model
# (examples/vllm_llm/isa_inventory.py, docs/qwen/MODEL-vllm-kernels.md):
#   orig_<source>.s   vLLM's HIP kernels as vLLM compiles them (the sliced source with the launched specialisations)
#   imp_wrap.s        each imported kernel alone in a 1024-thread workgroup, exactly as a megakernel task runs it
#                     (etx_imp_<e>(args, block, tid, slot) behind the adapter's slot split; own register allocation)
# Triton kernels: their cached .amdgcn (examples/vllm_llm/triton/<export>) is vLLM's own compile.
#   bash examples/vllm_llm/isa_build.sh <build dir from rocm_build_check.sh>
set -euo pipefail
B=$(cd "${1:?build dir}" && pwd)
case "$B" in "$HOME/code/"*) ;; *) echo "build dir must be under ~/code"; exit 1;; esac
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
python3 - "$B" <<'EOF'
import json, sys
from pathlib import Path
b = Path(sys.argv[1]); imp = json.loads((b / "imp/imports.json").read_text())
out = ['#define ETX_IMPORT_LDS %d' % max(i["lds_bytes"] for i in imp.values() if i),
       '#define ETX_IMPORT_SLOTS %d' % max(i["slots"] for i in imp.values()), '#include "etx/import.h"']
for e, i in imp.items():
    st, k = i["slot_threads"], i["slots"]
    out += [f'extern "C" __device__ void etx_imp_{e}(const void* args, int blk, int tid, int slot);',
            f'extern "C" __global__ void __launch_bounds__(1024, 1) wrap_{e}(const unsigned char* a, int task) {{',
            '  etx_import_init(); __syncthreads();',
            '  int tx; asm volatile("v_mov_b32 %0, %1" : "=v"(tx) : "v"((int)threadIdx.x));',
            f'  const int slot = tx / {st}, tid = tx - slot * {st};',
            f'  if (slot < {k}) etx_imp_{e}(a, task * {k} + slot, tid, slot);', '}']
(b / "imp_wrap.hip").write_text("\n".join(out) + "\n")
EOF
C_B=/code/${B#$HOME/code/}; C_ROOT=/code/${ROOT#$HOME/code/}
docker run --rm --platform linux/amd64 -v "$HOME/code:/code" -v "$HOME/code:$HOME/code" -e B="$C_B" -e ROOT="$C_ROOT" etx-rocm22 bash -c '
set -euo pipefail
R=/code/rocm-headers; RL=$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm; DL=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
F=(-x hip --cuda-device-only --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path=$DL -isystem $R/inc
   -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -DUSE_ROCM -DNDEBUG -isystem $R/pytorch)
cd $ROOT
for src in $B/imp/*.hip; do
  base=$(basename $src .hip)
  $RL/bin/clang++ "${F[@]}" -include etx/importer/shim/etx_cuda2hip.h -I etx/importer/shim -I etx/importer/shim_nofp8 \
    -I /code/vllm/csrc -I /code/vllm/csrc/libtorch_stable -I /code/vllm/csrc/rocm -I /code/vllm/csrc/libtorch_stable/moe -DENABLE_FP8 \
    -S $src -o $B/orig_$base.s 2> $B/orig_$base.log || { echo "orig $base failed"; grep -m3 error $B/orig_$base.log; }
done
$RL/bin/clang++ "${F[@]}" -Ietx/runtime/include -Xclang -mlink-builtin-bitcode -Xclang $B/imports.bc -S $B/imp_wrap.hip -o $B/imp_wrap.s \
  2> $B/imp_wrap.log || { grep -m5 error $B/imp_wrap.log; exit 1; }
ls $B/*.s | wc -l
'
