#!/usr/bin/env bash
# Run after base build; no network or device execution.
set -euo pipefail
ROOT=${1:?root}; FLEET=${2:?source-bound capture fleet}; OUT=${3:?output}
CAPTURE_NAME=${4:-native-capture}
FLEET_SRC=${ETX_COLUMN_FLEET_SRC:?accepted cooked Fleet source required}
HEADERS=${ETX_ROCM_HEADERS:-$HOME/code/rocm-headers}
PYTHON=${ETX_PYTHON:-/Users/ckl/code/etx/.venv/bin/python}
PYTHONPATH="$ROOT" "$PYTHON" -m etx compile "$ROOT/examples/experiment_program/columns/g13_graph.py" --arch gfx942 --out "$OUT/g13-ready-final" --inline-tiles --relay off --no-event-elim > "$OUT/g13-ready-final-compile.log" 2>&1
PYTHONPATH="$ROOT" "$PYTHON" -c 'import json,sys;from pathlib import Path;from examples.experiment_program.columns.g13_graph import verify_compiled_queues;p=Path(sys.argv[1]);(p/"queue-proof.json").write_text(json.dumps(verify_compiled_queues(json.loads((p/"plan.json").read_text())),indent=2)+"\n")' "$OUT/g13-ready-final"
# The patcher writes its own copies, retaining every original source digest.
PYTHONPATH="$ROOT" "$PYTHON" -m examples.experiment_program.columns.patch_capture --fleet-src "$FLEET_SRC" --shim "$ROOT/examples/dsv2lite/fleet_shim.hip" --host "$ROOT/examples/dsv2lite/host.hip" --out "$OUT/$CAPTURE_NAME" > "$OUT/capture-patch.log" 2>&1
docker run --rm --pull=never --network=none --platform linux/amd64 \
 -v "$ROOT:/etx:ro" -v "$FLEET:/fleet:ro" -v "$FLEET_SRC:/fleet/src:ro" -v "$OUT:/out" -v "$HEADERS:/rocm:ro" \
 -v "$HOME/.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/link-deps/root:/link-deps:ro" \
 -e CAPTURE_NAME="$CAPTURE_NAME" -e LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu -w /etx etx-rocm22 bash -c '
set -euo pipefail
RL=/rocm/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=/rocm/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
export TMPDIR=/out/compiler-tmp
CXX=$RL/bin/clang++
COMMON=(-x hip --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path="$DL" -isystem /rocm/inc -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -DFLEET_NT_WEIGHTS=1 -I/fleet/src -I/fleet/src/kernels -I/etx/etx/runtime/include -I/out -I/etx -I/out/g13-ready-final)
"$CXX" "${COMMON[@]}" --cuda-device-only -Rpass-analysis=kernel-resource-usage -S /out/g13-ready-final/megakernel_d0.hip -o /out/g13-ready-final/g13.s > /out/g13-ready-final/device.stdout.log 2> /out/g13-ready-final/resource.log
"$CXX" -target amdgcn-amd-amdhsa -mcpu=gfx942 -c /out/g13-ready-final/g13.s -o /out/g13-ready-final/g13.o > /out/g13-ready-final/assemble.log 2>&1
"$RL/bin/ld.lld" -shared /out/g13-ready-final/g13.o -o /out/g13-ready-final/g13.hsaco > /out/g13-ready-final/link.log 2>&1
"$RL/bin/llvm-readobj" --notes /out/g13-ready-final/g13.hsaco > /out/g13-ready-final/metadata.txt
ln -s g13.s /out/g13-ready-final/mk.s
ln -s g13.hsaco /out/g13-ready-final/model.hsaco
"$CXX" -x c++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -DETX_G13_DRIVER -I/rocm/inc -I/etx -I/etx/etx/runtime/include -I/out/g13-ready-final examples/experiment_program/columns/driver.cpp -fuse-ld=lld -ldl -o /out/g13-ready-run > /out/g13-ready-final/host.log 2>&1
# Include the exact patched original kernel from a shim copy without copying the
# rest of the Fleet tree. It is read-only behind the /fleet include fallback.
sed '\''s@#include "kernels/fleet_kernel.hip"@#include "kernel.hip"@'\'' /out/$CAPTURE_NAME/shim.hip > /out/$CAPTURE_NAME/compile-shim.hip
"$CXX" "${COMMON[@]}" --cuda-device-only -Rpass-analysis=kernel-resource-usage -S /out/$CAPTURE_NAME/compile-shim.hip -o /out/$CAPTURE_NAME/shim.s > /out/$CAPTURE_NAME/device.stdout.log 2> /out/$CAPTURE_NAME/resource.log
"$CXX" -target amdgcn-amd-amdhsa -mcpu=gfx942 -c /out/$CAPTURE_NAME/shim.s -o /out/$CAPTURE_NAME/shim.o > /out/$CAPTURE_NAME/assemble.log 2>&1
# Host model body is compiled to an actual object. Its final linked model
# executable needs the generated full-model plan and original HIP runtime.
"$CXX" "${COMMON[@]}" --cuda-host-only -c /out/$CAPTURE_NAME/host.hip -o /out/$CAPTURE_NAME/host.o > /out/$CAPTURE_NAME/host.log 2>&1
sha256sum /out/g13-ready-final/*.hip /out/g13-ready-final/plan.json /out/g13-ready-final/plan_data.h /out/g13-ready-final/g13.hsaco /out/g13-ready-run /out/$CAPTURE_NAME/*.hip /out/$CAPTURE_NAME/shim.o /out/$CAPTURE_NAME/host.o > /out/EXTENDED_SHA256SUMS
' > "$OUT/extended-docker.log" 2>&1
cat "$OUT/extended-docker.log"
