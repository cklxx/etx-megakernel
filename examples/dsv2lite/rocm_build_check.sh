#!/usr/bin/env bash
# Offline device + host compilation using the already-local x86 ROCm 7.2.4 toolchain.
# Usage: rocm_build_check.sh [PINNED_FLEET_EXPORT] --arm NAME [OUT]
# Legacy: rocm_build_check.sh PINNED_FLEET_EXPORT [default|stamps|<lever>[+<lever>...]|all] [OUT]
#   all = default, stamps and every lever on this branch alone (examples/dsv2lite/levers/).
# No GPU, pull or network. Fleet must be an export, never a checkout. Extra compiler
# flags go in ETX_CFLAGS; ETX_DSV2_* graph settings are forwarded explicitly.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HEADERS=${ETX_ROCM_HEADERS:-$HOME/code/rocm-headers}
REGISTERED=""
if [ "${1:-}" = --arm ]; then
  FLEET=$(cd "$ROOT" && .venv/bin/python -m examples.dsv2lite.pinned export)
else
  FLEET="$(cd "${1:?pinned fleet export or --arm required}" && pwd)"; shift
fi
if [ "${1:-}" = --arm ]; then
  REGISTERED=${2:?registered arm name required}; shift 2
  # The session strips these too: an arm never inherits ambient plan switches or flags.
  while IFS= read -r name; do case "$name" in ETX_*|FLEET_*) unset "$name";; esac; done < <(compgen -e)
  ARM_ENV=$(cd "$ROOT" && .venv/bin/python -m examples.dsv2lite.arms shell "$REGISTERED")
  eval "$ARM_ENV"
  FLEET=$(cd "$ROOT" && .venv/bin/python -c 'from pathlib import Path; import sys; from examples.dsv2lite.pinned import fleet_export; print(fleet_export(Path(sys.argv[1])))' "$FLEET")
  ARM=$REGISTERED
else
  ARM=${1:-all}; if [ "$#" -gt 0 ]; then shift; fi
fi
OUT=${1:-$ROOT/build/dsv2lite-rocm22}
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
if [ -e "$FLEET/.git" ]; then echo 'refusing a fleet checkout; use git archive 3fb67a6' >&2; exit 2; fi
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
(cd "$ROOT" && "$PY" -c 'import sys; from pathlib import Path; from examples.dsv2lite.pinned import source_sha, FLEET_SRC_SHA256, FLEET_COMMIT
got = source_sha(Path(sys.argv[1]) / "src")
sys.exit(0 if got == FLEET_SRC_SHA256 else f"fleet source {sys.argv[1]}/src is not the pinned fleet {FLEET_COMMIT}: {got} != {FLEET_SRC_SHA256}")' "$FLEET" ) || exit 2
if ! docker image inspect etx-rocm22 >/dev/null 2>&1; then
  echo 'Missing local image. Not pulling or building.' >&2
  echo 'docker build --platform linux/amd64 -t etx-rocm22 -f examples/vllm_llm/rocm_build_check.Dockerfile .' >&2
  echo 'Image creation needs Ubuntu/package downloads; time and download cost unknown; GPU cost $0.' >&2
  exit 2
fi
docker image inspect etx-rocm22 --format '{{.Id}}' > "$OUT/image.txt"
LEVERS=$(cd "$ROOT" && .venv/bin/python -c 'from examples.dsv2lite.patch_fleet import levers; print(" ".join(levers()))')
if [ -z "$REGISTERED" ] && [ "$ARM" = all ]; then ARMS=(default stamps $LEVERS); else ARMS=("$ARM"); fi
for arm in "${ARMS[@]}"; do
  if [ -n "$REGISTERED" ]; then continue; fi
  case "$arm" in default) continue;; esac
  for lever in ${arm//+/ }; do          # "stamps" combines with levers: stamps+<lever>+...
    case " stamps $LEVERS " in *" $lever "*) ;; *) echo "unknown arm $arm: lever $lever not on this branch ($LEVERS)" >&2; exit 2;; esac
  done
done
ENV_ARGS=()
while IFS= read -r name; do
  if [ -n "$REGISTERED" ]; then
    case "$name" in ETX_*) ENV_ARGS+=(-e "$name=${!name}");; esac
  else
    case "$name" in ETX_DSV2_*|ETX_COHERENCE) ENV_ARGS+=(-e "$name=${!name}");; esac
  fi
done < <(compgen -e)
for arm in "${ARMS[@]}"; do
  PATCH_ARM=$arm
  if [ -n "$REGISTERED" ]; then PATCH_ARM=${ETX_FLEET_PATCH// /+}; fi
  mkdir -p "$OUT/$arm"
  echo "== $arm: offline ROCm 7.2.4 compile"
  docker run --rm --pull=never --network=none --platform linux/amd64 \
    -v "$ROOT:/etx:ro" -v "$FLEET:/fleet:ro" -v "$OUT:/out" \
    -v "$HEADERS:/rocm:ro" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -e NAME="$arm" -e ARM="$PATCH_ARM" -e ETX_CFLAGS="${ETX_CFLAGS:-}" \
    -w /etx etx-rocm22 bash -c '
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
RL=/rocm/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=/rocm/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
OUT=/out/$NAME
python3 -c '\''import json, os; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith("ETX_")}, sort_keys=True, indent=2))'\'' > "$OUT/build_env.json"
"$RL/bin/clang++" --version > "$OUT/compiler.txt"
python3 -m etx compile examples/dsv2lite/graph.py --arch gfx942 --out "$OUT" --inline-tiles --relay on > "$OUT/compile.log" 2>&1
SRC=/fleet/src
PATCHES=(); FLAGS=()
# the build-wide coherence mode needs the fleet family agent-scope accessors (same rule as build.sh)
if [ "${ETX_COHERENCE:-declared}" = l1 ] && [[ "+$ARM+" != *"+l1_loads+"* ]]; then ARM="$ARM+l1_loads"; fi
STAMPS=0
for token in ${ARM//+/ }; do
  case "$token" in default) ;; stamps) STAMPS=1; FLAGS=(-DETX_DSV2_STAMPS2);; *) PATCHES+=("$token");; esac
done
if [ ${#PATCHES[@]} -gt 0 ] || [ "$STAMPS" = 1 ]; then
  rm -rf "$OUT/fleet_src"; mkdir -p "$OUT/fleet_src"
  cp -r /fleet/src/. "$OUT/fleet_src/"; SRC="$OUT/fleet_src"
  if [ ${#PATCHES[@]} -gt 0 ]; then
    python3 -m examples.dsv2lite.patch_fleet "$SRC" "${PATCHES[@]}" > "$OUT/patch.log" 2>&1
  fi
  if [ "$STAMPS" = 1 ]; then
    python3 -m examples.dsv2lite.patch_stamps "$SRC" >> "$OUT/patch.log" 2>&1
  fi
fi
python3 -m examples.dsv2lite.coh_check "$OUT/megakernel_d0.hip" "$SRC" > "$OUT/coh_check.log" 2>&1 || { cat "$OUT/coh_check.log"; exit 2; }
# ETX_CFLAGS is a whitespace-separated compiler argument list, never evaluated as code.
read -ra EXTRA <<< "${ETX_CFLAGS:-}"
COMMON=(-x hip --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path="$DL"
  -isystem /rocm/inc -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -DFLEET_NT_WEIGHTS=1
  -I "$SRC" -I /etx/etx/runtime/include -I "$OUT" -I /etx "${FLAGS[@]}" "${EXTRA[@]}")
declare -p COMMON > "$OUT/compiler_args.txt"
"$RL/bin/clang++" "${COMMON[@]}" --cuda-device-only -Rpass-analysis=kernel-resource-usage \
  -S "$OUT/megakernel_d0.hip" -o "$OUT/mk.s" > "$OUT/device.stdout.log" 2> "$OUT/resource.log"
"$RL/bin/llvm-mc" -triple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj "$OUT/mk.s" -o "$OUT/mk.o" > "$OUT/assemble.log" 2>&1
"$RL/bin/llvm-objcopy" --dump-section .text="$OUT/isa.bin" "$OUT/mk.o"
"$RL/bin/clang++" "${COMMON[@]}" --cuda-device-only -S examples/dsv2lite/host.hip -o "$OUT/host_dev.s" > "$OUT/host_dev.log" 2>&1
"$RL/bin/clang++" "${COMMON[@]}" --cuda-host-only -fsyntax-only examples/dsv2lite/host.hip > "$OUT/host_syntax.log" 2>&1
"$RL/bin/clang++" "${COMMON[@]}" --cuda-host-only -c examples/dsv2lite/host.hip -o "$OUT/host.o" > "$OUT/host_object.log" 2>&1
# A device object plus a host object is not a linked HIP executable. Probe the local
# runtime separately; record the missing dependency and keep all compile evidence.
if [ -f /rocm/lib/libamdhip64.so ]; then
  set +e
  "$RL/bin/clang++" "${COMMON[@]}" "$OUT/megakernel_d0.hip" examples/dsv2lite/host.hip \
    -L/rocm/lib -lamdhip64 -o "$OUT/run" > "$OUT/link.log" 2>&1
  LINK_RC=$?
  set -e
  echo "$LINK_RC" > "$OUT/link.exit"
  [ "$LINK_RC" = 0 ] || { cat "$OUT/link.log"; exit "$LINK_RC"; }
else
  echo "UNAVAILABLE: local libamdhip64.so is absent; executable/native runtime load is a remaining gate" > "$OUT/link.log"
  echo 77 > "$OUT/link.exit"
fi
sha256sum "$OUT/plan.json" "$OUT/plan_data.h" "$OUT/megakernel_d0.hip" "$OUT/mk.s" "$OUT/isa.bin" > "$OUT/SHA256SUMS"
sha256sum "$OUT/plan.json"
echo "$NAME: device and host passes OK"
' > "$OUT/$arm/docker.log" 2>&1 || { cat "$OUT/$arm/docker.log"; echo "FAILED: full logs in $OUT/$arm" >&2; exit 1; }
  cat "$OUT/$arm/docker.log"
done
cd "$ROOT"
"$ROOT/.venv/bin/python" -m examples.dsv2lite.build_resources "$OUT"
