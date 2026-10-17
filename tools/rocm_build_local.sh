#!/usr/bin/env bash
# Docker-free version of examples/dsv2lite/rocm_build_check.sh: the same compile steps with a locally installed ROCm
# compiler instead of the etx-rocm22 image + ~/code/rocm-headers. Logic taken from the local device build
# (xbobuild.sh, 2026-10-18) and from rocm_build_check.sh's container script.
# Usage: tools/rocm_build_local.sh TOOLCHAIN PINNED_FLEET_EXPORT ARM [OUT]
#        tools/rocm_build_local.sh TOOLCHAIN PINNED_FLEET_EXPORT --arm REGISTERED_ARM [--stamps] [OUT]
#   TOOLCHAIN: 7.2.4 | 7.14 (roots: ETX_ROCM_724 / ETX_ROCM_714, defaults below)
#   ARM (legacy form): default | stamps | <lever>[+<lever>...]
# Writes OUT/<arm>/{compiler.txt,compiler.json,build_env.json,plan.json,megakernel_d0.hip,resource.log,mk.s,isa.bin,...}
# and OUT/resources.json (examples.dsv2lite.build_resources). No GPU, no network.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TC=${1:?toolchain 7.2.4 or 7.14}; shift
FLEET="$(cd "${1:?pinned fleet export}" && pwd)"; shift
PY=${ETX_PYTHON:-$ROOT/.venv/bin/python}
REGISTERED=""
STAMPED=0
if [ "${1:-}" = --arm ]; then REGISTERED=${2:?registered arm name required}; shift 2; ARM=$REGISTERED
  # --stamps: the stamped build of the arm (patch_stamps after the arm's patches, -DETX_DSV2_STAMPS2), OUT/<arm>_st
  if [ "${1:-}" = --stamps ]; then STAMPED=1; shift; ARM=${REGISTERED}_st; fi
else ARM=${1:?arm}; shift; fi
OUT=${1:-$ROOT/build/dsv2lite-local-$TC}
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
if [ -e "$FLEET/.git" ]; then echo 'refusing a fleet checkout; use git archive 3fb67a6' >&2; exit 2; fi
cd "$ROOT"
case "$TC" in
  7.2.4) S=${ETX_ROCM_724:-/workspace/rocm724deb/stage/opt/rocm-7.2.4}; RL=$S/lib/llvm; DL=$S/amdgcn/bitcode; INC=$S/include;;
  7.14)  S=${ETX_ROCM_714:-/workspace/rocm714venv/lib/python3.13/site-packages/_rocm_sdk_core}; RL=$S/lib/llvm
         DL=$S/lib/llvm/amdgcn/bitcode; INC=$S/include;;
  *) echo "unknown toolchain $TC (7.2.4 | 7.14)" >&2; exit 2;;
esac
for f in "$RL/bin/clang++" "$RL/bin/llvm-mc" "$RL/bin/llvm-objcopy" "$DL/ockl.bc" "$INC/hip/hip_runtime.h"; do
  [ -e "$f" ] || { echo "local ROCm $TC incomplete: missing $f" >&2; exit 2; }
done
# the fleet source must be the pinned export (examples/dsv2lite/pinned.py FLEET_SRC_SHA256): a build from another fleet
# commit gives other resource numbers under the same arm name
"$PY" -c 'import sys; from pathlib import Path; from examples.dsv2lite.pinned import source_sha, FLEET_SRC_SHA256, FLEET_COMMIT
got = source_sha(Path(sys.argv[1]) / "src")
sys.exit(0 if got == FLEET_SRC_SHA256 else f"fleet source {sys.argv[1]}/src is not the pinned fleet {FLEET_COMMIT}: {got} != {FLEET_SRC_SHA256}")' "$FLEET" || exit 2
if [ -n "$REGISTERED" ]; then
  # an arm never inherits ambient plan switches or flags (as rocm_build_check.sh --arm and the session)
  while IFS= read -r name; do case "$name" in ETX_*|FLEET_*) unset "$name";; esac; done < <(compgen -e)
  eval "$(cd "$ROOT" && "$PY" -m examples.dsv2lite.arms shell "$REGISTERED")"
  PATCH_ARM=${ETX_FLEET_PATCH// /+}; PATCH_ARM=${PATCH_ARM:-default}
  [ "$STAMPED" = 0 ] || PATCH_ARM="$PATCH_ARM+stamps"
else
  LEVERS=$(cd "$ROOT" && "$PY" -c 'from examples.dsv2lite.patch_fleet import levers; print(" ".join(levers()))')
  if [ "$ARM" != default ]; then
    for lever in ${ARM//+/ }; do
      case " stamps $LEVERS " in *" $lever "*) ;; *) echo "unknown arm $ARM: lever $lever not on this branch" >&2; exit 2;; esac
    done
  fi
  PATCH_ARM=$ARM
fi
D="$OUT/$ARM"; rm -rf "$D"; mkdir -p "$D"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
"$PY" -c 'import json, os; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith("ETX_")}, sort_keys=True, indent=2))' > "$D/build_env.json"
"$RL/bin/clang++" --version > "$D/compiler.txt"
"$PY" - "$TC" "$S" "$RL/bin/clang++" "$D/compiler.txt" > "$D/compiler.json" <<'PYEOF'
import json, re, sys
tc, root, clang, txt = sys.argv[1:]
first = open(txt).read().splitlines()[0]
m = re.search(r"\(([^ )]+) ([^)]*?)([0-9a-f]{40})", first)
print(json.dumps({"mode": "local", "toolchain": tc, "rocm_root": root, "compiler_path": clang,
                  "clang_version": first, "clang_commit": m.group(3) if m else None}, indent=2, sort_keys=True))
PYEOF
"$PY" -m etx compile examples/dsv2lite/graph.py --arch gfx942 --out "$D" --inline-tiles --relay on > "$D/compile.log" 2>&1
SRC=$FLEET/src
PATCHES=(); FLAGS=()
if [ "${ETX_COHERENCE:-declared}" = l1 ] && [[ "+$PATCH_ARM+" != *"+l1_loads+"* ]]; then PATCH_ARM="$PATCH_ARM+l1_loads"; fi
STAMPS=0
for token in ${PATCH_ARM//+/ }; do
  case "$token" in default) ;; stamps) STAMPS=1; FLAGS=(-DETX_DSV2_STAMPS2);; *) PATCHES+=("$token");; esac
done
if [ ${#PATCHES[@]} -gt 0 ] || [ "$STAMPS" = 1 ]; then
  mkdir -p "$D/fleet_src"; cp -r "$FLEET/src/." "$D/fleet_src/"; SRC="$D/fleet_src"
  [ ${#PATCHES[@]} -eq 0 ] || "$PY" -m examples.dsv2lite.patch_fleet "$SRC" "${PATCHES[@]}" > "$D/patch.log" 2>&1
  [ "$STAMPS" = 0 ] || "$PY" -m examples.dsv2lite.patch_stamps "$SRC" >> "$D/patch.log" 2>&1
fi
"$PY" -m examples.dsv2lite.coh_check "$D/megakernel_d0.hip" "$SRC" > "$D/coh_check.log" 2>&1 || { cat "$D/coh_check.log"; exit 2; }
read -ra EXTRA <<< "${ETX_CFLAGS:-}"
COMMON=(-x hip --offload-arch=gfx942 -O3 -std=c++17 -nogpuinc --rocm-device-lib-path="$DL"
  -isystem "$INC" -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -DFLEET_NT_WEIGHTS=1
  -I "$SRC" -I "$ROOT/etx/runtime/include" -I "$D" -I "$ROOT" ${FLAGS[@]+"${FLAGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})
declare -p COMMON > "$D/compiler_args.txt"
"$RL/bin/clang++" "${COMMON[@]}" --cuda-device-only -Rpass-analysis=kernel-resource-usage \
  -S "$D/megakernel_d0.hip" -o "$D/mk.s" > "$D/device.stdout.log" 2> "$D/resource.log"
"$RL/bin/llvm-mc" -triple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj "$D/mk.s" -o "$D/mk.o" > "$D/assemble.log" 2>&1
"$RL/bin/llvm-objcopy" --dump-section .text="$D/isa.bin" "$D/mk.o"
# the host passes need the full HIP host headers; record their result, they do not change the device resources
set +e
"$RL/bin/clang++" "${COMMON[@]}" --cuda-host-only -fsyntax-only examples/dsv2lite/host.hip > "$D/host_syntax.log" 2>&1
echo $? > "$D/host_syntax.exit"
set -e
echo "UNAVAILABLE: local mode does not link the executable" > "$D/link.log"; echo 77 > "$D/link.exit"
sha256sum "$D/plan.json" "$D/plan_data.h" "$D/megakernel_d0.hip" "$D/mk.s" "$D/isa.bin" > "$D/SHA256SUMS"
echo "$ARM: local ROCm $TC device pass OK"
"$PY" -m examples.dsv2lite.build_resources "$OUT"
