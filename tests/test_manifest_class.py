"""One test per past occurrence of the define/manifest bug class, so none of the four can recur silently.

Each of these was a build that compiled, ran, and in three of four cases still matched its output tokens, while the
plan and the compiled code disagreed about a feature (etx/codegen/manifest.py records the four).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

from etx.codegen import manifest

ROOT = Path(__file__).resolve().parents[1]
BUILDS = [ROOT / "examples/vllm_llm/build.sh", ROOT / "examples/vllm_llm/rocm_build_check.sh"]


def test_round14_nb_and_requests_come_from_the_plan():
    """ETX_VL_NB: the tile was emitted without it while the plan batched, so extra requests read column 0."""
    for key, mac in (("nb", "ETX_VL_NB"), ("requests", "ETX_VL_REQUESTS")):
        assert manifest.FEATURES[key][0] == mac and mac in manifest.RETIRED
    block = "\n".join(manifest.guard_block({"nb": 2, "requests": 2}))
    assert "#define ETX_VL_NB 2" in block and "#define ETX_VL_REQUESTS 2" in block
    assert "#define ETX_VL_NB 1" in "\n".join(manifest.guard_block({}))      # the default is published too


def test_round19_a_define_clobbered_by_a_later_assignment_cannot_matter():
    """build.sh appended -DETX_DEV_RESET to an array that a later `if` overwrote whenever ETX_VL_COH was set. No
    shell array can carry these any more: the generated artefacts define them, and a stray -D is an #error."""
    for f in BUILDS:
        src = f.read_text()
        for mac in manifest.RETIRED:
            assert f"-D{mac}" not in src, f"{f.name} still passes -D{mac}; the plan publishes it"


def test_round20_both_translation_units_get_the_same_block():
    """The define reached the final link but not the separate device-code compile. The megakernel TU and the host
    TU now include the same block from the same function, so one cannot have it without the other."""
    k = (ROOT / "etx/codegen/kernel.py").read_text()
    h = (ROOT / "etx/codegen/plan_header.py").read_text()
    assert "manifest.guard_block(" in k and "manifest.guard_block(" in h
    # in the kernel it must precede the tile includes, or a tile reads the macro before it is defined
    assert k.index("manifest.guard_block(") < k.index("single-TU build: tile sources included")


def test_round30_a_derived_decision_cannot_be_contradicted_by_a_define():
    """model.py derived "do not fold the top-k" while build.sh's define defaulted to "fold": the plan gave the
    top-k its own phase and the expert tiles recomputed it anyway. 0.63 ms, tokens still matched."""
    block = "\n".join(manifest.guard_block({"fold_topk": False}))
    assert "#define ETX_EMIT_FOLD_TOPK 0" in block
    assert '#error' in block and "ETX_VL_NO_TOPK_FOLD" in block
    model = (ROOT / "examples/vllm_llm/model.py").read_text()
    assert 'g.codegen["fold_topk"]' in model


def test_every_macro_the_tiles_test_is_either_published_or_tile_local():
    """The audit that found the fifth: a macro a tile source tests must come from the manifest, or be local to the
    tile with no plan or host counterpart. Anything else is the bug class waiting to happen."""
    tile_local = {"ETX_DSL_NT", "ETX_DSL_VECNORM", "ETX_DSL_UNIFORM", "ETX_DSL_PREP_EARLY", "ETX_IMPORT_LDS",
                  "ETX_DSL_COH_HELPERS", "DSL_G", "DSL_U32X4_DEFINED", "DSL_SPLIT_TAIL",
                  "ETX_DSL_U4",   # load depth: only the tile's own instantiation changes (round 34)
                  # tile code only, same argument layout and output bits (docs/TILES-2026-10-07.md)
                  "ETX_DSL_WSUM", "ETX_DSL_SCALAR", "ETX_DSL_GU1", "ETX_DSL_RBATCH"}
    published = {m for m, _, _ in manifest.FEATURES.values()}
    tested = set()
    for f in (ROOT / "etx/ops").glob("*.py"):
        for m in re.finditer(r"#if(?:n?def)?\s+!?\s*([A-Z][A-Z_0-9]+)", f.read_text()):
            tested.add(m.group(1))
    unknown = {m for m in tested if m not in tile_local and m not in published and not m.startswith(("DSL_ABLATE_", "ETX_ABL_"))}
    assert not unknown, f"tile macros that are neither published nor tile-local: {sorted(unknown)}"


def test_the_trap_fires_on_a_command_line_define():
    """The mechanism itself: a -D for a published feature must stop the compile, not change behaviour."""
    block = "\n".join(manifest.guard_block({"coh": True}))
    compiler = shutil.which("c++")
    assert compiler, "CPU acceptance requires a C++ preprocessor"
    good = subprocess.run([compiler, "-E", "-x", "c++", "-"], input=block, capture_output=True, text=True)
    bad = subprocess.run([compiler, "-E", "-x", "c++", "-DETX_DSL_COH=1", "-"],
                         input=block, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert bad.returncode != 0 and "must not be defined on the command line" in bad.stderr
