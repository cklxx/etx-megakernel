# Kernel library

Maintain reusable kernel implementations and their compatibility and measurement records here, organized by machine and model. This is a library/catalogue, not an automatic kernel selector. An entry does not imply that a kernel is imported, correct, or fastest.

## Layout

```text
kernels/<architecture>/<machine>/<model>/
    README.md                 implementations, contracts and evidence
    <operator>/<variant>/     source or capture and its provenance, when added
```

Use architecture identifiers from `etx/machine/arch/` (for example `gfx942` or `sm_90`), lowercase machine names (`mi300x`, `mi325x`), and explicit model names (`qwen3-30b-a3b`, `glm-5.3-flash`). Keep different machine products separate even when they share an ISA. Record VF/bare-metal status, topology and software versions in each measurement; the directory name alone does not establish compatibility.

## Initial entries

- [gfx942 / MI300X / Qwen3-30B-A3B](gfx942/mi300x/qwen3-30b-a3b/README.md): existing DSL/import hybrid and imported-kernel sources, with qualified session evidence.
- [gfx942 / MI325X / GLM-5.3-Flash](gfx942/mi325x/glm-5.3-flash/README.md): native implementation and diagnostic execution; no qualified fastest kernel established.

These entries point to the existing source of truth. They are an initial inventory, not a complete catalogue of supported models or machines.

## What to record

For each implementation or variant, record:

- **Identity and provenance:** operator, source location, upstream revision, license when vendoring, and hashes for captured artifacts.
- **Compatibility:** GPU/ISA, compiler/runtime versions, model revision, shapes, batch/context limits, precision, quantization, layouts, launch ABI, scratch/LDS requirements and parallel topology.
- **Integration:** standalone launch, ETX-importable, or ETX-native; adapter/build commands and unsupported assumptions. Source availability is not import support.
- **Correctness:** local/fixture checks separately from GPU checks; imported-versus-original checks and independent model checks, tied to the tested build identity.
- **Performance:** conditions, timing boundary, repetitions/spread and raw evidence. Report both same-kernel unfused and real-system comparisons for fusion claims. Do not transfer timings between machines or sessions.
- **Status:** distinguish an external candidate, a buildable implementation, a correctness-qualified implementation and a measured winner. A winner is scoped to its tested workload and candidate set, never unconditional.

## Source ownership

Do not copy shared code into every machine/model directory. Keep reusable DSL operators in `etx/ops/`, import machinery in `etx/importer/`, and model graph/host wiring in `examples/`; reference them from the library. New machine/model-specific kernel sources or captures may live here with the records above. Move existing sources only as an explicit migration with build/test updates.

Do not place checkpoints, weight packs, generated build trees or large raw profiling dumps here. Link to their retained evidence/artifact locations. Do not label a diagnostic trace or a successful compile as a correctness-qualified benchmark.
