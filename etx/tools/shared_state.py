"""Local shared-state checks. These checks do not prove GPU memory visibility."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re

from etx.ir.instantiate import instantiate

ROOT = Path(__file__).resolve().parents[2]


def uncomment(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


def functions(text):
    """Extract preprocessed device functions with balanced braces."""
    text = uncomment(text)
    result = {}
    pattern = r"\b(?:void|float|bool|uint64_t|DslArgs)\s+(\w+)\s*\(([^;{}]*?)\)\s*\{"
    for match in re.finditer(pattern, text):
        start = match.end()
        depth, end = 1, start
        while depth and end < len(text):
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            raise ValueError(f"{match[1]}: incomplete source")
        result[match[1]] = (match[2], text[start:end - 1])
    return result


def preprocess(text, coherent=1, software=0):
    """Use the local C preprocessor. GPU headers and device code are not needed."""
    import shutil
    import subprocess
    compiler = shutil.which("clang") or shutil.which("cc")
    if not compiler:
        raise RuntimeError("shared-state gate needs a local C preprocessor")
    text = re.sub(r"^\s*#include[^\n]*", "", text, flags=re.M)
    flags = ["-DETX_EMIT_DSL_EXT=1", "-DETX_EMIT_DSL_BIGK=1"]
    if coherent:
        flags.append("-DETX_DSL_COH=1")
    if software:
        flags.append("-DETX_IMP_SW_BARRIER=1")
    return subprocess.run([compiler, "-E", "-P", "-x", "c++", *flags, "-"],
                          input=text, text=True, capture_output=True, check=True).stdout


def split_args(text):
    depth, start, result = 0, 0, []
    for i, ch in enumerate(text):
        depth += (ch in "([{") - (ch in ")]}")
        if ch == "," and depth == 0:
            result.append(text[start:i])
            start = i + 1
    return result + [text[start:]]


def calls(body, name):
    for match in re.finditer(rf"\b{name}(?:<[^;{{}}]+?>)?\s*\(", body):
        start, depth, end = match.end(), 1, match.end()
        while depth and end < len(body):
            depth += (body[end] == "(") - (body[end] == ")")
            end += 1
        yield split_args(body[start:end - 1])


def sources():
    from etx.ops import gemv, topk, attnprep
    return {
        "gemv": gemv.emit_tiles(), "topk": topk.emit_tiles(), "attnprep": attnprep.emit_tiles(),
        "native": (ROOT / "examples/vllm_llm/vl_tiles.hip").read_text(),
        "reduction": (ROOT / "etx/runtime/include/etx/tile_util.h").read_text(),
        "import": (ROOT / "etx/runtime/include/etx/import.h").read_text(),
    }


def lds_inventory(src=None):
    """Find arena aliases and propagate them through helper arguments.

    This is a source index, not a C++ alias proof. Access expressions remain in
    the report so that a reviewer can see lane and wave conditions in the source.
    """
    src = sources() if src is None else src
    fs = {}
    for coherent, software in ((1, 0), (0, 1)):
        for unit, text in src.items():
            for name, (args, body) in functions(preprocess(text, coherent, software)).items():
                if name in fs:
                    body = fs[name][2] + "\n" + body
                fs[name] = (unit, args, body)
    params = {name: [re.findall(r"\w+", a)[-1] for a in args.split(",") if a.strip()]
              for name, (_, args, _) in fs.items()}
    shared = {name: set() for name in fs}
    globals_ = set()
    for text in src.values():
        for decl in re.findall(r"__shared__[^;]+;", uncomment(text)):
            globals_.update(re.findall(r"\b(\w+)\s*(?:\[|[,;])", decl))
    for name, (_, _, body) in fs.items():
        shared[name].update(x for x in globals_ if re.search(rf"\b{x}\b", body))
    changed = True
    while changed:
        changed = False
        for name, (_, _, body) in fs.items():
            names = shared[name]
            # Pointer aliases include casts and offsets into the same LDS arena.
            for var, expr in re.findall(r"\*\s*(\w+)\s*=\s*([^;]+);", body):
                if "c->lds" in expr or any(re.search(rf"\b{x}\b", expr) for x in names):
                    if var not in names:
                        names.add(var)
                        changed = True
            for callee, (_, args, _) in fs.items():
                for actual in calls(body, callee):
                    for formal, arg in zip(params[callee], actual):
                        if any(re.search(rf"\b{x}\b", arg) for x in names):
                            if "*" in args.split(",")[params[callee].index(formal)] and formal not in shared[callee]:
                                shared[callee].add(formal)
                                changed = True
    sites = {}
    for name, names in shared.items():
        unit, _, body = fs[name]
        for var in sorted(names):
            writes, reads, aliases = set(), set(), set()
            for line in body.splitlines():
                line = line.strip()
                if not re.search(rf"\b{var}\b", line):
                    continue
                for access in re.finditer(rf"\b{var}\s*\[[^;]*?\]", line):
                    tail = line[access.end():].lstrip()
                    if re.match(r"(?:\+=|=(?!=))", tail):
                        writes.add(line)
                    if not re.match(r"=(?!=)", tail):
                        reads.add(line)
                if re.search(rf"__hip_atomic_(?:store|fetch_add)\(&{var}\b", line):
                    writes.add(line)
                if re.search(rf"__hip_atomic_(?:load|fetch_add)\(&{var}\b", line):
                    reads.add(line)
                if var in globals_ and re.search(rf"\b{var}\s*=(?!=)", line):
                    writes.add(line)
                if re.search(rf"\*\s*{var}\s*=|=.+\b{var}\s*[+;]", line):
                    aliases.add(line)
            sites[name, var] = dict(site=f"{unit}:{name}:{var}", writers=writes, readers=reads,
                                   aliases=aliases, edge=set(s.strip() for s in body.splitlines() if re.search(
                                       r"__syncthreads|barrier\(|fence\(|etx_block_(sum|max)\(|dsl_stage\(", s)))
    aliases = []
    for name, names in shared.items():
        for var, expr in re.findall(r"\*\s*(\w+)\s*=\s*([^;]+);", fs[name][2]):
            for base in names:
                if var in names and var != base and re.search(rf"\b{base}\b", expr):
                    aliases.append((name, var, base))
    # Report indirect effects through helper parameters. Keep each call as a source reference.
    changed = True
    while changed:
        changed = False
        for name, var, base in aliases:
            child, parent = sites[name, var], sites[name, base]
            for effect in ("writers", "readers"):
                ref = f"alias {var} in {name}"
                if child[effect] and ref not in parent[effect]:
                    parent[effect].add(ref)
                    changed = True
        for name, names in shared.items():
            for callee in fs:
                for actual in calls(fs[name][2], callee):
                    for formal, arg in zip(params[callee], actual):
                        child = sites.get((callee, formal))
                        if child is None:
                            continue
                        for var in names:
                            if not re.search(rf"\b{var}\b", arg):
                                continue
                            parent = sites[name, var]
                            for effect in ("writers", "readers"):
                                ref = f"{callee}:{formal} through {arg.strip()}"
                                if child[effect] and ref not in parent[effect]:
                                    parent[effect].add(ref)
                                    changed = True
                            if child["edge"]:
                                parent["edge"].add(f"helper {callee}:{formal}")
    for (name, var), site in sites.items():
        callers = sorted(caller for caller in fs if list(calls(fs[caller][2], name)))
        site["callers"] = callers
        if not site["edge"]:
            site["edge"].update(f"caller protocol: {caller}" for caller in callers)
    return [{key: sorted(value) if isinstance(value, set) else value for key, value in site.items()}
            for _, site in sorted(sites.items())]


def check_source(src=None):
    """Check both activation paths after the C preprocessor removes dead branches."""
    src = sources() if src is None else src
    return sorted(set(error for coherent in (0, 1) for error in _check_source(
        {name: preprocess(text, coherent) for name, text in src.items()})))


def _check_source(src):
    fs = {name: body for text in src.values() for name, (_, body) in functions(text).items()}
    errors = []

    def ordered(name, site, patterns):
        body = fs.get(name, "")
        at = 0
        for pattern in patterns:
            hit = re.search(pattern, body[at:], re.S)
            if not hit:
                errors.append(f"{name}:{site}: required write or barrier absent: {pattern}")
                return
            at += hit.end()

    bar = r"__syncthreads\(\);"
    ordered("dsl_stage", "xs", [r"xs\[8 \* i \+ 2 \* j\] =", r"xs\[8 \* i \+ 2 \* j \+ 1\] =", bar])
    # Both coherent and scalar branches must retain the round48 barrier.
    norm = fs.get("dsl_stage_norm_at", "")
    modes = re.findall(r"if \(d.mode == 4\) \{(.*?)float s1", norm, re.S)
    if len(modes) != 1 or any(not re.search(bar, mode) for mode in modes):
        errors.append("dsl_stage_norm_at:xs: LayerNorm mean lacks its staging barrier")
    ordered("dsl_stage_norm_at", "xs", [r"xs\[(?:8 \* i \+ j|i)\] =", bar, r"s1 \+= xs\[i\]", bar])
    vector = "xs[8 * i + j]" in norm
    for target, count in (("rout", 1), ("xout", 2 if vector else 1)):
        stores = re.findall(rf"if \(store && {target} != nullptr\) dsl_a_st(?:128|bf)\(", norm)
        if len(stores) != count:
            errors.append(f"dsl_stage_norm_at:{target}: required fold output write absent")
    for name in ("dsl_run_dsl_gemv", "dsl_run_dsl_gemv_fold"):
        ordered(name, "out", [r"r0 = blk \* d.rpt, r1 = min\(d.M, r0 \+ d.rpt\)",
                               r"dsl_gemv_chunked_nb<", r"dsl_a_stbf\(out \+"])
    for suffix in ("", "_fold"):
        ordered("dsl_run_dsl_moe_up" + suffix, "act", [
            r"S0 = blk / d.tpe", r"r0 = j \* d.rpt, r1 = min\(d.M, r0 \+ d.rpt\)",
            r"dsl_a_stbf\(actb \+ r", r"dsl_a_stbf\(act \+ r"])
        ordered("dsl_run_dsl_moe_down" + suffix, "out", [
            r"S0 = blk / d.tpe", r"r0 = j \* d.rpt, r1 = min\(d.M, r0 \+ d.rpt\)",
            r"dsl_a_stbf\(outb \+ r", r"dsl_a_stbf\(out \+ r"])
    ordered("dsl_topk_lds", "ids/wts", [r"ids\[t\] =", r"wts\[t\] =", bar])
    for name in ("etx_block_sum", "etx_block_max"):
        ordered(name, "red", [r"red\[etx_wave\(\)\] =", bar, r"red\[w\]", bar])
    for name in ("dsl_gemv_rows_nb", "dsl_gemv_split"):
        ordered(name, "part", [r"pl\[i\] =", bar,
                                r"(?:pl\[.*?\] \+=|dsl_gemv_accum<)", bar,
                                r"(?:v\[b\]|v) \+= pl\[", r"epi\(", bar])
    ordered("dsl_gemv_bigk", "xs/part", [r"pl\[i\] =", r"dsl_stage\(", r"dsl_gemv_accum<", bar,
                                         r"v \+= pl\[", r"epi\(", bar])
    ordered("vl_argmax", "sv/si", [r"sv\[threadIdx.x\] =", r"si\[threadIdx.x\] =", bar,
                                   r"sv\[threadIdx.x \+ s\]", r"si\[threadIdx.x\] = oi", bar])
    ordered("etx_imp_counted_barrier", "etx_lds_arena", [r"__builtin_amdgcn_fence\(3,", r"__builtin_amdgcn_s_barrier\(\)", r"__builtin_amdgcn_fence\(2,"])
    ordered("etx_imp_rounds_end", "etx_imp_exit", [r"__hip_atomic_store\(&etx_imp_exit", r"etx_imp_counted_barrier\(\)", r"__hip_atomic_load\(&etx_imp_exit"])
    return errors


@dataclass
class Contract:
    """Keep API requirements separate from mutable compiler output."""
    graph: object
    launches: list
    dims: object
    imports: dict
    maxlen: int

    @classmethod
    def capture(cls, graph, launches, dims, imports, maxlen):
        import copy
        return cls(*copy.deepcopy((graph, launches, dims, imports, maxlen)))


def read_launches(path):
    fields = "id export layer role gx gy gz offset tasks M K wv rep stride copyoff req nb".split()
    return [{k: v if k in ("export", "role") else int(v) for k, v in zip(fields, line.split())}
            for line in Path(path).read_text().splitlines()[2:]]


def _blocks(plan, launch, info, tasks_by_grid):
    """Use the adapter's coordinate rules. Never infer writes from task count alone."""
    for grid in plan.graph.grids:
        if "impargs" not in grid.args or launch["offset"] not in grid.consts[:1] and not (
                grid.name.startswith("chain_") and launch["offset"] in grid.consts):
            continue
        for task in tasks_by_grid.get(grid.name, []):
            sym = grid.body.symbol
            if "_rep_" in sym:
                for block in range(launch["gx"] * launch["gy"] * launch["gz"]):
                    yield task.coord[0], block, task
            elif "chain_" in sym:
                for block in range(launch["gx"] * launch["gy"] * launch["gz"]):
                    yield 0, block, task
            else:
                first = (task.coord[0] * grid.consts[2] + task.coord[1]) if len(task.coord) == 2 else task.coord[0]
                for slot in range(info["slots"]):
                    block = first * info["slots"] + slot
                    if block < launch["gx"] * launch["gy"] * launch["gz"]:
                        yield 0, block, task


def coverage(plan, contract):
    """Visit each output element. Required extents come from model dimensions."""
    import numpy as np
    d = contract.dims
    errors, report = [], []
    tasks_by_grid = {}
    for task in plan.tasks:
        tasks_by_grid.setdefault(task.grid, []).append(task)
    for launch in contract.launches:
        role, exp = launch["role"], launch["export"]
        site = f"r{launch['req']}/L{launch['layer']}/{role}"
        blocks = list(_blocks(plan, launch, contract.imports[exp], tasks_by_grid))
        nb = launch["nb"]
        replicas = launch["rep"] if any("_rep_" in plan.graph.grid(t.grid).body.symbol for _, _, t in blocks) else 1
        if role in ("qkv", "o_proj", "router", "lm_head"):
            size = {"qkv": (d.NH + 2 * d.NKV) * d.HD, "o_proj": d.H, "router": d.E, "lm_head": d.V}[role]
            if not exp.startswith("dsl_gemv"):
                raise ValueError(f"{site}: unsupported coverage body {exp}")
            width = math.ceil(launch["M"] / (launch["gx"] * launch["gy"] * launch["gz"]))
            mask = np.zeros((nb, size), dtype=np.uint16)
            for _, block, _ in blocks:
                start, stop = block * width, min(launch["M"], (block + 1) * width)
                mask[:, start:stop] += 1
        elif role in ("moe_up", "moe_down"):
            size, width = (d.MI, 32) if role == "moe_up" else (d.H, 64)
            mask = np.zeros((nb * d.TOPK, size), dtype=np.uint16)
            for _, block, _ in blocks:
                slot, row = divmod(block, size // width)
                if slot >= len(mask):
                    errors.append(f"{site}: output slot {slot} exceeds {len(mask)}")
                else:
                    mask[slot, row * width:(row + 1) * width] += 1
        else:
            parts = math.ceil(contract.maxlen / 256)
            shape = {
                "in_norm": (replicas, d.H), "post_norm": (replicas, d.H), "final_norm": (replicas, d.H),
                "q_norm": (d.NH, d.HD), "k_norm": (d.NKV, d.HD),
                "rope": (1, (d.NH + d.NKV) * d.HD), "cache": (1, 2 * d.NKV * d.HD),
                "prep": (d.NKV, (d.G + 3) * d.HD),
                "attn": (d.NKV * parts, d.G * (d.HD + 2)), "reduce": (d.NH, d.HD),
                "topk": (replicas, nb * 3 * d.TOPK), "moe_sum": (replicas, d.H),
            }
            if role not in shape:
                raise ValueError(f"{site}: no output contract for {exp}")
            mask = np.zeros(shape[role], dtype=np.uint16)
            for replica, block, _ in blocks:
                row = replica if role in ("in_norm", "post_norm", "final_norm", "topk", "moe_sum") else block
                if row >= len(mask):
                    errors.append(f"{site}: output row {row} exceeds {len(mask)}")
                else:
                    mask[row] += 1
        holes = np.argwhere(mask == 0)
        if len(holes):
            errors.append(f"{site}: missing write at {tuple(map(int, holes[0]))}, {len(holes)} unwritten elements")
        if np.any(mask > 1):
            errors.append(f"{site}: multiple writers to one output element")
        grids = sorted({t.grid for _, _, t in blocks})
        edges = sorted({e for name in grids for e in plan.graph.grid(name).out_edges})
        readers = sorted({g.name for g in plan.graph.grids if set(g.in_edges) & set(edges)})
        report.append(dict(site=site, elements=int(mask.size), writers=grids, readers=readers, edge=edges))
        # Only block zero stores the complete fold side outputs.
        if exp == "dsl_gemv_fold" and (role == "router" or role == "qkv" and launch["layer"] > 0):
            side = np.full((nb, d.H), sum(block == 0 for _, block, _ in blocks), dtype=np.uint16)
            for field in (["resid_out", "xout"] if role == "router" else ["resid_out"]):
                if not np.all(side == 1):
                    errors.append(f"{site}/{field}: missing write or duplicate writer")
                report.append(dict(site=f"{site}/{field}", elements=int(side.size),
                                   writers=[f"{name}:block0" for name in grids], readers=readers, edge=edges))
    for grid in contract.graph.grids:
        if grid.body.symbol not in ("vl_prologue", "vl_argmax"):
            continue
        count = sum(t.grid == grid.name for t in plan.tasks)
        # Prologue: residual replicas plus position, length, slot, and start time.
        # Argmax: token_out, optional token_in, and end time.
        size = d.H * (8 if any(x["rep"] == 8 for x in contract.launches) else 1) + 4 if grid.body.symbol == "vl_prologue" else 3
        mask = np.full(size, count, dtype=np.uint16)
        if not np.all(mask == 1):
            errors.append(f"{grid.name}: missing write or duplicate writer in native output")
        edges = list(grid.out_edges)
        report.append(dict(site=grid.name, elements=size, writers=[grid.name],
                           readers=[g.name for g in plan.graph.grids if set(g.in_edges) & set(edges)], edge=edges))
    return errors, report


def check_plan(plan, contract):
    """Check rows, required API edges, event counts, and task membership."""
    errors, report = coverage(plan, contract)
    expected = instantiate(contract.graph, plan.bindings, plan.runtime)
    actual = instantiate(plan.graph, plan.bindings, plan.runtime)
    tasks = Counter((t.grid, t.coord) for t in plan.tasks)
    placed_tasks = {(t.grid, t.coord): t for t in plan.tasks}
    queued = Counter(tid for queue in plan.static_queues.values() for tid in queue)
    for task in plan.tasks:
        if task.mode == "static" and queued[task.id] != 1:
            errors.append(f"{task.grid}{task.coord}: required write task has {queued[task.id]} queue entries")
    extra = set(tasks) - set(expected.all_tasks())
    for tid in sorted(extra):
        errors.append(f"{tid[0]}{tid[1]}: unexpected writer task")
    for tid in expected.all_tasks():
        site = tid[0]
        if tasks[tid] != 1:
            errors.append(f"{site}{tid[1]}: required write task count is {tasks[tid]}")
        for attr in ("task_in", "task_out"):
            required = set(getattr(expected, attr)[tid])
            for observed in (getattr(actual, attr), getattr(plan.inst, attr)):
                if not required <= set(observed.get(tid, [])):
                    errors.append(f"{site}: required event edge absent in {attr}")
    for ev, writers in expected.producers.items():
        name, coord = ev
        ep = plan.events.get(name)
        if ep is None or len(ep.counts) <= plan.linear(name, coord) or ep.counts[plan.linear(name, coord)] != len(writers):
            errors.append(f"{name}: required event count differs from writers")
        if ep is not None:
            from etx.ir import Scope
            participants = writers + expected.consumers[ev]
            placed = {(t.device, t.domain) for tid in participants if tid in placed_tasks for t in [placed_tasks[tid]]}
            if len(placed) > 1 and ep.scope < Scope.DEVICE:
                errors.append(f"{name}: shared output needs DEVICE scope")
    # A non-root phase must have an input event. This also checks the API graph.
    for grid in plan.graph.grids:
        if not grid.name.endswith("prologue") and not grid.in_edges:
            errors.append(f"{grid.name}: required input event absent")
    return errors, report


def argument_inventory(plan, contract):
    """Derive byte regions from launch offsets and the import ABI."""
    result = []
    for launch in contract.launches:
        info = contract.imports[launch["export"]]
        for replica in range(launch["rep"]):
            start = launch["offset"] + replica * launch["stride"]
            readers = [g for g in plan.graph.grids if "impargs" in g.args and
                       (g.consts[0] == launch["offset"] or
                        "chain_" in g.body.symbol and launch["offset"] in g.consts)]
            result.append(dict(site=f"impargs:{launch['export']}:L{launch['layer']}:r{launch['req']}:copy{replica}",
                               start=start, stop=start + info["arg_bytes"],
                               layout=launch["export"],
                               writers=["host.hip:Args", "host.hip:etx_host::upload(ab.data(), ab.size())"],
                               readers=sorted({g.body.symbol for g in readers} |
                                              {g.body.pre_wait for g in readers if g.body.pre_wait}),
                               edge=["host upload completes before kernel launch. Block stays immutable during the step"],
                               task_events=sorted({e for g in readers for e in g.in_edges})))
    return result


def check_arguments(plan, contract):
    errors = []
    for grid in contract.graph.grids:
        if "impargs" in grid.args:
            try:
                actual = plan.graph.grid(grid.name)
            except KeyError:
                errors.append(f"{grid.name}: argument reader absent")
                continue
            if actual.consts != grid.consts:
                errors.append(f"{grid.name}: argument offset or replica stride differs from the host layout")
    end = 0
    for site in sorted(argument_inventory(plan, contract), key=lambda x: x["start"]):
        if site["start"] < end or site["stop"] > plan.graph.tensors["impargs"].shape[0]:
            errors.append(f"{site['site']}: argument blocks overlap or exceed allocation")
        if not site["readers"]:
            errors.append(f"{site['site']}: no task reads the argument block")
        for field in contract.imports[site["layout"]].get("params", []):
            if not 0 <= field["offset"] < field["offset"] + field["size"] <= site["stop"] - site["start"]:
                errors.append(f"{site['site']}:{field['name']}: field exceeds argument block")
        end = site["stop"]
    return errors


def build_local(model, directory, layers=None):
    """Build default batch-one graphs with source-only import descriptors.

    Weights have no effect on addresses in these checks. Do not allocate them.
    The imported ABI sizes match the existing test_vllm_moe fixtures.
    """
    import importlib.util
    import json
    import os
    from etx.ops import topk, attnprep
    from etx.passes import compile_graph
    from examples.llm.model import load_dims
    from etx.importer.vllm_kernels import SOURCES

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved = dict(os.environ)
    try:
        for key in list(os.environ):
            if key.startswith(("ETX_VL_", "ETX_LLM_")):
                del os.environ[key]
        os.environ["ETX_LLM_CONFIG"] = str(ROOT / "examples/llm/configs" / f"{model}.json")
        if layers is not None:
            os.environ["ETX_LLM_LAYERS"] = str(layers)
        d = load_dims()
        os.environ["ETX_VL_DSL_EXT"] = "1" if getattr(d, "ln_norm", False) else "0"

        def module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        gm = module("coverage_gemv", ROOT / "etx/ops/gemv.py")
        sizes = {"rms_norm_2d": 100, "rope_neox": 108, "reshape_and_cache": 96,
                 "attn_mfma4_g4": 160, "attn_mfma16_g8": 160, "attn_reduce": 80}
        imp = {}
        for source in SOURCES:
            for kernel in source.kernels:
                if kernel.export not in sizes:
                    continue
                block = math.prod(kernel.block)
                if kernel.export == "rms_norm_2d":
                    block = min(d.H // 8, 1024)
                elif kernel.export == "reshape_and_cache":
                    block = min(d.NKV * 16, 512)
                slots = 1024 // block
                if kernel.export == "attn_mfma16_g8" and d.qk_norm:
                    slots = 1
                imp[kernel.export] = dict(slots=slots, arg_bytes=sizes[kernel.export])
        for mod in (gm, topk, attnprep):
            imp.update(mod.import_entries())
        imports_path = directory / "imports.json"
        imports_path.write_text(json.dumps(imp))
        os.environ.update(ETX_VL_IMPORTS=str(imports_path), ETX_VL_PLAN=str(directory / "vl_plan.txt"),
                          ETX_VL_CHAINS=str(directory / "chains.hip"))
        vm = module("coverage_model", ROOT / "examples/vllm_llm/model.py")
        graph = vm.build()
        contract = Contract.capture(graph, read_launches(directory / "vl_plan.txt"), d, imp, vm.MAXLEN)
        return compile_graph(graph, "gfx942", {}, {}), contract
    finally:
        os.environ.clear()
        os.environ.update(saved)


def main():
    import argparse
    import json
    import tempfile
    parser = argparse.ArgumentParser(description="Check shared state without a GPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, help="Use a shorter graph for a fault test")
    args = parser.parse_args()
    errors = check_source()
    import hashlib
    report = dict(lds=lds_inventory(), graphs=[], arguments=[], gpu_executed=False,
                  source_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources().items()})
    with tempfile.TemporaryDirectory() as temp:
        for model in ("qwen3-30b-a3b", "mixtral-8x7b", "phi-3.5-moe"):
            print(f"CHECK {model}", flush=True)
            plan, contract = build_local(model, Path(temp) / model, args.layers)
            failures, outputs = check_plan(plan, contract)
            errors += failures + check_arguments(plan, contract)
            report["graphs"].append(dict(model=model, layers=contract.dims.L, tasks=len(plan.tasks), outputs=outputs))
            report["arguments"].append(dict(model=model, layouts=contract.imports, sites=argument_inventory(plan, contract)))
    report.update(errors=errors, passed=not errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'FAIL' if errors else 'PASS'}: {len(errors)} errors; inventory: {args.output}", flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
