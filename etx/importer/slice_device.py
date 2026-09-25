"""Keep only the device side of a CUDA/HIP extension source.

vLLM's .cu files mix kernel templates with torch host wrappers (at::Tensor, TORCH_CHECK, AT_DISPATCH...).
The importer only needs the kernels, compiled for the device, so this removes every top-level definition
whose text mentions the torch API, and the torch includes, leaving the kernel code byte-for-byte as it is.

  python -m etx.importer.slice_device in.cu out.hip [--drop REGEX]

--drop removes, in addition, every top-level definition matching REGEX (kernels the import does not need
and whose dependencies, e.g. the fp8 headers, would otherwise have to be provided).
"""
from __future__ import annotations

import re
import sys

TORCH = re.compile(r"\b(at::|torch::|c10::(?!Half|BFloat16)|TORCH_CHECK|AT_DISPATCH|STD_TORCH_CHECK|"
                   r"torch_utils|OptionalCUDAGuard|getCurrentCUDAStream|THO_DISPATCH|VLLM_DISPATCH|"
                   r"torch::stable|Tensor\b)")
DROP_INCLUDE = re.compile(r'#\s*include\s*[<"](torch/|ATen/|c10/|torch_utils\.h|libtorch_stable/|core/registration)')


def _top_level_chunks(src: str):
    """Split into top-level chunks: preprocessor lines, and declarations/definitions up to their end
    (';' at depth 0, or the closing '}' of a body at depth 0). Comments and strings are respected."""
    i, n, start, depth = 0, len(src), 0, 0
    out = []
    pending = False                                            # a code chunk has started (non-comment text seen)
    while i < n:
        c = src[i]
        if c == "/" and src.startswith("//", i):
            j = src.find("\n", i); j = n if j < 0 else j
            if depth == 0 and not pending: out.append(("pp", src[start:j])); start = j
            i = j; continue
        if c == "/" and src.startswith("/*", i):
            j = src.find("*/", i + 2); j = n if j < 0 else j + 2
            if depth == 0 and not pending: out.append(("pp", src[start:j])); start = j
            i = j; continue
        if c == "#" and (i == 0 or src[i - 1] == "\n" or src[:i].rsplit("\n", 1)[-1].strip() == "") and (depth > 0 or pending):
            # a directive inside a chunk: count braces in the first branch of #if/#else only
            d = re.match(r"#\s*(\w+)", src[i:])
            if d and d.group(1) in ("else", "elif"):
                lvl, j = 1, i
                while lvl and j < n:
                    e = src.find("\n", j); j = n if e < 0 else e + 1
                    m = re.match(r"[ \t]*#\s*(\w+)", src[j:])
                    if m:
                        if m.group(1) in ("if", "ifdef", "ifndef"): lvl += 1
                        elif m.group(1) == "endif": lvl -= 1
                i = j; continue
            e = src.find("\n", i); i = n if e < 0 else e; continue
        if not c.isspace() and c != "#": pending = True
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            i = j + 1; continue
        if c == "#" and depth == 0 and not pending:
            j = i
            while True:                                        # continued preprocessor lines
                e = src.find("\n", j)
                if e < 0: e = n; break
                if src[e - 1] == "\\": j = e + 1; continue
                break
            out.append(("pp", src[start:e + 1])); start = i = e + 1; continue
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                k = i + 1                                      # a body ends the chunk unless ';' follows (struct/class)
                m = re.match(r"\s*;", src[k:])
                if m: k += m.end()
                out.append(("code", src[start:k])); start = i = k; pending = False; continue
        elif c == ";" and depth == 0:
            out.append(("code", src[start:i + 1])); start = i + 1; pending = False
        i += 1
    if src[start:].strip():
        out.append(("code", src[start:]))
    return out


def slice_device(src: str, drop: str | None = None) -> str:
    extra = re.compile(drop) if drop else None
    keep = []
    for kind, text in _top_level_chunks(src):
        if kind == "pp":
            keep.append("// [etx: dropped] " + text.strip().replace("\n", " ") + "\n" if text.lstrip().startswith("#") and DROP_INCLUDE.search(text) else text)
            continue
        body = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
        if TORCH.search(body):
            keep.append("\n// [etx: dropped a torch host definition]\n")
        elif extra and extra.search(body):
            keep.append("\n// [etx: dropped by --drop]\n")
        else:
            keep.append(text)
    return "".join(keep)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--drop", default=None)
    a = ap.parse_args()
    open(a.out, "w").write(slice_device(open(a.src).read(), a.drop))
