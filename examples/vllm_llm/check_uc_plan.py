#!/usr/bin/env python3
"""Refuse an ETX_VL_UC plan that would read stale KV-cache lines.

An ETX_VL_UC build compiles DEVICE events with DOMAIN fences (ETX_COHERENT_DEVICE_DATA). That is only correct if
every tensor crossing XCDs is uncached, and the KV cache is not: it relies on its writer (cache / chain) and its
readers (attn, reduce) all running on XCD 0. This checks, on the compiled plan, that every task is static
(no queue could move one) and that every task of those grids is placed on domain 0.

  python examples/vllm_llm/check_uc_plan.py build/vl/<model>_uc/plan.json
"""
import json
import sys

KV_PREFIXES = ("cache_", "attn_", "reduce_", "chain_")


def main(path: str) -> int:
    plan = json.load(open(path))
    types = plan["types"]
    bad = [n for n, t in types.items() if t["mode"] != "static"]
    kv_ids = {t["id"]: n for n, t in types.items() if n.startswith(KV_PREFIXES)}
    off = sorted({kv_ids[d["type"]] for d in plan["descs"] if d["type"] in kv_ids and d["domain"] != 0})
    if bad or off or not kv_ids:
        print(f"UC PLAN REJECTED: non-static grids {bad[:5]}, KV grids off XCD 0 {off[:5]}, KV grids found {len(kv_ids)}")
        return 1
    print(f"UC plan ok: all {len(types)} grids static, {len(kv_ids)} KV-cache grids entirely on XCD 0")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
