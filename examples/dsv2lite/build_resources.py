"""Read ROCm resource remarks for ETX's persistent kernel (not fleet's unused kernel)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIELDS = {
    "vgpr": "VGPRs", "agpr": "AGPRs", "sgpr": "TotalSGPRs",
    "scratch": "ScratchSize [bytes/lane]", "lds": "LDS Size [bytes/block]",
    "occupancy": "Occupancy [waves/SIMD]",
}


def parse(text: str) -> dict[str, int]:
    parts = re.split(r"remark: Function Name: ", text)
    matches = [p for p in parts[1:] if p.split()[0] == "etx_megakernel_d0"]
    if len(matches) != 1:
        raise ValueError("expected exactly one etx_megakernel_d0 resource record")
    result = {}
    for key, label in FIELDS.items():
        values = re.findall(r"remark:\s+" + re.escape(label) + r": (\d+)", matches[0])
        if len(values) != 1:
            raise ValueError(f"missing/duplicate {label}")
        result[key] = int(values[0])
    return result


# Per-CU budgets of the target (a SIMD's register file is 512 registers per lane, VGPRs and AGPRs from one pool; 64 KiB
# of LDS per CU; four SIMDs per CU). A 256-thread workgroup puts one 64-lane wave on each SIMD.
REGS_PER_SIMD_LANE, LDS_PER_CU, SIMDS_PER_CU, WAVE_LANES = 512, 65536, 4, 64


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def wg_per_cu(res: dict, threads: int = 256) -> int:
    """Workgroups of `threads` that fit one CU, from the record's registers and LDS, never above the compiler's own
    occupancy figure. Registers: AGPRs start at the VGPR count rounded to 4, the total is allocated in granules of 8
    (248 + 6 AGPR -> 256: 2 per CU; 250 + 6 -> 264: 1 per CU)."""
    waves_per_wg = -(-threads // WAVE_LANES)
    waves_per_simd_per_wg = -(-waves_per_wg // SIMDS_PER_CU)
    regs = _align(_align(res["vgpr"], 4) + res["agpr"], 8)
    by_regs = REGS_PER_SIMD_LANE // max(regs, 1) // waves_per_simd_per_wg
    by_lds = LDS_PER_CU // res["lds"] if res["lds"] else by_regs
    by_compiler = res["occupancy"] // waves_per_simd_per_wg
    return min(by_regs, by_lds, by_compiler)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    args = p.parse_args(argv)
    rows = {}
    # Every arm directory rocm_build_check.sh wrote (default first, so deltas read against it).
    for path in sorted(args.out.glob("*/resource.log"), key=lambda p: (p.parent.name != "default", p.parent.name)):
        rows[path.parent.name] = parse(path.read_text())
    base = rows.get("default")
    print("arm | VGPR | AGPR | SGPR | scratch B/lane | LDS B | waves/SIMD | delta vs default (same order)")
    for arm, row in rows.items():
        delta = [row[k] - base[k] for k in FIELDS] if base else None
        print(arm + " | " + " | ".join(str(row[k]) for k in FIELDS) + " | " + str(delta))
    (args.out / "resources.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
