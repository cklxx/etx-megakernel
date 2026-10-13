"""CPU oracle for `hw_bench moerows` (bench/hw/hw_bench.hip, k_moerows): the row/address coverage and cache coldness
the GPU run relies on. It mirrors the kernel's index formulas one to one; it cannot say anything about throughput --
HBM channel and cache behaviour under each address pattern exists only on the device.

  python bench/hw/moerows_oracle.py      prints the checks; exit 0 = all hold
"""
from __future__ import annotations

ITERS, ACTIVE, TPE, NEXP, ROWS, FIRST = 12, 192, 24, 128, 32, 2     # the kernel's moerows host constants
ROW = 4096                                                          # bytes per row: 2048 bf16
HALF = 768 * ROW                                                    # 768 rows per gate (and up) half
ESTRIDE = 2 * HALF
BUF = 2 << 30                                                       # hw_bench's weight buffer
MALL, L2_TOTAL = 256 << 20, 8 * (4 << 20)                           # Infinity Cache, 8 XCD L2s


def expert(s: int, it: int) -> int:
    return (s * 37 + 11 + it * 13) % NEXP


def task_rows(j: int, interleaved: bool) -> list[int]:
    first, step = (j, TPE) if interleaved else (j * ROWS, 1)      # k_moerows: row = first + r * step
    return [first + r * step for r in range(ROWS)]


def check() -> list[str]:
    out = []
    assert ACTIVE == (ACTIVE // TPE) * TPE and TPE * ROWS == HALF // ROW, "tasks must cover each half exactly"
    for il in (False, True):
        for j in range(TPE):
            assert len(set(task_rows(j, il))) == ROWS
        cover = sorted(r for j in range(TPE) for r in task_rows(j, il))
        assert cover == list(range(HALF // ROW)), f"interleaved={il}: an expert half is not covered exactly once"
    assert all(task_rows(j, False) != task_rows(j, True) for j in range(1, TPE)), "the two modes must differ"
    out.append(f"OK coverage: each expert half's {HALF // ROW} rows read exactly once per pass in both modes; "
               f"{ROWS} rows = {ROWS * ROW * 2 // 1024} KB per task in both")
    last, read = {}, 0
    per_pass = ACTIVE * 2 * ROWS * ROW
    worst = None
    for it in range(ITERS):
        experts = {expert(s, it) for s in range(ACTIVE // TPE)}
        for e in experts:
            if it >= FIRST and e in last:
                gap = read - last[e]
                worst = gap if worst is None else min(worst, gap)
                assert gap > MALL + L2_TOTAL, f"pass {it}: expert {e} re-read after only {gap >> 20} MB"
        for e in experts:
            last[e] = read
        read += per_pass
    out.append(f"OK cold: every re-read expert in passes {FIRST}..{ITERS - 1} follows >= {worst >> 20} MB of other reads "
               f"(> {(MALL + L2_TOTAL) >> 20} MB = MALL + all L2s)" if worst is not None else
               f"OK cold: no expert is re-read in passes {FIRST}..{ITERS - 1}")
    top = max(expert(s, it) for s in range(ACTIVE // TPE) for it in range(ITERS)) * ESTRIDE + 2 * HALF
    assert top <= BUF, "addresses leave the buffer"
    out.append(f"OK bounds: highest byte {top >> 20} MB < {BUF >> 20} MB buffer; {per_pass / 2**20:.1f} MB per pass")
    return out


if __name__ == "__main__":
    print("\n".join(check()))
