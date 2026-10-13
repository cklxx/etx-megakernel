"""D4 expert placement (docs/DSV2-D1-D5.md): the shared XCDs start without routing, routed slices go to them, and each
slice's gate_up and down stay on one XCD. The build refuses any placement that breaks this.

Units of one MoE layer: routing slots 0..5 (one routed expert each, `inter` = 1408 rows of h) and the two shared halves
(1408 rows each). With D4, routed XCD k (= slot k) keeps rows [rows, inter) of slot k; shared XCD 6 + x runs its shared
half first -- with its own fold + norm and no routing, so it starts when o_proj completes -- then rows [0, rows) of slots
[x * SPX, (x + 1) * SPX) once the routed XCDs have published their routing (an in-body wait), and one down for all of it.

`check(stages)` is the rule, `verify_graph(graph, L, stages)` checks that the ETX grids really place every stage where
the table says (domain_map evaluated per task coordinate); graph.build() runs both, so `etx compile` refuses.

The rules:
  same chiplet   each slice's gate_up and down run on the same XCD: h is XCD-local (written by gate_up into L2, read by
                 down after an XCD-local event); a split slice would read h across XCDs behind a DOMAIN event;
  exact cover    for each unit and each stage the slices' row ranges cover [0, inter) exactly once;
  alignment      every row boundary a multiple of 32 (128 B of fp32 h: no cache line written by two XCDs; it is also a
                 multiple of 8, the 16-byte weight loads);
  no routing     the first stage on a shared XCD (its shared half's gate_up) does not depend on the layer's routing.
"""
from __future__ import annotations

from dataclasses import dataclass

INTER = 1408          # moe_intermediate_size of DeepSeek-V2-Lite: rows of h per routed expert and per shared half
TOP_K = 6             # routing slots = routed XCDs 0..5
SHARED = 2            # shared halves = shared XCDs 6, 7
SPX = TOP_K // SHARED  # routed slots whose slice one shared XCD runs
ALIGN = 32
ROWS = 32             # default slice rows per routed expert (estimate in docs/DSV2-D1-D5.md, D4)


class PlacementError(ValueError):
    pass


@dataclass(frozen=True)
class Stage:
    unit: str        # "slot<k>" (routed expert in routing slot k) or "shared<h>"
    lo: int          # rows [lo, hi) of the unit's h
    hi: int
    stage: str       # "gate_up" | "down"
    xcd: int
    grid: str        # the per-layer grid that runs it (graph.py: <grid>_<L>)
    coord: int       # its coordinate 0 in that grid
    routing: bool    # needs the layer's routing


def shared_xcd(h: int) -> int:
    return TOP_K + h


def layout(rows: int = ROWS) -> list[Stage]:
    out = []
    for k in range(TOP_K):
        sx = shared_xcd(k // SPX)
        out += [Stage(f"slot{k}", rows, INTER, "gate_up", k, "gate_up", k, True),
                Stage(f"slot{k}", rows, INTER, "down", k, "down", k, True)]
        if rows:
            out += [Stage(f"slot{k}", 0, rows, "gate_up", sx, "shgu", k // SPX, True),
                    Stage(f"slot{k}", 0, rows, "down", sx, "sdown", k // SPX, True)]
    for h in range(SHARED):
        out += [Stage(f"shared{h}", 0, INTER, "gate_up", shared_xcd(h), "shgu", h, False),
                Stage(f"shared{h}", 0, INTER, "down", shared_xcd(h), "sdown", h, False)]
    return out


def check(stages: list[Stage], inter: int = INTER) -> None:
    by_slice: dict[tuple[str, int, int], dict[str, Stage]] = {}
    for s in stages:
        if s.stage not in ("gate_up", "down"):
            raise PlacementError(f"{s.unit} rows [{s.lo}, {s.hi}): unknown stage {s.stage!r}")
        if not 0 <= s.lo < s.hi <= inter:
            raise PlacementError(f"{s.unit} rows [{s.lo}, {s.hi}): outside [0, {inter}) or empty")
        for b in (s.lo, s.hi):
            if b not in (0, inter) and b % ALIGN:
                raise PlacementError(f"{s.unit} row boundary {b} is not a multiple of {ALIGN} (an h cache line would "
                                     f"be written by two XCDs)")
        key = (s.unit, s.lo, s.hi)
        if s.stage in by_slice.setdefault(key, {}):
            raise PlacementError(f"{s.unit} rows [{s.lo}, {s.hi}): {s.stage} placed twice")
        by_slice[key][s.stage] = s
    for (unit, lo, hi), st in sorted(by_slice.items()):
        if set(st) != {"gate_up", "down"}:
            raise PlacementError(f"{unit} rows [{lo}, {hi}): has {sorted(st)} only; a slice needs gate_up and down")
        if st["gate_up"].xcd != st["down"].xcd:
            raise PlacementError(f"split slice: {unit} rows [{lo}, {hi}) gate_up on XCD {st['gate_up'].xcd} but down on "
                                 f"XCD {st['down'].xcd} (h is XCD-local: both stages must run on the same chiplet)")
    for unit in sorted({s.unit for s in stages}):
        for stage in ("gate_up", "down"):
            spans = sorted((s.lo, s.hi) for s in stages if s.unit == unit and s.stage == stage)
            at = 0
            for lo, hi in spans:
                if lo != at:
                    raise PlacementError(f"{unit} {stage}: rows [{min(lo, at)}, {max(lo, at)}) "
                                         f"{'uncovered' if lo > at else 'covered twice'}")
                at = hi
            if at != inter:
                raise PlacementError(f"{unit} {stage}: rows [{at}, {inter}) uncovered")
    for h in range(SHARED):
        first = [s for s in stages if s.xcd == shared_xcd(h) and s.unit == f"shared{h}" and s.stage == "gate_up"]
        if len(first) != 1 or first[0].routing:
            raise PlacementError(f"shared XCD {shared_xcd(h)} must start with shared half {h}'s gate_up and no routing")


def verify_graph(g, L: int, stages: list[Stage], router_events: set[str]) -> None:
    """The grids of layer L place every stage on its XCD (all 37 workers of its coordinate), the shared gate_up waits on
    no routing event, and per worker the shared stages run in order (shared half first)."""
    from etx.ir.edgemap import EdgeMap
    names = [gr.name for gr in g.grids]
    for s in stages:
        grid = g.grid(f"{s.grid}_{L}")
        pin = EdgeMap.parse(grid.domain_map)
        workers = int(grid.grid[1])
        doms = {pin.targets((s.coord, w), (8,), None, {})[0][0] % 8 for w in range(workers)}
        if doms != {s.xcd}:
            raise PlacementError(f"{grid.name} coordinate {s.coord} ({s.unit} rows [{s.lo}, {s.hi}) {s.stage}) runs on "
                                 f"XCD {sorted(doms)}, the placement says XCD {s.xcd}")
    for gr in (g.grid(f"shgu_{L}"),):
        bad = set(gr.in_edges) & router_events
        if bad:
            raise PlacementError(f"{gr.name} waits on the routing ({sorted(bad)}): the shared XCDs must start without it")
    if names.index(f"shgu_{L}") > names.index(f"sdown_{L}"):
        raise PlacementError(f"layer {L}: the shared down is queued before the shared gate_up")


def trace_lead(plan, trace) -> dict:
    """GPU criterion (a) from one traced token of the d4 build: per MoE layer, how long before the layer's routing
    completes (last router task done) the shared XCDs' gate_up became ready (first shgu task)."""
    from examples.dsv2lite import ledger as ld
    tasks = ld.load_etx(plan, trace, None)
    leads = {}
    for L in sorted({t.layer for t in tasks if t.phase == "shgu"}):
        routed_done = max(t.done for t in tasks if t.layer == L and t.phase == "router")
        shared_ready = min(t.ready for t in tasks if t.layer == L and t.phase == "shgu")
        leads[L] = routed_done - shared_ready
    if not leads:
        raise ValueError("no shgu tasks in this trace: not a d4 build")
    vals = sorted(leads.values())
    return dict(layers=len(vals), before_routing=sum(v > 0 for v in vals), median_lead_us=vals[len(vals) // 2],
                min_lead_us=vals[0], leads_us={str(k): v for k, v in leads.items()})
