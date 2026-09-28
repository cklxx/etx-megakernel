"""Model check of the imported-kernel barrier protocol (etx/runtime/include/etx/import.h, hardware barriers in rounds):
16 waves, s_barrier semantics (a barrier completes when all 16 waves have arrived), and a random interleaving of every
wave's shared-memory operations between barriers. Each wave runs several uses (chained launches): its block's barriers (a
random count per slot, 0 for idle waves, early returns), then the round protocol. The protocol must never deadlock or
livelock and every wave must take the same number of barriers. The first version (leave when the done count reads 16,
reset at the next begin) fails this check: the race a review found (2026-09-28)."""
import random

def run(protocol, seed, uses=4, nw=16):
    rng = random.Random(seed)
    sh = {"done": 0, "exit": 0, "nbar": [0] * nw}
    plans = []
    for u in range(uses):
        slots = rng.choice([1, 2, 4, 8, 16])
        per = nw // slots
        nb = [rng.randint(0, 5) for _ in range(slots)]
        active = rng.randint(1, slots)
        plan = []
        for w in range(nw):
            s = w // per
            b = nb[s] if s < active else 0
            if s < active and rng.random() < 0.2:
                b = rng.randint(0, b)      # an early return (all waves of that block alike in real code; per wave here is harsher)
            plan.append(b)
        plans.append(plan)
    # per wave: a generator of steps; 'B' = arrive at barrier (yield), other ops mutate sh
    def wave(w):
        for u in range(uses):
            for _ in range(plans[u][w]):
                yield "B"; sh["nbar"][w] += 1
            if protocol == "count":            # the first version: increment, then barrier until count == 16
                sh["done"] += 1
                yield "op"
                while True:
                    yield "B"
                    yield "op"
                    if sh["done"] == nw:
                        break
                yield "op"
                # next use's begin: reset + barrier
                if w == 0:
                    sh["done"] = 0
                yield "B"
            else:                              # round numbers
                r = sh["nbar"][w]
                sh["done"] += 1
                if sh["done"] == nw:
                    sh["done"] = 0; sh["exit"] = r + 1
                yield "op"
                while True:
                    yield "B"; sh["nbar"][w] += 1; r += 1
                    yield "op"
                    if sh["exit"] == r:
                        break
        yield "END"
    gens = [wave(w) for w in range(nw)]
    state = ["run"] * nw
    barriers = [0] * nw
    steps = 0
    while True:
        steps += 1
        if steps > 20000:
            return "LIVELOCK", barriers               # waves keep taking barriers without ever leaving
        runnable = [w for w in range(nw) if state[w] == "run"]
        if not runnable:
            waiting = [w for w in range(nw) if state[w] == "bar"]
            ended = [w for w in range(nw) if state[w] == "end"]
            if len(waiting) == nw:
                for w in waiting:
                    state[w] = "run"; barriers[w] += 1
                continue
            if len(ended) == nw:
                return "ok", barriers
            return "DEADLOCK", (len(waiting), len(ended))
        w = rng.choice(runnable)
        try:
            x = next(gens[w])
        except StopIteration:
            x = "END"
        if x == "B":
            state[w] = "bar"
        elif x == "END":
            state[w] = "end"



def test_rounds_protocol_never_hangs_or_desynchronises():
    for seed in range(1500):
        res, info = run("rounds", seed)
        assert res == "ok" and len(set(info)) == 1, (seed, res, info)


def test_the_first_protocol_is_caught():
    bad = sum(1 for seed in range(40) if run("count", seed)[0] != "ok" or len(set(run("count", seed)[1])) != 1)
    assert bad > 0
