"""CPU protocol adapter, explicitly distinct from device execution."""
from etx.api.types import PlannedWorkload, SimulationResult
from etx.sim import simulate as simulate_plan
from .planning import recompile


def simulate(planned, *, seed=0):
    if not isinstance(planned, PlannedWorkload):
        raise TypeError('simulate requires PlannedWorkload')
    if type(seed) is not int:
        raise ValueError('simulation seed requires an integer')
    compiled = recompile(planned)
    result = simulate_plan(compiled, seed=seed)
    return SimulationResult(planned.identity, seed,
                            dict(makespan_us=result.makespan_us, busy_us=result.busy_us,
                                 wait_us=result.wait_us, sched_us=result.sched_us,
                                 n_workers=result.n_workers, per_grid_us=result.per_grid_us,
                                 timing_scope='simulated_protocol'),
                            result.deadlock, tuple(result.blocked))
