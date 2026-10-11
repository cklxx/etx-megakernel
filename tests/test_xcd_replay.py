"""Replay must retain costs even when worker reuse changes."""
import pytest

from etx.sim.xcd import Cost, Hw, ReplayTiming, simulate


def fixture(consumer_worker):
    return dict(types={"a": {"id": 0}}, workers_per_domain=2,
                descs=[dict(id=i, type=0, coord=[i], domain=0) for i in range(2)],
                events=[dict(name="E", scope="DOMAIN")], ev_inst=[["E", [0], [0]]],
                task_in=[[], [0]], static_queues={"0:0": [0, 1] if consumer_worker == 0 else [0],
                                                "0:1": [] if consumer_worker == 0 else [1]})


@pytest.mark.parametrize("worker,expected", [(0, 10), (1, 6)])
def test_arrive_fetch_kept_with_producer_but_event_can_wake_free_worker(worker, expected):
    rows, stats = simulate(fixture(worker), lambda *_: Cost(chain_us=2),
                           Hw(handoff_us={"DOMAIN": 0}),
                           replay=ReplayTiming([4, 0], [0, 2]))
    assert stats["unfinished"] == 0
    assert stats["makespan_us"] == expected
    # The body-done stamp is before arrive/fetch, not inflated by it.
    assert (rows[0][7] - rows[0][6]) * .01 == 2
    assert (rows[1][7] - rows[1][6]) * .01 == 2


def test_replay_initial_take_and_ready_delay_do_not_vanish():
    rows, stats = simulate(fixture(0), lambda *_: Cost(chain_us=2),
                           Hw(handoff_us={"DOMAIN": 0}),
                           replay=ReplayTiming([4, 0], [1, 2], {0: 3}))
    assert stats["makespan_us"] == 14
    assert rows[0][5:8] == (301, 401, 601)
