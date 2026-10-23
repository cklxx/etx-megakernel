"""Executable protocol specification, not a GPU visibility/correctness certificate.

All head state is external to imported matrix temporaries. Epoch reset is a
quiescent operation, including rollover; partial buffers retain partition indices.
The model explicitly represents each producing wave's store retirement.
"""
from dataclasses import dataclass, field
from itertools import permutations


class ProtocolError(ValueError):
    pass


def require(condition, invariant):
    if not condition:
        raise ProtocolError(invariant)


@dataclass
class Head:
    epoch: int = 1
    partitions: int = 5
    waves: int = 16
    claimed: bool = False
    prep_drained: set = field(default_factory=set)
    ready: bool = False
    started: set = field(default_factory=set)
    drained: dict = field(default_factory=dict)
    partials: dict = field(default_factory=dict)
    arrived: set = field(default_factory=set)
    acquired: set = field(default_factory=set)
    reduced: bool = False
    output_drained: set = field(default_factory=set)
    published: bool = False
    prep_count: int = 0
    last: int = -1
    retired: set = field(default_factory=set)

    def claim(self, partition):
        require(partition == 0 and not self.claimed, "exact_once_owner")
        self.claimed = True
        self.prep_count += 1

    def drain_prep(self, wave):
        require(self.claimed and 0 <= wave < self.waves, "prep_writer")
        self.prep_drained.add(wave)

    def publish_prep(self):
        require(len(self.prep_drained) == self.waves, "all_prep_waves")
        self.ready = True

    def start(self, partition, epoch):
        require(epoch == self.epoch, "epoch_acquire")
        require(self.ready, "early_publication")
        require(0 <= partition < self.partitions and partition not in self.started, "partition_once")
        self.started.add(partition)
        self.drained[partition] = set()
        # Values deliberately make arrival-order floating-point sums unsafe.
        self.partials[partition] = [1e20, -1e20, 3.0, 4.0, 5.0][partition]

    def drain_partial(self, partition, wave):
        require(partition in self.started and 0 <= wave < self.waves, "partial_writer")
        self.drained[partition].add(wave)

    def arrive(self, partition):
        require(partition not in self.arrived, "arrival_once")
        require(len(self.drained.get(partition, ())) == self.waves, "all_partial_waves")
        self.arrived.add(partition)
        if len(self.arrived) == self.partitions:
            self.last = partition
        return partition == self.last

    def acquire_partials(self):
        require(len(self.arrived) == self.partitions, "completion_fanin")
        self.acquired = set(self.arrived)

    def reduce(self, order):
        require(self.acquired == set(range(self.partitions)), "all_partial_acquires")
        require(tuple(order) == tuple(range(self.partitions)), "partition_order")
        require(not self.reduced, "reduce_once")
        self.reduced = True
        value = 0.0
        for p in order:
            value += self.partials[p]
        return value

    def publish_output(self):
        require(self.reduced and len(self.output_drained) == self.waves, "all_reduce_waves")
        self.published = True

    def retire(self, partition):
        require(partition in self.arrived and partition not in self.retired and
                (partition != self.last or self.published), "worker_retirement")
        self.retired.add(partition)

    def reset(self, epoch):
        require(self.published and len(self.retired) == self.partitions and epoch != self.epoch and 0 <= epoch <= 0xffffffff, "quiescent_reset")
        self.__dict__.update(Head(epoch, self.partitions, self.waves).__dict__)


def slot_participation(waves, participants):
    require(set(participants) == set(range(waves)), "idle_slot_participation")


def queue_liveness(queues, dependencies):
    """Topological execution of per-worker FIFO edges plus payload dependencies.

    A cycle includes a waiting consumer monopolising its queued producer. This is
    a scheduling proof only; callers still need actual resident producers on GPU.
    """
    nodes = [n for q in queues for n in q]
    require(len(nodes) == len(set(nodes)), "queue_exact_once")
    edges = set(dependencies) | {(a, b) for q in queues for a, b in zip(q, q[1:])}
    require(all(a in nodes and b in nodes for a, b in edges), "queue_endpoints")
    remaining = set(nodes)
    while remaining:
        runnable = {n for n in remaining if not any(b == n and a in remaining for a, b in edges)}
        require(bool(runnable), "queue_progress")
        remaining -= runnable


def reset_queues(cursors, heads):
    require(all(h.published and len(h.retired) == h.partitions for h in heads), "queue_epoch_reset")
    return [0 for _ in cursors]


def dry_run():
    count = 0
    for order in permutations(range(5)):
        h = Head(epoch=0xffffffff)
        h.claim(0)
        for w in range(16):
            h.drain_prep(w)
        h.publish_prep()  # observable before owner attention has even started
        for p in order:
            h.start(p, h.epoch)
            for w in reversed(range(16)):
                h.drain_partial(p, w)
            assert h.arrive(p) == (p == order[-1])
        h.acquire_partials()
        assert h.reduce(range(5)) == 12.0
        h.output_drained.update(range(16))
        h.publish_output()
        for p in range(5):
            h.retire(p)
        assert reset_queues([5], [h]) == [0]
        h.reset(0)
        assert h.prep_count == 0 and not h.partials and not h.ready
        count += 1
    slot_participation(16, range(16))
    queue_liveness([["prep", "p0"], ["p1"], ["p2"], ["p3"], ["p4", "reduce"]],
                   [("prep", f"p{p}") for p in range(5)] + [(f"p{p}", "reduce") for p in range(5)])
    return {"orders": count, "epochs": [0xffffffff, 0], "waves": 16, "gpu_certificate": False}


def main():
    import json
    print(json.dumps(dry_run(), sort_keys=True))


if __name__ == "__main__":
    main()
