"""CPU state model for row ownership, completion and quiescent reset.

This is not a GPU memory model. A drained-wave premise is explicit. The tests
remove that premise to expose counter-before-payload errors.
"""
from dataclasses import dataclass, field
import argparse
import json
import random


@dataclass
class Protocol:
    rows: list[int] = field(default_factory=lambda:[8]*32)
    stats: int = 32
    full: int = 256
    producers: set = field(default_factory=set)
    finished: set = field(default_factory=set)
    payload: set = field(default_factory=set)
    merged: set = field(default_factory=set)
    owners: dict = field(default_factory=dict)
    scalar: bool = False
    live: bool = False
    readers: set = field(default_factory=set)
    epoch: int = 0

    def produce(self, slot, group, drained=16):
        if slot not in range(8) or group not in range(32): raise ValueError("coordinate")
        key=(slot,group)
        if key in self.producers: raise ValueError("duplicate producer")
        self.live=True
        if drained != 16: raise ValueError("all writing waves must drain before row arrival")
        self.producers.add(key); self.payload.add(key)
        self.rows[group]-=1
        if self.rows[group]==0:
            if any((s,group) not in self.payload for s in range(8)):
                raise ValueError("row owner acquired stale payload")
            self.owners[group]=slot
            return True
        return False

    def publish(self, group, drained=16):
        if group not in self.owners or group in self.merged: raise ValueError("invalid row owner")
        if drained != 16: raise ValueError("statistics must drain before arrival")
        self.merged.add(group); self.stats-=1
        if self.stats==0: self.scalar=True

    def complete(self, slot, group):
        key=(slot,group)
        if key not in self.producers: raise ValueError("completion without producer")
        if key in self.finished: raise ValueError("duplicate completion")
        if self.owners.get(group)==slot and group not in self.merged:
            raise ValueError("original completion must follow appended row work")
        if self.stats==0 and not self.scalar: raise ValueError("scalar publication precedes completion")
        self.finished.add(key); self.full-=1
        if self.full<0: raise ValueError("duplicate completion")
        if self.full==0:
            if len(self.merged)!=32 or not self.scalar: raise ValueError("consumer released before scalar")

    def consume(self, task):
        if self.full or not self.scalar: raise ValueError("consumer before full completion")
        if task not in range(256) or task in self.readers: raise ValueError("duplicate/invalid consumer")
        self.readers.add(task)
        if len(self.readers)==256: self.live=False

    def reset(self):
        if self.live or self.full not in (0,256): raise ValueError("reset before all readers retire")
        epoch=self.epoch+1
        self.__dict__.update(Protocol().__dict__); self.epoch=epoch


def exercise(seed=20261015, tokens=128):
    rng=random.Random(seed); p=Protocol(); owner_slots=set(); orders=[]
    for token in range(tokens):
        tasks=[(s,j) for s in range(8) for j in range(32)]; rng.shuffle(tasks)
        # Randomly defer owner tails. Non-owners never wait for an owner.
        pending=[]
        for s,j in tasks:
            if p.produce(s,j): pending.append((s,j))
            else: p.complete(s,j)
            if pending and rng.random()<.25:
                os,og=pending.pop(rng.randrange(len(pending))); p.publish(og); p.complete(os,og)
        rng.shuffle(pending)
        for s,j in pending: p.publish(j); p.complete(s,j)
        assert p.full==p.stats==0 and p.scalar and len(p.owners)==32
        consumers=list(range(256)); rng.shuffle(consumers)
        for consumer in consumers: p.consume(consumer)
        owner_slots.update(p.owners.values()); orders.append(list(p.owners.values()))
        p.reset()
    assert owner_slots==set(range(8))
    return dict(kind="CPU protocol only; drained-wave premise, not GPU visibility",seed=seed,
                tokens=tokens,row_owners=tokens*32,producers=tokens*256,
                owner_slots=sorted(owner_slots),quiescent_resets=tokens,
                extra_tail="Final group has zero slack. Full completion waits for its merge and scalar.",
                elapsed_prediction=None)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--tokens",type=int,default=128)
    a=p.parse_args(); print(json.dumps(exercise(tokens=a.tokens),indent=2))
