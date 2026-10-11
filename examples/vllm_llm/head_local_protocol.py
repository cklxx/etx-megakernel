"""C6 head-local protocol contract and opt-in boundary (no GPU implementation).

Emit the future probe contract:
  python -m examples.vllm_llm.head_local_protocol --spec out.json
ETX_VL_HEAD_LOCAL_PROTOCOL=1 reaches a fail-closed model hook: until an
import-preserving adapter exists it cannot silently run ordinary scheduling.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

NET_US_PER_LAYER = 50 / 48


def specification():
    return {
        'schema': 'etx.head-local-probe.v1', 'implemented': False, 'enabled_by_default': False,
        'shape': {'heads': 4, 'partitions_per_head': 5, 'q_heads_per_kv_head': 8,
                  'head_size': 128, 'kv_block_size': 16, 'partition_tokens': 256,
                  'block_threads': 1024},
        'state': ['epoch:u64', 'prep_owner:u32', 'prep_ready_epoch:u64',
                  'completion_count:u32', 'partition_seen_mask:u32', 'output_ready_epoch:u64'],
        'placement': 'all workers for a head on its discovered XCD; producers precede waiters',
        'thresholds': {'net_us_per_layer': NET_US_PER_LAYER, 'net_us_per_step': 50,
                       'max_vgpr': 128, 'max_lds_bytes': 29752},
        'protocol': [
            'One fixed prep owner (partition 0); epoch-scoped exact-once claim.',
            'Run original prep; all writing waves drain, join, release prep-ready(epoch).',
            'Every partition waits for matching epoch, acquires KV/Q visibility, joins all reading waves.',
            'Run original imported attention; finish all slot rounds before arena reuse.',
            'Each partition drains ALL producing waves, joins, release/acq_rel arrival exactly once.',
            'Last arrival elects reducer; acquire ALL partition payloads before original imported reduce.',
            'Preserve original partition order and global partials; sequentially reuse LDS arena.',
            'All reducer waves drain and join before publishing normal output event for o_proj.',
            'Reset only after output publication and participant quiescence; reject stale epochs.',
            'Before u64 wrap, quiesce entire launch and clear epoch-tagged state; no in-flight reuse.',
        ],
        'future_gpu_gate': ['reused-cache-line payload stress, duplicates, epochs and rollover',
                            'host --check every imported constituent under new scheduling',
                            'CHECK-DSL, 32/32 greedy tokens, actual whole-kernel residency/spills',
                            'unchanged-body pinned interleaved wall pairs, net >=50 us/step'],
        'budget_basis': {'gross_boundary_us_per_layer_estimate': 2,
                         'exposed_attn_reduce_hops_us_per_layer_estimate': 260 / 48,
                         'hop_credit_in_boundary_budget': 0},
    }


def check_opt_in():
    """Future adapter integration point. Default has no graph/source effect."""
    requested = os.environ.get('ETX_VL_HEAD_LOCAL_PROTOCOL', '0')
    if requested == '0':
        return
    if requested != '1':
        raise SystemExit('ETX_VL_HEAD_LOCAL_PROTOCOL must be 0 or 1')
    raise SystemExit('ETX_VL_HEAD_LOCAL_PROTOCOL: specification hook only; no validated GPU adapter. '
                     'Emit the probe contract with python -m examples.vllm_llm.head_local_protocol --spec PATH')


@dataclass
class EpochModel:
    """CPU state-transition oracle, not a simulation of cache/fence behavior."""
    partitions: int = 5
    epoch: int = 1
    owner: int | None = None
    prep_ready: bool = False
    acquired: set[int] = field(default_factory=set)
    released: set[int] = field(default_factory=set)
    reducer: int | None = None
    all_acquired: bool = False
    output: bool = False

    def _check(self, epoch, partition=None):
        if epoch != self.epoch:
            raise ValueError('stale epoch')
        if partition is not None and not 0 <= partition < self.partitions:
            raise ValueError('partition out of range')

    def claim_prep(self, epoch, partition):
        self._check(epoch, partition)
        if partition != 0 or self.owner is not None:
            raise ValueError('prep must be claimed exactly once by partition 0')
        self.owner = partition

    def publish_prep(self, epoch, partition, *, all_waves_drained):
        self._check(epoch, partition)
        if self.owner != partition or self.prep_ready or not all_waves_drained:
            raise ValueError('prep publication needs owner and all-wave drain')
        self.prep_ready = True

    def acquire_prep(self, epoch, partition):
        self._check(epoch, partition)
        if not self.prep_ready:
            raise ValueError('prep not published')
        self.acquired.add(partition)

    def arrive(self, epoch, partition, *, all_waves_drained):
        self._check(epoch, partition)
        if partition not in self.acquired or partition in self.released or not all_waves_drained:
            raise ValueError('arrival needs acquired prep, exact-once partition and all-wave drain')
        self.released.add(partition)
        if len(self.released) == self.partitions:
            self.reducer = partition
        return self.reducer == partition

    def acquire_partials(self, epoch, partition):
        self._check(epoch, partition)
        if self.reducer != partition or len(self.released) != self.partitions:
            raise ValueError('only elected last arriver acquires all partials')
        self.all_acquired = True

    def publish_output(self, epoch, partition, *, all_waves_drained):
        self._check(epoch, partition)
        if self.reducer != partition or not self.all_acquired or not all_waves_drained or self.output:
            raise ValueError('output publication needs acquired partials, reduce completion and all-wave drain')
        self.output = True

    def reset(self, *, quiescent):
        if not self.output or not quiescent or self.epoch == (1 << 64) - 1:
            raise ValueError('reset needs complete quiescence; rollover needs full-launch reinitialization')
        fresh = EpochModel(self.partitions, self.epoch + 1)
        self.__dict__.update(fresh.__dict__)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path, required=True)
    args = ap.parse_args()
    args.spec.write_text(json.dumps(specification(), indent=2) + '\n')


if __name__ == '__main__':
    main()
