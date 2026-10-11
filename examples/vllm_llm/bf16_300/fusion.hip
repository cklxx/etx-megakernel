// Standalone preparation, not wired into model.py or eligible for model timing.
// Launch only after host queue/discovery/residency checks and quiescent epoch reset.
struct Bf16HeadState {
  unsigned epoch, xcd, head, claimed, arrived, mask;
  unsigned long long ready, output;
};
struct Bf16FusionArgs {
  etx_ctx prep, attention, reduce;  // loaded separately at each call: bounded live ranges
  Bf16HeadState* state;
  unsigned epoch, partition, partitions, spin_limit;
};
struct Bf16DomainQueue { const Bf16FusionArgs* entries; unsigned count, next; };
struct Bf16SnapshotSpan { const unsigned char* src; unsigned char* dst; unsigned long long bytes; };
// Three stage lists; attention's list refers to THIS partition's original partials.
struct Bf16SnapshotList { const Bf16SnapshotSpan* spans; unsigned count; };
__shared__ unsigned bf16_broadcast[4];  // +16 B, separate from the reused import arena

static __device__ __forceinline__ void bf16_snapshot(const Bf16SnapshotList& list) {
  for (unsigned s = 0; s < list.count; ++s) {
    const Bf16SnapshotSpan span = list.spans[s];
    for (unsigned long long i = threadIdx.x; i < span.bytes; i += blockDim.x) span.dst[i] = span.src[i];
  }
  ETX_RELEASE_DOMAIN();
  __syncthreads();
}
static __device__ __forceinline__ void bf16_fail(int* abort) {
  if (threadIdx.x == 0) __hip_atomic_store(abort, 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
extern "C" __global__ __launch_bounds__(1024, 1)
void bf16_300_fusion(Bf16DomainQueue* queues, int* ctrl_abort
#ifdef BF16_300_CHECK
                     , const Bf16SnapshotList* snapshots
#endif
                     ) {
  etx_import_init();
  __syncthreads();
  // Actual XCD identity selects a queue; the first ticket is the prep owner.
  // Host validates five resident slots per nonempty XCD before launch. Queue
  // cursors and head state reset only after ALL blocks have retired.
  const unsigned domain = ETX_DOMAIN_ID();
  if (domain >= ETX_NUM_DOMAINS) { bf16_fail(ctrl_abort); return; }
  Bf16DomainQueue* q = queues + domain;
  if (!q->count) return;
  if (q->count != 5) { bf16_fail(ctrl_abort); return; }
  if (threadIdx.x == 0)
    bf16_broadcast[1] = __hip_atomic_fetch_add(&q->next, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  const unsigned ticket = bf16_broadcast[1];
  if (ticket >= q->count) return; // surplus blocks do not poll or claim producers' resources
  const Bf16FusionArgs* a = q->entries + ticket;
  if (a->partition != ticket) { bf16_fail(ctrl_abort); return; }
  Bf16HeadState* s = a->state;
  if (ETX_DOMAIN_ID() != s->xcd || a->epoch != s->epoch || a->partitions != 5 || a->partition >= 5 ||
      a->prep.coord[0] != (int)s->head || a->attention.coord[0] != (int)(5 * s->head + a->partition) ||
      a->reduce.coord[0] != (int)s->head) {
    bf16_fail(ctrl_abort); return;
  }
  if (a->partition == 0) {
    if (threadIdx.x == 0)
      bf16_broadcast[0] = __hip_atomic_exchange(&s->claimed, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (bf16_broadcast[0]) { bf16_fail(ctrl_abort); return; }
    { const etx_ctx c = a->prep; etx_tile_dsl_attnprep(&c); }
    ETX_RELEASE_DOMAIN();  // EACH producing wave drains before leader publishes
    __syncthreads();
#ifdef BF16_300_CHECK
    bf16_snapshot(snapshots[3 * (5 * s->head + a->partition)]); // before attention or any in-place successor
#endif
    if (threadIdx.x == 0) __hip_atomic_store(&s->ready, (unsigned long long)a->epoch + 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    if (threadIdx.x == 0) {
      unsigned n = 0;
      bf16_broadcast[0] = 0;
      while (__hip_atomic_load(&s->ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != (unsigned long long)a->epoch + 1ull) {
        if (++n >= a->spin_limit || __hip_atomic_load(ctrl_abort, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
          bf16_broadcast[0] = 1; bf16_fail(ctrl_abort); break;
        }
        ETX_BACKOFF();
      }
    }
    __syncthreads();
    if (bf16_broadcast[0]) return;
  }
  ETX_ACQUIRE_DOMAIN();  // every consumer wave invalidates re-used cached lines
  { const etx_ctx c = a->attention; etx_tile_attn_mfma16_g8(&c); }
  // The adapter has uniformly finished import rounds, including inactive waves.
  ETX_RELEASE_DOMAIN();
  __syncthreads();
#ifdef BF16_300_CHECK
  bf16_snapshot(snapshots[3 * (5 * s->head + a->partition) + 1]); // partials before the original reduce
#endif
  if (threadIdx.x == 0) {
    const unsigned bit = 1u << a->partition;
    const unsigned old = __hip_atomic_fetch_or(&s->mask, bit, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    bf16_broadcast[0] = (old & bit) ? 2u :
      (__hip_atomic_fetch_add(&s->arrived, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == 4u ? 1u : 0u);
  }
  __syncthreads();
  if (bf16_broadcast[0] == 2) { bf16_fail(ctrl_abort); return; }
  if (bf16_broadcast[0] == 1) {
    // acq_rel RMW chain covers ALL five releases; all waves acquire payload.
    ETX_ACQUIRE_DOMAIN();
    { const etx_ctx c = a->reduce; etx_tile_attn_reduce(&c); } // original partition index/order untouched
    ETX_RELEASE_DOMAIN();
    __syncthreads();
#ifdef BF16_300_CHECK
    bf16_snapshot(snapshots[3 * (5 * s->head + a->partition) + 2]);
#endif
    if (threadIdx.x == 0) __hip_atomic_store(&s->output, (unsigned long long)a->epoch + 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}
