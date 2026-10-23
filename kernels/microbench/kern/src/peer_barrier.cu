// Cross-rank barrier over exported flag words, the primitive every EP
// superstep is built from. Rank r owns `flags[nranks]` (exported); every
// rank's copy is reachable through `peers[]`. One barrier: publish the new
// epoch into slot r of every member's flags with a release store, then
// spin with acquire loads on the local flags until every slot reached the
// epoch, or `timeout_ns` of globaltimer pass — then `err[0] = 1 + first
// missing rank` and the barrier gives up (nothing hangs on a dead peer).
// `epoch[0]` is a carry the kernel advances itself, so a captured graph
// replays without host help. Plain st/ld only: no TMA, no bulk copy.
//
//   kern_peer_barrier(inout u32 flags[nranks], in u64 peers[nranks],
//                     inout u32 epoch[1], out i32 err[1],
//                     i32 rank, i32 nranks, i64 timeout_ns)
//   block [32,1,1], grid [1,1,1]; nranks <= 32.

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

extern "C" __global__ void kern_peer_barrier(unsigned* flags, const unsigned long long* peers,
                                             unsigned* epoch, int* err, int rank, int nranks,
                                             long long timeout_ns) {
    const int t = threadIdx.x;
    __shared__ unsigned e;
    __shared__ int fail;
    if (t == 0) {
        e = epoch[0] + 1;
        fail = 0;
    }
    __syncthreads();
    if (t < nranks) {
        unsigned* remote = reinterpret_cast<unsigned*>(peers[t]);
        st_release_sys(remote + rank, e);
        const unsigned long long t0 = gtimer();
        while (ld_acquire_sys(flags + t) < e) {
            if ((long long)(gtimer() - t0) > timeout_ns) {
                atomicMax(&fail, 1 + t);
                break;
            }
        }
    }
    __syncthreads();
    if (t == 0) {
        epoch[0] = e;
        err[0] = fail;
    }
}
