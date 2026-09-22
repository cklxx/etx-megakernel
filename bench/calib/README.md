# Phase-0 calibration

Fills the `costs:` block of `etx/machine/arch/<arch>.yaml`. Every number the
scheduler uses comes from here or from a cited public measurement; nothing is
guessed in code. Run on the target machine:

```bash
bash bench/calib/run_calib.sh gfx942      # builds with hipcc, runs, writes build/calib_gfx942.json
```

| bench | measures | feeds |
|---|---|---|
| `atomic_pingpong.hip` | round trip of an atomic counter between two workgroups on the same / a different domain | `t_local_ns`, `t_cross_ns` |
| `flag_latency.hip` | one-way flag latency with the release / acquire sequence from the lowering table | `t_flag_cross_ns`, `t_fence_ns` |
| `phase_switch.hip` | whole-GPU phase switch: counter + fence vs sentinel polling | `t_phase_switch_*_ns` |
| `queue_contention.hip` | N workgroups popping one ring; throughput vs N | `queue_contention` curve, `t_pop_ns` |
| `discover_domain.hip` | which domain every workgroup lands on, via HW_ID | validates `wg_to_domain_map` |
| `p2p_atomic.hip` | two devices, fine-grained memory, system-scope atomic round trip | `t_dev_ns` |

Status: sources are written against the ETX ABI and lowering macros but have
not been compiled on hardware yet. The first GPU session runs them before any
kernel work (design §15, phase 0).
