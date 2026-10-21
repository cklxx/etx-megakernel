# fused_moe_kernel at batch 1: an operation- and instruction-level model (Qwen3-30B-A3B, MI300X)

This covers vLLM's Triton `fused_moe_kernel` (bf16, no quantization) at decode batch 1, in two forms: as vLLM ran it, and as
ETX imports it. It is written as input to the ETX performance model. Every claim cites a source line.

## 0. Sources and shorthand

| Tag | File |
|---|---|
| **[S]** | `~/code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py` (the kernel is at 299; `invoke_fused_moe_triton_kernel` is at 763) |
| **[G]** | gate_up assembly vLLM ran: `~/code/etx/results_2026-09-27/vlnext/moe/triton/DPMBXHZFE5XP5YVBH5OONA6SL322JU7BTL4E2ZYIRHTUFPE6OPMQ/fused_moe_kernel.amdgcn` |
| **[D]** | down assembly vLLM ran: `.../NVIB2QML2AAGSMNQMEOQWWK62D5THRUJXU3N75EOTTTWYQVIDYOQ/fused_moe_kernel.amdgcn` |
| **[E]** | ETX imported form: `~/code/etx/build/rocm22/qwen3-30b-a3b_base/imp_wrap.s`, `wrap_fused_moe_up` 45033-46133, `wrap_fused_moe_down` 46222-47330 |
| **[M]** | the same body inlined in ETX's unfused megakernel: `.../qwen3-30b-a3b_base/mk.s`, `etx_unfused_d0` (moe_up body at about 148560-149560) |
| **[H]** | `~/code/etx/etx/runtime/include/etx/import.h` (the slot barrier) |
| **[W]** | `.../qwen3-30b-a3b_base/imp_wrap.hip` (wrappers at 89-101) |
| **[T]** | vLLM kernel trace `results_2026-09-27/vlnext/moe/prof/8f2e8af04e27/7_kernel_trace.csv` (3120 launches of each kernel) |
| **[hw]** | `results_2026-09-27d/hw_wg.log` (`__syncthreads` 77 ns = 161 clk; clock under load 1.977 GHz) |
| **[ISA]** | CDNA3 ISA guide (`scratchpad/cdna3_isa.txt`). Table 28: `V_MFMA_F32_16X16X16_BF16` = 16 cycles. §S_SLEEP: `s_sleep 1` sleeps 0-64 clk. |

Clock: 1.98 GHz (**[hw]**), so 1 us is about 1980 clk. Unloaded latencies are L2 100-108 ns and HBM 333-350 ns
(docs/deepseek/MODEL-dsv2.md:26). Achievable HBM bandwidth is about 4.3 TB/s (CLAUDE.md:13).

## 1. Launch shape at batch 1

| | gate_up (w13) | down (w2) |
|---|---|---|
| B (weights) per expert | 1536 x 2048 bf16, K contiguous (`stride_bk` = 1, `stride_bn` = 2048; **[S]** 877-879) | 2048 x 768 bf16, `stride_bn` = 768 |
| N, K, top_k, MUL_ROUTED_WEIGHT | 1536, 2048, 8, False | 2048, 768, 1, True |
| EM = num_tokens x BLOCK_M (**[S]** 835) | 8 x 16 = 128, so num_pid_m = 8 | 8 x 16 = 128 |
| grid = 8 x N/64 (**[S]** 836-839) | 192 programs | 256 programs |
| pid mapping (GROUP_SIZE_M 1, **[S]** 388-396) | pid_m = pid / 24 (expert slot), pid_n = pid mod 24 | pid_m = pid / 32, pid_n = pid mod 32 |
| K tiles (BLOCK_K 128) | 16 | 6 |
| Programs per XCD (round-robin), CUs used | 24 of 38 (112 CUs idle) | 32 of 38 (48 idle) |
| Per program | 256 threads = 4 wave64, 78 VGPRs, 20480 B LDS (**[G]** 953, 990; fused_moe_up/.json) | same (**[D]** 963) |

All programs run at once, one per CU, so the launch time is the time of the slowest program plus launch and drain.

## 2. Work per program: one real row out of 16

**Real rows.** With naive assignment, `offs_token = where(offs == 0, pid_m, num_valid_tokens)` (**[S]** 412-419), so `token_mask`
(**[S]** 421) is true only for row 0 of the 16-row M tile. In **[G]** the row index is `v2 = tid >> 4` (148-150). Row 0 gets pid_m and
the rest get `num_valid_tokens` (155-161), and `token_mask` is computed at 166. For gate_up the A row is `pid_m / top_k = pid_m >> 3`
(278-281 divide by 8), which is always token 0. For down it is `pid_m` itself (**[D]** 282, no divide).

**How masking is implemented.** A masked lane's buffer offset is replaced by `0x80000000` (`v_bfrev_b32 v1, 1`, **[G]** 294, used at
300-302). The buffer resource has `num_records = 0x7ffffffe` (**[G]** 234), so masked lanes are out of range. They return 0 and cause
no memory traffic.

| Per program | gate_up | down |
|---|---|---|
| B read (64 n rows x K x 2 B) | 64 x 2048 x 2 = **262,144 B** (16 KiB per K tile) | 64 x 768 x 2 = **98,304 B** (16 KiB per K tile) |
| A read (1 real row) | 2048 x 2 = 4,096 B (256 B per tile: 16 lanes x 16 B) | 768 x 2 = 1,536 B |
| C written (1 row x 64 cols bf16) | 128 B (16 lanes x 8 B, **[G]** 803) | 128 B (**[D]** 813) |
| Scalars | num_tokens_post_padded 4 B (**[G]** 123), expert_ids[pid_m] 4 B (146) | same, plus topk_weights 4 B (**[D]** 758) |
| **Per launch** | B **50,331,648 B (48 MiB)**; A 786 KB requested but 4 KiB unique (every program reads token 0); C 24,576 B | B **25,165,824 B (24 MiB)**; A 12 KiB unique; C 32,768 B |
| **Per K step, whole device** | 192 x 16 KiB = **3.0 MiB** | 256 x 16 KiB = **4.0 MiB** |

**What the 15 masked rows cost:**

- **DRAM: nothing.** Masked A lanes are out of range, and masked stores are dropped (**[G]** 795-803).
- **MFMA: 15/16 of the work is wasted.** gate_up runs 16 tiles x 8 MFMA x 4 waves = 512 `v_mfma_f32_16x16x16_bf16` per program, and one row
  of useful work would need only 1/16 of them. At 16 clk each (**[ISA]** Table 28) that is 128 clk per wave per tile, which is small.
- **LDS: about 36% of the traffic is zeros.** Each K step writes a 4 KiB A tile, of which 3,840 B are zeros. Every wave then reads the
  whole 4 KiB A tile (4 of its 8 `ds_read2_b64`, see §3), so 16 KiB of the 32 KiB read per step is A, and 15/16 of that is zeros.
  At 128 B/clk per CU the zero A reads cost about 120 clk per step. They sit on the critical path (§3.5), so they cost roughly
  0.06 us x 16 = about 1 us per gate_up launch.
- **Instructions:** waves 1-3 issue an A `buffer_load_dwordx4` with every lane masked, one per tile. This is negligible.

## 3. The software pipeline as compiled ([G]; [D] is instruction-for-instruction the same shape)

### 3.1 LDS layout: one LDS stage plus a register stage (not an LDS double buffer)

The shared size is 20,480 B, which is exactly one A tile (16 x 128 x 2 = 4,096 B) plus one B tile (128 x 64 x 2 = 16,384 B).

- B occupies [0, 16384). `ds_write2st64_b64 ... offset1:8` / `offset0:16 offset1:24` place each lane's four n-rows (n, n+16, n+32, n+48)
  in 4 KiB quarters.
- A occupies [16384, 20480) (`ds_write_b64 ... offset:16384`, **[G]** 339-340).
- The addresses are XOR-swizzled to avoid bank conflicts (**[G]** 321-327).

`num_stages = 2` does not give two LDS buffers. It gives a **register prefetch**: the global loads for tile j+1 land in VGPRs while
tile j is read from the single LDS buffer. This is why the kernel needs **two `s_barrier`s per K tile**:

- **RAW barrier** (after `ds_write` of tile j+1, before its `ds_read`). No wave may read tile j+1 until all four waves have written
  their parts. It is always preceded by `s_waitcnt lgkmcnt(0)` (**[G]** 370-371, 504-505, 603-604, 686-687).
- **WAR barrier** (after the last `ds_read` of tile j, before overwriting the same buffer with tile j+1). No wave may overwrite LDS
  while another wave is still reading tile j (**[G]** 457, 576, 652). No wait precedes it, because the reads were already drained by
  `lgkmcnt(0)` at 424/540/637.

A true LDS double buffer (2 x 20 KiB, which fits in 64 KiB) would need only one barrier per tile.

### 3.2 Prologue ([G] 14-347)

1. **Kernarg preload.** 14 SGPRs are preloaded (**[G]** 943). Lines 14-18 are only the compatibility path, and the real entry is
   `.LBB0_0` (23).
2. **pid_m / pid_n.** Two integer divisions for the grouped ordering (30-121; **[S]** 388-396).
3. **Dependent round trip 1.** `s_load_dword` of num_tokens_post_padded (123), then `s_waitcnt lgkmcnt(0)` (131), then the early exit
   `pid_m*16 >= ntpp` (130-136; **[S]** 405-407).
4. **Dependent round trip 2.** `s_load_dword expert_ids[pid_m]` (146), then wait (163), then the `== -1` check leading to the
   write-zeros path `.LBB0_8`/`.LBB0_13` (164-168, 805-825; **[S]** 423-439).
5. **Address math.** `offs_bn % N` for the 4 n-rows per lane, about 100 VALU (171-276; **[S]** 442), plus A/B address mads (287-307).
6. **Tile 0 loads.** 1 A plus 4 B `buffer_load_dwordx4` (303, 312, 313, 318, 319).
7. **Tile 0 LDS write.** `vmcnt(4)` then 2 `ds_write_b64` of A (338-340), `vmcnt(2)` (342) and `vmcnt(0)` (344) then
   4 `ds_write2st64_b64` of B (343-347). The whole tile-0 load latency is exposed here.
8. **K = 128 case.** If K+127 < 256 (one tile only), jump to `.LBB0_9` (333, 349).

**Peeled first iteration (352-470).** The compiler rotates the loop by half an iteration:

- A(tile 1) is loaded (369).
- **RAW barrier** (370-371).
- 8 `ds_read2_b64` of tile 0 (373-413) feed 8 MFMA (396-459).
- B(tile 1) is loaded after the 5th MFMA (438, 439, 446, 447).
- **WAR barrier** (457).
- `vmcnt(4)/(2)/(0)`, then 6 `ds_write` of tile 1 (461-470).
- If there are 2 tiles, go to `.LBB0_10` (455, 472). If there are 3, go to the odd path `.LBB0_15` (474-475).

### 3.3 Main loop `.LBB0_6` ([G] 489-669): two K tiles per trip, 4 barriers

The loop counter `s19` goes 0, 2, 4, ... and the loop exits once `s19 == s18 = (ntiles & ~1) - 4` (477-480, 588-590, 669). So it runs
**(ntiles - 2) / 2** trips:

- K = 2048: 16 tiles, **7 trips**.
- K = 768: 6 tiles, **2 trips** (**[D]** 472-475, 583-585, 664).

Both are even, so `s_bitcmp1 s16, 7` (671) sends control straight to the tail (`.LBB0_16` to `.LBB0_10`), and the odd path at 838-927
is not taken.

On entry, LDS holds tile j and VGPRs v[70:73] hold A(j+1). One trip, per wave, in program order:

| # | [G] lines | Operation | Count / width | Waits |
|---|---|---|---|---|
| 1 | 491-502 | A-mask and address for tile j+1 (SALU/VALU) | ~8 instr | |
| 2 | 503 | `buffer_load_dwordx4` **A(j+1)**, 16 B/lane (only 16 lanes of wave 0 real) | 1 | |
| 3 | 504-505 | **RAW `s_barrier`** (tile j visible) | 1 | `lgkmcnt(0)` |
| 4 | 507-524 | 8 x `ds_read2_b64` (2 x 8 B/lane = 16 B/lane each): 4 from B at [0,16K), 4 from A at [16K,20K) | 128 B/lane, 8 KiB/wave, 32 KiB/program | `lgkmcnt(2)` 515, `(4)` 526, `(2)` 533, `(0)` 540 |
| 5 | 516-548 | 5 x `v_mfma_f32_16x16x16_bf16` into v[0:3] (serial SrcC chain) | 5 x 16 clk | |
| 6 | 550-552, 560 | 4 x `buffer_load_dwordx4` **B(j+1)** (n, n+16, n+32, n+48; 256 B contiguous per n per 16 lanes) | 64 B/lane, 4 KiB/wave, 16 KiB/program | |
| 7 | 558-574 | remaining 3 MFMA | 3 x 16 clk | |
| 8 | 576 | **WAR `s_barrier`** | 1 | none |
| 9 | 577 | `buffer_load_dwordx4` **A(j+2)** | 1 | |
| 10 | 592-601 | `vmcnt(5)` releases A(j+1), then 2 x `ds_write_b64` A (8 B each); `vmcnt(3)`/`vmcnt(1)` release B(j+1), then 4 x `ds_write2st64_b64` B (16 B each) | 80 B/lane, 20 KiB/program | `vmcnt(5)`, `(3)`, `(1)` |
| 11 | 603-604 | **RAW `s_barrier`** (tile j+1) | 1 | `lgkmcnt(0)` |
| 12 | 606-623 | 8 x `ds_read2_b64` of tile j+1 | as row 4 | `lgkmcnt(2/4/2/0)` 614, 625, 631, 637 |
| 13 | 615-667 | 8 MFMA (B(j+2) issued after the 6th: 644, 648-650) | 8 | |
| 14 | 652 | **WAR `s_barrier`** | 1 | none |
| 15 | 656-665 | `vmcnt(4)` A(j+2); `vmcnt(2)`, `vmcnt(0)` B(j+2); then 6 `ds_write` | 80 B/lane | `vmcnt(4)`, `(2)`, `(0)` |

Per trip:

- 10 `buffer_load_dwordx4`, 16 `ds_read2_b64`, 4 `ds_write_b64` + 8 `ds_write2st64_b64`, 16 MFMA, **4 `s_barrier`**, 16 `s_waitcnt`.
- 113 instructions in total (counted over 489-669). That is about 57 per K tile per wave.

**Loads in flight.** At most 6 VMEM instructions are outstanding per wave (the `vmcnt(5)` at 592 covers A(j+1), 4 x B(j+1) and A(j+2)).
Per program that is **16 KiB of B plus up to 0.5 KiB of A**, only one K tile's worth.

The crucial point is that **B(j+1) is issued after the reads and five MFMAs of tile j, and waited in the same half-trip.** The A load
runs about one half-trip ahead, and B runs zero tiles ahead. Nothing overlaps a B load except 3 MFMAs (48 clk) and the WAR barrier.

### 3.4 Tail and epilogue ([G] 682-803; [D] 677-813)

1. **Final RAW barrier** at `.LBB0_10` (686-687).
2. 8 `ds_read2_b64` and 8 MFMA of the last tile (698-743).
3. fp32 to bf16 with round-to-nearest-even and NaN handling (753-801).
4. One `buffer_store_dwordx2` (803). Only the 16 lanes with `tid & 15 == 0` are unmasked: the MFMA operands are A-from-B and
   B-from-A (516: first source from the B region), so D = C^T, lane = M row, and each lane holds 4 consecutive n.
5. **down only:** `buffer_load_dword topk_weights[offs_token]` (**[D]** 758), then `s_waitcnt vmcnt(0)` (**[D]** 783), then
   `v_pk_mul_f32` (784-785). This is one more exposed memory round trip after the last MFMA (**[S]** 593-599).

**Barriers executed per program:**

- 1 (peeled RAW) + 1 (peeled WAR) + 4 x trips + 1 (final RAW) = **2 x ntiles - 1**.
- gate_up: **31** (16 RAW + 15 WAR). down: **11** (6 RAW + 5 WAR).

### 3.5 Per-K-step budget and the critical path

Per K tile, per program (4 waves on 4 SIMDs):

| Resource | Amount | Time at peak |
|---|---|---|
| VMEM | 4 KiB B per wave (16 KiB per program) + 256 B A | device-wide 3 MiB (gate_up) / 4 MiB (down) per step |
| LDS write | 20 KiB | 160 clk at 128 B/clk/CU |
| LDS read | 32 KiB (8 KiB per wave, half of it A) | 256 clk |
| MFMA | 8 per wave, serial chain on v[0:3] (0 extra waits for same-vDst SrcC, **[ISA]** 7.5) | 128 clk per SIMD |
| VALU/SALU | about 25 per wave | ~100 clk |
| Barriers | 2 | |

**Issue time** is therefore about 420 clk (0.21 us) per step, bound by LDS. That is well below the memory time, but it is **not
overlapped** with it. The per-step critical path is:

`[B(j+1) in flight: L] -> vmcnt -> ds_write (tile j+1) -> lgkmcnt(0) -> RAW barrier -> 8 ds_read2 -> lgkmcnt(0) -> 5 MFMA -> issue B(j+2) -> [L] ...`

Everything between "B returns" and "next B issued" is exposed. Call that **g**:

| Part of g | Estimate |
|---|---|
| B ds_write drain (16 KiB, ~128 clk + latency) | ~200 clk |
| RAW barrier | ~50 clk |
| All 32 KiB of reads (the 5th MFMA needs `lgkmcnt(0)` at 540) | ~350 clk |
| 2 MFMA + VALU | ~50 clk |
| **g** | **~650 clk ≈ 0.33 us** |

The WAR barrier and the last 3 MFMAs are hidden under L.

## 4. ETX's imported form ([E], [M], [H])

### 4.1 Placement

- `wrap_fused_moe_up` is a 1024-thread workgroup (`__launch_bounds__(1024,1)`). Threads 0-255 (slot 0) run the program and waves 4-15
  return (**[W]** 89-94; **[E]** 45051-45053).
- The unfused megakernel does the same (`if (slot >= 1) return;`, imp_tiles.hip:578). One program per workgroup, one workgroup per CU,
  192 / 256 workgroups, the same as vLLM.
- LDS is 32,896 B: a 32 KiB arena plus 16 count words at 0x8000 and 16 generation words at 0x8040 (**[E]** 46137, 45351-45353).
  The kernel's own buffer is 20 KiB.
- 97 VGPRs vs 78 (**[E]** 46155 vs **[G]** 953). This is irrelevant at one workgroup per CU.
- One extra hardware `s_barrier`: `etx_import_init` zeroes the barrier words, then `__syncthreads` (**[E]** 45041, 45046). It runs once
  per workgroup, before any load.

### 4.2 Every s_barrier became a slot barrier: 31 (up) / 11 (down) per program

| Site | [G] original | [E] up (`ds_add_rtn` / poll loop / `; wave barrier`) | [E] down |
|---|---|---|---|
| peeled RAW | 370-371 | 45345-45403 (45353 / 45368 / 45402) | 46539 |
| peeled WAR | 457 | 45453-45502 (45460 / 45475 / 45501) | 46646 |
| loop RAW 1 | 504-505 | 45547-45588 (45553 / 45567 / 45587) | 46739 |
| loop WAR 1 | 576 | 45621-45669 (45626 / 45640 / 45668) | 46812 |
| loop RAW 2 | 603-604 | 45680-45721 (45686 / 45700 / 45720) | 46872 |
| loop WAR 2 | 652 | 45759-45801 (45764 / 45778 / 45800) | 46950 |
| odd-tail RAW / WAR (not taken) | 852-853 / 911 | 45834-45882 / 45924-45971 | 47028 / 47117 |
| final RAW | 686-687 | 45981-46031 (45990 / 46005 / 46030) | 47176 |

The loop is `.LBB12_28` (45535-45813) with 7 trips for up and `.LBB13_28` (46721ff) with 2 trips for down. Dynamic count:
**2 + 4 x 7 + 1 = 31 slot barriers for up, 2 + 4 x 2 + 1 = 11 for down.** The same count is inlined in **[M]** (for example
`ds_add_rtn` at 148884, 148991, 149084, 149157, 149217, 149296; `s_sleep 1` at 148900, 149007, 149100, ...).

### 4.3 Instruction cost of one slot barrier

The source is **[H]** 26-39. All of it runs on lane 0 of each wave under an exec mask; the other lanes idle.

**A wave that is not last** (loop site 45547-45588):

| Step | [E] lines | Instructions |
|---|---|---|
| release fence | 45547 | `s_waitcnt lgkmcnt(0)` |
| enter lane-0 path | | `s_and_saveexec`, `s_cbranch_execz` |
| read gen | 45551-45552 | `ds_read_b32` gen, wait |
| arrive | 45553-45554 | **`ds_add_rtn_u32`** count, wait |
| test | 45555-45558 | `v_cmp_ne 3`, saveexec/xor/branch |
| first poll | 45560-45564 | `ds_read_b32` gen, wait, `v_cmp_eq`, saveexec, branch |
| poll loop, per iteration | 45569-45575 | `s_sleep 1`, `ds_read_b32`, `s_waitcnt lgkmcnt(0)`, `v_cmp_ne`, `s_or`, `s_andn2 exec`, `s_cbranch_execnz` (7 instructions) |
| exit | 45577-45586 | 3-4 exec-restore instructions |
| acquire fence | 45588 | `s_waitcnt lgkmcnt(0)` |

Total: **about 21 + 7p instructions and 3 + p serialized LDS round trips** (p = poll iterations).

**The last arriver:** `ds_read` gen, `ds_add_rtn`, test, `ds_write_b32` count = 0 (45582), `s_waitcnt lgkmcnt(0)` (45583), then
`ds_add_u32` gen += 1 (45584). That is **about 18 instructions and 4 LDS ops, 3 of them waited serially**.

The original is 1-2 instructions (`s_waitcnt lgkmcnt(0)` plus `s_barrier`), costing 161 clk per round when all waves arrive together
(**[hw]**).

**Latency after the last wave arrives.** Assume an LDS round trip of about 100 clk (issue to satisfied `lgkmcnt`):

| Part | Estimate |
|---|---|
| Last arriver's 3 waited LDS ops plus the gen update (the 4th) becoming visible | ~360 clk |
| Remaining `s_sleep 1` (0-64 clk, **[ISA]**), average | ~32 clk |
| Poller's `ds_read` + wait | ~100 clk |
| Exec restore | ~20 clk |
| **Total** | **~520 clk ≈ 0.26 us** |

That is about 0.18 us above the 161-clk hardware barrier. Uncertainty: plus or minus 50% with the LDS round trip (70-150 clk).

### 4.4 Do the slot barrier's fences add `vmcnt(0)`? No.

`.amdhsa_tg_split 0` holds for both kernels (**[G]** 965, **[E]** 46166). With it, the workgroup-scope release/acquire fences in
**[H]** 27 and 38 lower to `s_waitcnt lgkmcnt(0)` only. Every `vmcnt` in the two loops, side by side:

| Place | vLLM [G] | ETX [E] up | ETX [M] |
|---|---|---|---|
| peeled, before ds_write | `vmcnt(4)` 461, `(2)` 465, `(0)` 467 | `vmcnt(4) lgkmcnt(0)` 45502, `(2)` 45505, `(0)` 45507 | 149033-149038 |
| loop half 1, before ds_write | `vmcnt(5)` 592, `(3)` 596, `(1)` 598 | `vmcnt(5)` 45671, `(3)` 45674, `(1)` 45676 | 149202-149207 |
| loop half 2, before ds_write | `vmcnt(4)` 656, `(2)` 660, `(0)` 662 | `vmcnt(4) lgkmcnt(0)` 45801, `(2)` 45804, `(0)` 45806 | 149333-149338 |
| at RAW barriers | `lgkmcnt(0)` 504, 603 | `lgkmcnt(0)` 45547, 45680 (release) + 45588, 45721 (acquire) | |
| at WAR barriers | none (576, 652) | none before; acquire `lgkmcnt(0)` 45669, and merged into 45801 | |

- The **only** additions are `lgkmcnt(0)` from the acquire side. At the three WAR sites it merges into the already existing
  `vmcnt(4)`, which becomes `vmcnt(4) lgkmcnt(0)`, with the vmcnt value unchanged. No LDS op is outstanding there, so it costs nothing.
- Loads keep their places relative to the barriers: B loads before the WAR barrier (45615-45619, 45752-45756), and the A(j+2) load
  right after it (45670), as in the original (550-560, 576-577).
- The ETX-only `vmcnt(0)` waits are all in the prologue, before any tile load: 45164 and 45184 in **[E]**; 148585, 148592, 148697 and
  148719 in **[M]**. **The slot barrier does not drain the loads the pipeline keeps in flight.**

### 4.5 The other structural difference: a longer prologue chain

- **vLLM:** kernargs are preloaded in SGPRs, so there are 2 dependent round trips before tile 0: ntpp (123 to 131) and expert id
  (146 to 163).
- **ETX [E]:** 5 round trips: kernarg `s_load` (45055-45060), arg-block `s_load` (45061-45067), ntpp `global_load` (45069, wait 45164),
  arg `s_load` (45169, wait 45174), expert `global_load` (45176, wait 45184).
- **ETX [M]:** 6 round trips: task descriptor `global_load` (148580/148585), `flat_load` (148591/148592), `s_load` args (148597/148602),
  ntpp (148604/148697), `s_load` (148702/148707), expert (148709/148719). Tile 0 then loads at 148822-148831.

At 0.1-0.35 us per round trip, **P_etx ≈ 0.4-1.4 us** once per program.

## 5. Cost model

### 5.1 Model (per program; the launch time is the time of the slowest program)

`T = F + n_tiles x (g + t_B) [+ t_w for down]`, where:

- **n_tiles = K/128**: 16 for gate_up, 6 for down (§3.3).
- **g ≈ 0.33 us**: the exposed part of each step (§3.5).
- **t_B = max(L(ρ), bytes_step / BW)**: the B load, one tile per CU in flight.
  - L(ρ) is the HBM latency under load: 0.34 us idle, rising with utilization ρ.
  - bytes_step is 3.0 MiB (up) / 4.0 MiB (down), and BW ≈ 4.3 TB/s, so the bandwidth floor per step is 0.73 us / 0.98 us.
- **F**: launch and drain, the 2 prologue round trips, about 250 prologue instructions, last-tile compute, convert and store.
- **t_w**: down's topk-weight round trip (§3.4), about 0.35 us.

The form `F + trips x max(memory, issue)` does not describe this kernel. Its issue work (~0.21 us per step) is serialized with, not
overlapped with, its loads (§3.3), so memory and issue add rather than max. The max form predicts gate_up ≈ 3.7 + 16 x 0.58 = 13 us,
against a measured median of 17.8.

### 5.2 Measured baseline and decomposition (vLLM)

Distributions from **[T]**, n = 3120 each:

| Kernel | min | p1 | p5 | p50 | p90 |
|---|---|---|---|---|---|
| gate_up | 14.15 | 15.11 | 17.44 | **17.80** | 18.20 us |
| down | 9.06 | 9.34 | 9.58 | **9.86** | 10.06 us |

The 14.2 us gate_up minimum is below the 1st percentile. It implies 50.3 MB in (14.2 - 3.7) us ≈ 4.8 TB/s of weight streaming, above the
achievable HBM rate, so it must be partly MALL-served. It is **not the steady state**, and the model is fitted to the medians.

**Down is bandwidth-bound:**

- step_dn = max(g + L, 4.19 MB / 4.3 TB/s) ≈ 0.98 us.
- So F + t_w = 9.86 - 6 x 0.98 = 4.0 us, which matches the 4.0-4.5 us medians of the one-round-trip kernels in the same trace
  (act_and_mul 4.45, moe_sum 4.33, rms_norm 4.05-4.17).

**Gate_up is latency-bound:**

- F = 4.0 - t_w ≈ 3.65 us.
- step_up = (17.80 - 3.65) / 16 = **0.88 us**, which is 3.57 TB/s during the loop, below the 4.3 floor.
- So L_up = 0.88 - g ≈ **0.55 us** (the idle 0.34 us plus queueing at ~83% utilization).

| Term | gate_up | down |
|---|---|---|
| F (+ t_w) | 3.65 | 3.65 + 0.35 |
| n x g (exposed LDS / barrier / MFMA) | 16 x 0.33 = 5.3 | 6 x 0.33 = 2.0 |
| n x (t_B not hidden) | 16 x 0.55 = 8.8 | 6 x 0.65 = 3.9 |
| **Model / measured median** | **17.8 / 17.80** | **9.9 / 9.86** |

g takes 30% of gate_up. HBM is busy only about 60% of each gate_up step, because one CU holds just one 16 KiB tile in flight and
refills it only after the LDS round trip.

### 5.3 The imported form

Measured: ETX unfused gate_up 23.1 / 24.4 us (fastest / median), down 12.1 / 12.7 (measured.json `etx_unf`: 24.416, 12.709). The
excess over vLLM is 6.6 us (up) and 2.8 us (down) on medians.

**(a) The per-barrier division as asked** (all barriers exposed, no prologue delta):

| | Fastest | Median |
|---|---|---|
| gate_up | (23.1 - 14.2) / 31 = **0.287 us** | (24.4 - 17.8) / 31 = **0.213 us** |
| down | (12.1 - 9.5) / 11 = **0.236 us** | (12.7 - 9.9) / 11 = **0.255 us** |

The fastest-pair value is inflated by vLLM's MALL-served 14.2 us outlier (§5.2).

**(b) A two-kernel fit separating the one-time prologue P from a per-step cost Δ:**

- `6.6 = P + 16Δ` and `2.8 = P + 6Δ` give **Δ = 0.38 us per K step, P = 0.52 us** (P matches §4.5's 0.4-1.4 us).
- The same fit on the fastest times gives P = -1.2 us, which is impossible. This is another sign the 14.2 sample is an outlier.

**(c) Mechanism.** Per step there is one RAW and one WAR slot barrier.

- The **WAR** one runs while B(j+1) is in flight (issued at 45615-45619, before the barrier at 45621). At ~0.26 us it is shorter than
  L ≈ 0.55 us, so it should be **hidden**, as in vLLM.
- The **RAW** one is on the critical path. §4.3 estimates it at about 0.18-0.24 us above `s_barrier`. That explains about half to
  two-thirds of the fitted Δ = 0.38 us.
- The rest (~0.14-0.2 us per step) needs one of two explanations:
  - LDS round trips of ~150 clk, not 100, for the lane-0 atomics behind the other waves' `ds_write` traffic, or
  - the WAR barrier being partly exposed.
- The prologue and the vmcnt pattern (§4.4) are ruled out as per-step causes.

**Model for the imported form:**
`T_etx = T_vllm + P_etx + n_tiles x Δ`, with P_etx ≈ 0.5 us and Δ ≈ 0.38 us per K step. Attribute Δ to the RAW slot barrier
(≈0.2 us by construction) plus 0.1-0.2 us that is not yet attributed. This reproduces 24.4 us (up) and 12.7 us (down).

**Deciding microbenchmark (one GPU session, on the `imp_wrap` harness):**

1. **Same kernel, `etx_imp_slot_barrier` replaced by `__builtin_amdgcn_s_barrier`.** This is legal there, because the 12 idle waves have
   exited. It measures Δ directly.
2. **As (1), but only the RAW sites replaced.** It splits Δ into its RAW and WAR shares.
3. **A prologue-only variant** that returns after the expert-id load. It measures P.

Candidate fixes, each measurable:

- A single `ds_add_rtn` on a combined count/generation word, which removes the first `ds_read`: 5 serialized LDS ops become 3.
- The last arriver issues `s_wakeup` (**[ISA]** §S_WAKEUP) so pollers can sleep without the 0-64 clk granularity.
- An LDS double buffer (2 x 20 KiB), which halves the barriers to 1 per step.

## 6. What the ETX performance model should carry

1. **Do not model this kernel as a bandwidth-bound stream.**
   - gate_up is per-CU latency-bound: 0.88 us per 16 KiB tile, one tile in flight, and g ≈ 0.33 us exposed per step.
   - down reaches the HBM floor only because 256 CUs x 16 KiB per step saturates it.
   - The kernel parameters are n_tiles (K/128), bytes per step (programs x 16 KiB), g, L(ρ), F, and t_w.
2. **The barrier count is 2 x n_tiles - 1 per program**: 31 / 11. Only the n_tiles RAW barriers are on the critical path. When the
   barrier implementation changes, charge Δ per RAW barrier (fitted 0.38 us per step today) and P once per program.
3. **Moving the B issue before the RAW barrier and `ds_read` gains more than a faster barrier.** Issuing B(j+2) during step j, a second
   register stage of 4 more dwordx4 per lane (the VGPR budget allows it at one workgroup per CU), would overlap g with L. That would
   take gate_up toward max(g, L) x 16 + F ≈ 12.5 us. This is a scheduling change to the imported kernel; the rules say to measure
   before building.
4. **M = 16 padding costs no DRAM bytes, but it costs ~36% of LDS traffic.** About 0.06 us of g per step, roughly 1 us per gate_up
   launch, is zero A rows read by every wave.
