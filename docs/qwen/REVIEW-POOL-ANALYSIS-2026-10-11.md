# Review of the 300 tok/s pool analyses, 2026-10-11

Scope: Qwen3-30B-A3B, bf16, batch 1, context 1024, one MI300X VF. Reviewed on `main` at `f61d442`: `docs/qwen/DEP-GAP-LEDGER-2026-10-11.md` (DEP), `docs/qwen/XCD-TIMELINE-2026-10-11.md` (XT), `docs/qwen/XCD-PREFETCH-2026-10-11.md` (PF), `docs/qwen/MOE-BODY-ANATOMY-2026-10-11.md` (MOE) and `docs/qwen/PLAN-300TOKS-2026-10-11.md` (PLAN). CPU only: no GPU ran, nothing was provisioned, no network spend. Labels: **[M]** measured with `file:line`, **[E]** estimate with arithmetic, **[U]** unknown (adds nothing to any total). `results_*` paths are under `~/code/etx/`, outside git; `R/` means `results_qwen300_2026-10-11/`. `RB` means this review's `docs/review-pool-analysis-2026-10-11/rebase.txt`, `ACC` its `account_on.txt`.

## 0. What this review re-ran

- `examples.vllm_llm.xcd_account` on `R/tr_off.bin` reproduces `docs/dep-gap-ledger-2026-10-11/account.txt` **byte for byte**. The ledger's arithmetic is not in question; its interpretation is.
- `R/tr_on.bin` (sha256 `aa684682...`) is the trace of the **KEY32-on build that the plan now keeps** (build `3288aa5c`, `R/tr_on.log:3`): 32/32, span **4067.3 us [M]** (`R/tr_on.log:25-26`). None of the four documents analysed it. Both `tr_off.bin` and `tr_on.bin` match the borrowed `b_all/plan.json` row for row: type, coordinates and worker, **0 of 60,362 mismatches [M]** (`RB:1-2`). Its account is `ACC`.
- `rebase.py` applies the plan's own lever rule (B = 48 x (critical body - round11 serial floor), `docs/core/PLAN-generic-decoder-2026-10-08.md:154`) to three traces. On `tr_all` it reproduces every plan allowance exactly: 0.086/0.112/0.125/0.123/0.168/0.324/0.139/0.019 ms (`RB:5-13`, against `docs/core/PLAN-generic-decoder-2026-10-08.md:224-230`). That rule says a KEEP must be followed by "rebase every bound on the new trace" (`:154`). Nobody did this after KEY32. `RB` does it.

## 1. Verdicts

**1. KEY32 is a measured KEEP.**
VERDICT 1 HOLDS
Decided by: `R/job2_run4.log:138`.
Evidence: the five savings are 0.229-0.245 ms, all positive, median 0.231 > 2 x 0.016; both builds certify 97/97 imports at positions 0/1040 (`:47,60`); every timed run is 32/32 with the logged build identity; fused wall medians are 4.410/4.180 (`:139,144`); and the topk body fell 4.99 us/layer (`:77`), `x 48 = 0.240 ms [E]`, which matches the wall saving. Booking it on the plan's 4.471 ms historical anchor is a transfer, as PLAN:138 says, not a same-session result; the off arm here is 4.410, not 4.471.

**2. The entire critical path is now priced.**
VERDICT 2 WEAK
Decided by: `docs/qwen/DEP-GAP-LEDGER-2026-10-11.md:162`.
Evidence: 4303.5 / 871.0 / 231.2 on 127 edges reproduce exactly. But the ledger prices only the 871.0 us of gaps (20.2% of the span; the 3432.5 us of bodies remain on the [E] byte/hop model). "No unattributed remainder" is true by construction, because class (iii) is defined as the remainder ("by difference", :162). And the trace is the superseded KEY32-off build; on the kept build the gaps are 875.5 us, with 231.3 us before-taken on 123 late edges (`ACC:55,80`).

**3. Only 110.1 us is removable; 231.2 us is movable only; 529.7 us is irreducible.**
VERDICT 3 WEAK
Decided by: `docs/qwen/FLOORS-2026-10-08.md:319`, with `docs/qwen/PLAN-300TOKS-2026-10-11.md:66`.
Evidence: each of the three buckets is misclassified in part:
- **Removable.** Under the owner protocol, the four non-owner partitions still "acquire that publication" (PLAN:66). The 52.4 us prep->attn edge therefore becomes a mid-body wait on 4 of 5 partitions rather than disappearing.
- **Movable only.** BAR2's +0.095 ms came from a build that also raised the megakernel's LDS from 41,448 to 61,952 B (`docs/qwen/RESULTS-2026-10-01.md:98`), and its cause "is not known" (FLOORS:319). It is not evidence that the 231.2 us "reappears, larger" (DEP:160).
- **Irreducible.** The 529.7 us includes the 217.7 us after-take pool that the same ledger registers as an open touch question (DEP:260-261).

**4. The bandwidth denominator had to be per phase.**
VERDICT 4 REFUTED
Decided by: `docs/xcd-timeline-2026-10-11/latency.txt:1`, against `docs/qwen/XCD-TIMELINE-2026-10-11.md:281`.
Evidence: "XCDs live" is a time average over each grid's body span (`latency.txt:1`). qkv, o_proj, moe_up and moe_down place tasks on XCD 0-7 (XT:46,50,53,54), so the premise "no phase runs on all eight XCDs" is false. Dividing by 7.23-7.67 credits each streaming phase's own ramp and tail skew to its ceiling, which is part of the inefficiency being measured. Only attn's number moves materially, and XT:301 itself calls the narrow-phase shares upper bounds.

**5. The device has as much idle time as the whole remaining gap.**
VERDICT 5 WEAK
Decided by: `docs/qwen/XCD-TIMELINE-2026-10-11.md:322`, with `docs/dep-gap-ledger-2026-10-11/account.txt:40`.
Evidence: the arithmetic holds, and the idle time grows on newer builds: 1194.0 us on tr_all, 1211.3 on tr_off, 1216.7 on tr_on (`RB:17-19`). But idle XCD time is capacity off a 436-task serial critical path that sums exactly to the span (account.txt:40); no step time can be subtracted from it. The comparison also sets tr_all device time against wall gaps of two other builds. Those wall gaps include 0.131-0.135 ms of host time (`R/job2_run4.log:139-140,144-145`) that idle device capacity cannot absorb. "Idle" is also definition-dependent: 27% of XCD time, but about 60% of worker time (XT:324).

**6. Prefetch and XCD switching cannot close the gap.**
VERDICT 6 WEAK
Decided by: `results_2026-09-27/vlnext/overlap.log:11-16`.
Evidence: three of the four supporting reasons are misread, and the bound's arithmetic overstates it:
- **The doubling.** 1.52 -> 3.08 us is at a 4.31 TB/s background (:15-16). At the 0.67-1.71 TB/s the prefetches would add, the same bench gives 3.24 -> 3.32-3.36 us, +2.5-3.7% (:11-13).
- **Level-independent latency.** The cited figure is loaded, with 303 CUs streaming (MODEL:321). The latency phases run lightly loaded, where an idle round trip costs 0.143/0.261/0.412 us from L2/MALL/HBM (`docs/qwen/MODEL-vllm-kernels.md:320`).
- **86%.** That is the MALL occupancy *with* a prefetched layer added (PF:32). The KV cache alone fills `103.7/256 = 40.5%` [E].
- **The bound's arithmetic.** `702.3 x 0.172` treats a rate gain as a time saving. The time saving is `1 - 3.61/4.23 = 14.7%`, so the upper end is **102.9 us**; the central value is `1 - 3.43/3.74 = 8.3%`, **58.2 us** [E].

The conclusion survives only on a cruder bound: even deleting all byte time of qkv, o_proj and moe_down is `702.3 us < 846.7 us` [E].

**7. The MoE body cannot carry the gap either.**
VERDICT 7 HOLDS
Decided by: `docs/qwen/MOE-BODY-ANATOMY-2026-10-11.md:111`, with `RB:12-13`.
Evidence: the qualified pool is 0, and two measurements the anatomy did not use make the conclusion firmer than its 14-22x ratio, which compares the need with a nominal service price that MOE:123 says is "not ... an upper bound":
- Rebased on tr_on, the plan's MoE ceiling is **0.075 ms** (moe_up). moe_down's critical body, 9.069 us, is *below* its round11 floor of 9.48 (`RB:12-13`).
- AITER CK MoE costs 30.72 us/layer (`R/job1_pricing.log:95`), against ETX's `(770.3 + 435.4)/48 = 25.12 us/layer` of critical bodies (`ACC:42,44`) [E].

**8. The ledger trap.**
VERDICT 8 HOLDS
Decided by: `docs/qwen/PLAN-300TOKS-2026-10-11.md:148`.
Evidence: the 0.324 ms threshold must come "outside the rows already priced". A replacement MoE row therefore needs `0.158 + 0.323667 = 0.481667 ms` in total, and the arithmetic is right. The 0.158 underneath it, however, is a tr_all ceiling that is 0.075 on the kept build (`RB:12-13`), so 0.396-0.482 ms are historical-ledger figures, not same-session ones.

**9. 300 tok/s is not reachable on the present evidence; no priced pool is within 3x.**
VERDICT 9 WEAK
Decided by: `docs/qwen/PLAN-300TOKS-2026-10-11.md:129-132`, with `RB:5-13`.
Evidence: the verdict is right for a partly wrong reason. The 0.254 ms sum adds three sub-pools that each sit inside an existing plan row: fusion (:129), MoE skeleton/stream (:132) and layout/streaming (:131-132). Its 0.121 term is overstated (claim 6). It also omits the rows nobody priced today. Rebased on the kept build, the plan's non-KEY32 rows sum to `0.049 + 0.096 + 0.126 + (0.073 + 0.193) + 0.075 = 0.612 ms`. Adding the uncredited top-k residual of 0.093 gives 0.705 ms, still `0.142 ms` short of 0.847 [E]. Only the conditional attention+reduce ceiling of 0.236 (`RB:7-8`) closes the paper gap. The "3x" clause holds narrowly: the need is `0.847/0.266 = 3.18` times the largest row, qkv+router [E].

## 2. Answers

### A. One basis or several?

There are not three bases but six. Four are traces:
- **B1:** `results_tiles/tr_all.bin`, another build and session, 4412.1 us. Used by XT, PF and every plan allowance.
- **B2:** `R/attempt1/tr_off.log`, build `cd88dda`, log only, 4295.4 us. Used by MOE.
- **B3:** `R/tr_off.bin`, build `cd88dda`, 4303.5 us. Used by DEP.
- **B4:** `R/tr_on.bin`, build `3288aa5c`, 4067.3 us. Used by none.

Two are wall-time anchors:
- **W1:** the never-traced vlbest2 anchor, 4.471 ms. It gives the 1138 us gap and the 0.324/0.238 ms shortfalls.
- **W2:** the untraced KEY32-on arm, 4.180 ms. It gives 0.847 ms.

B2 and B3 are the same build identity (`R/tr_off.log:3`; MOE:13), so mixing them is harmless: the body sums differ by 0.2% (MOE:278). DEP's numbers also transfer to B4 within 0.5%: gaps 871.0 -> 875.5, attention edges 110.1 -> 109.1, before-taken 231.2 -> 231.3 (`ACC:55,69-70,80`). PF's byte bound is build-independent.

Two load-bearing conclusions do mix bases:
1. **Claim 9 and the plan ledger.** Today's pools are compared with W2, but the rows they sit in were derived on B1 against W1. On B4, the MoE row loses 0.083 ms, router gains 0.025, prep gains 0.014, and a 0.093 top-k residual appears (`RB:5-13`). The verdict survives only after this rebase. PLAN:136's "0.238 ms unpriced" is a B1/W1 figure.
2. **XT section 8.2's headline**, which sets B1 device idle against W1 and W2 wall gaps (claim 5).

Separately, DEP:26-27 is wrong that "no untraced run of this build exists". Five untraced off rounds of the same identity are at `R/job2_run4.log:139-140`. They give the trace overhead as `4.299 - 4.279 = 0.020 ms` device (0.47%) [E], not the 0.008 that DEP transferred.

### B. Unpriced pools: excluded or open

| Phase or lever | Status on the kept build (B4) | Excluded or open |
|---|---|---|
| router body | 0.193 ms ceiling, the largest single row; it grew from 0.168 (`RB:10`) | open; nobody priced it today |
| attn + reduce body (imported) | 0.117 + 0.119 ms ceilings (`RB:7-8`) | open, and the only row that closes the paper gap (claim 9) |
| vendor attention branch | S3's AITER pair is 6.77 + 4.65 = 11.42 us/layer (`R/job1_pricing.log:47-94`, mean [E]), against 5.360 + 5.285 = 10.65 us/layer critical bodies of ETX's imported `paged_attention_ll4mi` pair (`RB:7-8`; `docs/qwen/FLOORS-2026-10-08.md:44-45`) | AITER unified attention excluded on a cross-basis comparison [E]; other backends [U] |
| qkv folded body | 0.073 ms (`RB:5`) | open |
| prep body | 0.126 ms; KEY32 added 0.22 us/task (`R/job2_run4.log:77`) | open |
| topk after KEY32 | 0.093 ms residual (`RB:11`); PLAN:128 credits "selection only" | open, uncredited |
| MoE tile (ETX) | 0.075 ms (`RB:12-13`) | open but small; a new mapping needs a new floor (PLAN:99) |
| vendor MoE (CK) | 30.72 vs 25.12 us/layer | excluded [M vs M, cross-basis] |
| o_proj, lm_head | 0.008 ms (`RB:9`); lm_head is 7.2 us above its byte floor (`ACC:51`) | excluded by the 0.030 ms rule |
| host side | same-session wall minus device is 0.135 ms (`R/job2_run4.log:144-145`); PLAN's row is 0.049 from an older profile | open; 0.086 ms of it is unexplained [U] |
| touch on busy edges | 217.7 us stake (DEP:261) | open [U] |
| schedule/boundary-count redesign | DEP's (iii) is "for this schedule" only; `etx/sim/xcd.py` can test it on CPU (DEP:294) | open, unpriced |
| vLLM-configuration branch | same-session S3 wall is 4.967 ms (`R/job2_run4.log:149-191`); it needs `1.634 ms = 32.9%` [E] | not excluded; the untested switches are [U] |

### C. A single load-bearing figure

Within claim 9's own arithmetic, no single figure can flip the verdict. One pool would have to grow by `0.847 - 0.254 = 0.593 ms`. The nearest candidate, the prefetch bound, would have to reach `0.847 - 0.110 - 0.023 = 0.714 ms`. That is 6.9 times its corrected 102.9 us upper end (5.9 times the stated 120.8), and more than the entire 702.3 us of byte time in qkv, o_proj and moe_down [E]. The 0.847 ms need cannot be wrong by that much: the on-arm rounds span 0.011 ms (`R/job2_run4.log:144`).

The figure that actually decides reachability is the **attention+reduce ceiling, 0.236 ms** [E] (`RB:7-8`), resting on the round11 attn/reduce floors of 2.92/2.81 us (`docs/qwen/FLOORS-2026-10-08.md:241-242`). If the attainable correct attention+reduce cut is below **0.142 ms** (2.96 us/layer, or 28% of their 10.645 us/layer critical bodies, `RB:7-8`) [E], 300 tok/s is out of reach even with every other row at its ceiling. The figure would have to be overstated by `0.236 - 0.142 = 0.094 ms` (40%) to flip the paper envelope from "reachable" to "not". The floors omit LDS and MFMA work (PLAN:23), so an overstatement of that size is plausible, not exotic.

### D. Strongest surviving route

None has evidence. The strongest paper route needs four conditions at once:
- every rebased same-session ceiling at its full value: host 0.049, boundary 0.096, prep 0.126, qkv+router 0.266, MoE 0.075 and the top-k residual 0.093, together **0.705 ms**;
- a correct, certified attention+reduce change worth at least **0.142 ms** out of its 0.236 ceiling;
- `0.941 ms >= 0.847`, which means every row at **90%** of a ceiling that, by its own definition (PLAN:23,33), is generous rather than attainable;
- no attention implementation measured so far is faster: S3's AITER pair is not.

The measurement that decides it: price candidate attention+reduce kernels per layer at this shape (batch 1, context 1024, 4 KV heads) on the comparable critical-body basis, with the existing Job 1 harness (`examples/sessions/qwen300.py:276-300`). Pre-register **<= 7.69 us/layer** (`10.645 - 2.96`) [E]. If nothing legal reaches it, close the 300 tok/s target for this architecture.

## 3. Defects found, for the authors (not fixed here)

1. DEP:26-27 claims no untraced run exists, and DEP:44 attributes the 16:00 file to `job2_rerun.log:68-70`. Both the file and the untraced rounds are in `R/job2_run4.log:69-71,139-147`.
2. DEP:159-160 reads BAR2 as proof that the cost reappears; FLOORS:319 records its cause as unknown.
3. XT:281's premise is false (claim 4). XT section 8.2's comparison mixes three bases (claim 5).
4. PF:69-72 uses a rate gain as a time saving (102.9/58.2 us, not 120.8/63.2). PF:76 and PF:121 apply a 4.31 TB/s interference figure to 0.67-2.0 TB/s prefetches. PF:45 and PF:123 apply loaded latency to lightly loaded phases.
5. No document followed the lever rule's "rebase every bound on the new trace" after KEY32's KEEP (`docs/core/PLAN-generic-decoder-2026-10-08.md:154`). PLAN:130-132 still carry tr_all allowances.
6. The round11 "probe" floor is not a floor for moe_down: the critical body is 9.069 us against 9.48 (`RB:13`). Ceilings derived from it are not bounds.

## 4. What this review could not determine

- Whether any legal attention+reduce implementation beats today's imported pair by 2.96 us/layer. The AITER comparison here mixes stand-alone kernel time with critical task bodies [E].
- How the 875.5 us of gaps divides into arrive, visibility, poll, touch and acquire. The trace has no intra-gap stamps (DEP:232-236).
- Whether the qkv, o_proj and moe_up "probe" floors are real floors, given that moe_down's is not.
- What makes up the 0.135 ms host excess on the kept build. No current host profile exists.
- Whether the single tr_on observation is representative. It is one traced run, inflated by about 0.022 ms over the untraced 4.045 ms (`R/tr_on.log:466`; `R/job2_run4.log:145`).
- MALL eviction and NT-hit behaviour for any prefetch (PF:129-130), and every untested vLLM switch.
- Anything requiring a GPU. Nothing here is a new performance measurement.

Reproduce: `.venv/bin/python docs/review-pool-analysis-2026-10-11/rebase.py` (needs the untracked archive). `ACC` was written by `.venv/bin/python -m examples.vllm_llm.xcd_account ~/code/etx/results_vlbest2_2026-10-10/b_all/plan.json ~/code/etx/results_qwen300_2026-10-11/tr_on.bin --bytes docs/xcd-timeline-2026-10-11/bytes.json --out <dir>`. `pytest -q tests/test_xcd_account.py tests/test_timeline.py`: 8 passed.
