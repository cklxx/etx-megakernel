# Spec: the day-0 shape set, 2026-10-11

Repository `45f261b`. vLLM sources: `~/code/vllm/` (`f46a9dfe2`); `SG` = `csrc/rocm/skinny_gemms.cu`, `AT` = `csrc/rocm/attention.cu`.
`$PY` = `~/code/etx/.venv/bin/python`. Configs: `build/unseen/<model>/hf/config.json`. This spec changes no code and makes no performance claim.
Labels: [measured] = a command ran today; [simulated] = the widened registry was built in memory by a probe script; no file changed; [source] = read from code.

## 1. The question and the answer

The planner refuses five of the six unseen configs. Two refusals come from predicates that are narrower than the kernels. Three are correct.
- Attention: each imported vLLM attention binary is one template instance, with G and HD fixed. vLLM's launcher has instances for every G from 1 to 16. The registry declares three of them.
- Down projection: the `K <= 16384` limit comes from the importer's 32 KB LDS cap. vLLM's kernel holds up to 32768 elements.
- MoE: every MoE variant is fixed by its kernel. A wider predicate would be a silently wrong run.

## 2. Phase table

| phase | variant today | current predicate | the kernel's own constraint | widened predicate, or leave | deciding parameter |
|---|---|---|---|---|---|
| `attn` | `attn_mfma4_g4`, `attn_mfma16_g6`, `attn_mfma16_g8` (vLLM `paged_attention_ll4mi_QKV_*`) | `HD == 128 and G == g`, g in {4, 6, 8} (`variants.py:302-303`) | G and HEAD_SIZE are template constants (`AT:319`, `:371`, `:409`; mfma4 `:919-920`, `:966`). One workgroup has 4 waves of 4 rows, so it holds at most 16 q heads (`AT:341`, `:442`). The launcher has instances for G 1-4 as mfma4 and 5-16 as mfma16 (`AT:3295-3346`), and for HEAD_SIZE 64 and 128 only (`AT:3620-3629`). vLLM's own gate is the same: `vllm/platforms/rocm.py:383-392`. | One import per g in 1..16: `HD == 128 and G == g`. mfma4 for g <= 4, mfma16 for g >= 5, as the launcher does. Leave HD 64: the kernel supports it, but `attn_reduce` and `rms_norm_3d` are compiled for 128 (`spec.py:54-55`, `:63`, `:254`, `:258-259`), and no config here needs it. | G, HD |
| `attn` | `llm_attn` (generic) | `HD % 8 == 0 and HD <= 192 and G <= 16` (`variants.py:304-305`) | The vLLM host cannot launch it (section 4). | Leave. The planner can select it, and `model.py:268-270` then refuses loudly. | ABI |
| `down` and every unfolded GEMV phase | `dsl_gemv` | `K % 256 == 0 and K <= 4096` (`variants.py:251-252`; host rule `host.hip:370`) | x is staged in LDS as K_MAX = 4096 floats (`etx/ops/gemv.py:97-99`). The chunked path is written for the MoE forms only (`gemv.py:416-425`). | Leave. The kernel truly requires `K <= 4096`. A wider dense GEMV is ETX tile work, and CLAUDE.md section 1 requires the owner's request for it. | K |
| same | `wvsplitk_y1`, `wvsplitk_y2` (vLLM `wvSplitK_hf_sml_`) | `fold == 0 and K <= 16384`, split on `M <= CU * 4` (`variants.py:266-269`) | The kernel's array has 32768 bf16 elements (`SG:357`, `:383`). It loads `min(Kap*N, 32768)` elements (`SG:394-395`). Then it reads `s[k_]` for every `k_ < K` with no LDS bound (`SG:449-455`). vLLM selects `_sml_` only when `Kbp*N <= 32768` and `M % YTILE == 0` (`SG:1217`), and it requires `K % 8 == 0` (`SG:1199`). The store `C[m+y]` has no row bound (`SG:515`, `:561`). The importer sizes the slot to `lds_cap` = 32 KB (`spec.py:36`, `:39`; `ir_import.py:320-321`), so 16384 is the import's cap, not the kernel's. | New export `wvsplitk_l40_y2`: the same binary with `lds_cap = 40 * 1024`. Predicate: `fold == 0 and 16384 < K <= 20480 and K % 8 == 0 and M > CU * 4 and M % 2 == 0`. Also add `M % 2 == 0` to `wvsplitk_y2`, because an odd M writes one row past the output. | K (= INTER), M (= H), batch N = 1 (`spec.py:242`) |
| `moe_up`, `moe_down` (folded, the default MoE build) | `dsl_moe_up_fold`, `dsl_moe_down_fold` | `E <= 128`, `MI % 256 == 0`, `H % 256 == 0`, H and MI <= 65536 (`variants.py:276-283`) | Host rule `host.hip:443-444`. MI is the K of `moe_down`, and the DSL GEMV loads need `K % 256 == 0` (`gemv.py:345`, `:423`). | Leave. The tile truly requires `MI % 256 == 0`. A wider tile is ETX tile work for the owner. | MI, H, E |
| `moe_up`, `moe_down` (unfolded) | `fused_moe_up[_e8]`, `fused_moe_down[_e8]` (Triton) | exact `(H, MI, E, TOPK)` (`variants.py:290-295`) | `top_k`, `MUL_ROUTED_WEIGHT` and `BLOCK_SIZE_*` are `tl.constexpr` (`vllm/model_executor/layers/fused_moe/fused_moe.py:341-345`). The binary is taken from vLLM's cache for one model (`spec.py:128-130`) and matched on the exact shape (`launch_spec.py:93-96`). | Leave. The binary truly requires its captured shape. A new shape needs a new capture on a GPU. | H, MI, E, TOPK |
| `topk` | `dsl_topk`; `topk_softmax_e128`, `_e8` | `dsl_topk`: `E <= 128` (`variants.py:271`); imports: `E == e` (`:272-274`) | `topkGating` is a template on E (`spec.py:102-104`). | Leave. `dsl_topk` already covers E = 60. | E |
| `moe_sum` | `moe_sum_k8`, `moe_sum_k2` | `TOPK == k` (`variants.py:296-297`) | TOPK is a template constant (`spec.py:114-115`). | Leave. | TOPK |

## 3. The six configs after the widening

Today [measured, the FP command of `docs/core/SPEC-cpu-work-2026-10-11.md:9-10`]: Mistral plans 354 instances (`352f24a784c67a1a`). Phi-3.5 and OLMo-2 stop at the generic `attn` tile. Qwen3-14B and Qwen2.5-7B stop at `down`. Qwen1.5-MoE stops at `moe_up`. The simulated column uses the predicates of section 2 [simulated].

| model | after | reason | predicate that changes it |
|---|---|---|---|
| mistral-7b-v0.3 (G 4, K 14336) | plans: 354, `352f24a784c67a1a`, unchanged | G 4 and K <= 16384 select the same variants as today | none |
| phi-3.5-mini (HD 96, G 1) | still refused at `attn`, generic only | HD 96 is neither 64 nor 128 (`AT:3620-3629`; `rocm.py:387`). This refusal is correct. | none |
| olmo-2-13b (HD 128, G 1) | must stay refused. Edit 4 makes the loader refuse it. | Without edit 4, it plans 442 instances with `attn_mfma4_g1` [simulated], and that plan is wrong. OLMo-2 normalises q and k over the full projection width (`transformers/models/olmo2/modeling_olmo2.py:213-214` in the venv). It applies its norms after attention and after the MLP (`:307`, `:313`). `load_dims` sets `qk_norm` only for qwen3 (`examples/llm/model.py:120`), so the plan has no q/k norm and the wrong norm order. | the `attn` G = 1 predicate, blocked by edit 4 |
| qwen3-14b (G 5, K 17408) | plans: 402 instances [simulated] | `attn_mfma16_g5` and `wvsplitk_l40_y2` | `attn` G 5; `down` 16384 < K <= 20480 |
| qwen2.5-7b (G 7, K 18944, qkv bias) | plans: 310 instances [simulated] | `attn_mfma16_g7` and `wvsplitk_l40_y2`. The qkv bias already uses `wvsplitk_y2`'s BIAS input (`SG:502-506`). | `attn` G 7; `down` 16384 < K <= 20480 |
| qwen1.5-moe-a2.7b (E 60, TOPK 4, MI 1408) | still refused. Edit 4 makes the loader refuse it. | Today it stops at `moe_up` because 1408 % 256 = 128. That refusal is correct, but it hides a second gap: the model has a shared expert with a sigmoid gate (`vllm/model_executor/models/qwen2_moe.py:116`, `:145-161`). No phase implements that expert, and the loader accepts the model, because it refuses only `n_shared_experts` (`examples/llm/model.py:112-113`). | none |

The three known digests are unchanged in the simulation [simulated]: `qwen3-30b-a3b 434 e68e856650430532`, `qwen3-8b 362 c586df49f38d5c3d`, `mixtral-8x7b 322 14bbfbe336a339c0`. The new predicates do not overlap the old ones: G is outside {4, 6, 8}, and K is above 16384.

## 4. Can the vLLM host launch `llm_attn`?

No. That work belongs to the separate host-ABI workstream. The two ABIs differ in every input and output:
- Arguments. `llm_attn` takes only `etx_ctx` (`examples/llm/tiles.hip:175`) and reads the model from `__constant__ LlmParams P` (`tiles.hip:14`, `:176`; struct `examples/llm/llm_params.h:27-42`). The vLLM host fills a typed per-instance argument block from the import's descriptor (`examples/vllm_llm/host.hip:96-112`, `etx_imp_find` at `:98`). The attention arguments are listed in `spec.py:215`.
- Query. `llm_attn` reads fp32 `P.q` (`tiles.hip:191`). The vLLM host passes the bf16 q rows from the qkv buffer (`spec.py:215`, `qbuf`).
- KV cache. `llm_attn` reads `[CTX][NKV][HD]` (`llm_params.h:20`; `tiles.hip:196`, `:217`). The host's cache is vLLM's paged layout: key `[block, head, HD/8, BS, 8]`, value `[block, head, HD, BS]` (`spec.py:251`), written by `reshape_and_cache`.
- Output. `llm_attn` writes `attn_part[NH][NC][HD+2]` in chunks of `P.CHUNK` positions for `llm_merge` (`tiles.hip:182`, `:218`). `attn_reduce` reads `exp_sums`, `max_logits` and `tmp_out` in partitions of 256 positions (`spec.py:216`, `:254`).
An adapter cannot close these four gaps. Closing them means a new attention tile that keeps `llm_attn`'s math and uses the vLLM host's argument block, layouts and outputs, as `dsl_attnprep` does (`host.hip:392`).

## 5. Edits, file by file

1. `etx/importer/spec.py`
   - In the `attention.cu` source (`:42-57`), add one `Kernel` per new g. For g in {1, 2, 3}: export `attn_mfma4_g{g}`, spec `paged_attention_ll4mi_QKV_mfma4_kernel<{BF}, {BF}, vllm::Fp8KVCacheDataType::kAuto, {BF}, 16, 128, 256, false, {g}>`. For g in {5, 7, 9, ..., 16}: export `attn_mfma16_g{g}`, spec `paged_attention_ll4mi_QKV_mfma16_kernel<..., 16, 128, 256, false, {g}, MFMAType::F16>`. Use block `(256, 1, 1)`. Copy each `match` name from the `.ll` file that `etx/importer/vllm_kernels.py:117-120` writes. Do not type it by hand.
   - In the `skinny_gemms.cu` source (`:32-41`), add `Kernel("wvsplitk_l40_y2", <the y2 match>, <the y2 spec>, (64, 16, 1), lds_cap=40 * 1024)`.
   - `:245`: replace `g = int(e[-1])` with `g = int(e.rsplit("_g", 1)[1])`. The old code reads G 12 as 2.
   - `:240`, `:373`, `:401`: replace the literal 16384 with the export's `lds_cap // 2`. This keeps the emitted text byte-identical for `wvsplitk_y1` and `wvsplitk_y2`.
2. `etx/select/variants.py`
   - `:233`: add `"wvsplitk_l40_y2": 40960` to `IMPORTED_LDS`.
   - `:268`: change the y2 predicate to `s["fold"] == 0 and s["K"] <= 16384 and s["M"] > CU * 4 and s["M"] % 2 == 0`.
   - After `:269`, inside the loop, add `_imp("wvsplitk_l40_y2", _ph, lambda s: s["fold"] == 0 and 16384 < s["K"] <= 20480 and s["K"] % 8 == 0 and s["M"] > CU * 4 and s["M"] % 2 == 0, "SG:357,383,394-395,449-455 K-vector in LDS; SG:1199 K % 8; SG:1217 M % YTILE; lds_cap 40 KB", gemv_cost)`.
   - `:302-303`: loop over `g in range(1, 17)` with export `attn_mfma4_g{g}` if `g <= 4` else `attn_mfma16_g{g}`. Predicate: `s["HD"] == 128 and s["G"] == g`. Rule text: `"AT:3295-3346 G 1..16; AT:3620-3629 HEAD_SIZE 64/128 (128 here)"`.
3. `examples/vllm_llm/build.sh:24` and `examples/vllm_llm/rocm_build_check.sh:58`: for `1 <= G <= 16`, set `ATTN=attn_mfma$([ $G -le 4 ] && echo 4 || echo 16)_g$G`. Otherwise, exit with the existing message.
4. `examples/llm/model.py` `load_dims`:
   - Refuse `model_type == "olmo2"`. The message names the full-width q/k norm and the norms after attention and after the MLP.
   - Extend the refusal at `:112-113` to `c.get("shared_expert_intermediate_size", 0)`.

## 6. Tests and acceptance

Tests to add:
- `tests/test_select_planner.py`: for g in 1..16, set `NH = NKV * g` on Qwen3-8B dims. Then `plan(d).phases["attn"]` is the g export, imported. G 17 and HD 64 give `llm_attn`, and `M.instances` raises the generic-tile `SystemExit`.
- Rewrite `test_g5_attention_has_a_generic_plan_but_no_vllm_host_abi` (`:111`) to use HD 96. With edit 2, G 5 selects `attn_mfma16_g5`.
- `tests/test_select_planner.py`: set `INTER` on Qwen3-8B dims. K 16384 gives `wvsplitk_y2`. K 16392 and 20480 give `wvsplitk_l40_y2`. K 20488 and 17412 raise `NoLegalVariant` naming `down`. M 4097 makes `wvsplitk_y2`'s predicate false.
- `tests/test_operator_spec.py`: `OPERATORS["attn_mfma16_g12"].constants["G"] == 12`. `validate_values` rejects K 16392 for `wvsplitk_y2` and K 20488 for `wvsplitk_l40_y2`. The existing `test_host_argument_code_matches_before_change` digests stay unchanged.
- `tests/test_llm.py`: an `olmo2` config and a `qwen2_moe` config with `shared_expert_intermediate_size > 0` are refused by `load_dims`.
- Copy the six `config.json` files to `tests/fixtures/unseen/<model>/config.json`, because `build/` is untracked. Add one test that asserts the "after" column of section 3: 354 / 402 / 310 instances, or the named refusal.
Acceptance:
- The FP command (`docs/core/SPEC-cpu-work-2026-10-11.md:9-10`) prints the three known digests byte-identical, and Mistral prints `354 352f24a784c67a1a`.
- The six configs behave as section 3 says.
- `rm -rf build/cpu_check && $PY -m etx.tools.cpu_check --out build/cpu_check` reports `"cpu_core_passed": true`.
- `examples/vllm_llm/local_matrix.sh` still builds all 16. `examples/vllm_llm/rocm_build_check.sh` builds Qwen3-14B and Qwen2.5-7B. Its VGPR and spill report for each new attention instance is recorded, and the LDS total with a 40 KB arena fits 65536 B.

## 7. What this spec does not buy

It buys no GPU correctness for any widened shape. A plan that the planner accepts is not a correct run. Each new attention instance and `wvsplitk_l40_y2` still needs `host --check` byte-identity on a GPU, and each of Qwen3-14B and Qwen2.5-7B still needs 32/32 tokens equal to HF greedy. It also buys no speed. The 40 KB arena raises the LDS of every phase in those two builds, and nobody has measured that cost.

Not verified here: whether each new `mfma16` instance (up to G 16) stays within 128 VGPRs without spills, and whether the 40 KB arena plus the runtime's other `__shared__` (`etx/runtime/include/etx/import.h:22-24`, `:53`) fits 64 KB in a real gfx942 compile.
